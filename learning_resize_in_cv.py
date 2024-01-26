import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# import torch
import tensorflow as tf
import keras
# from keras import ops
from keras import layers
# import tensorflow as tf

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import matplotlib.pyplot as plt
import numpy as np


# define hyper parameters
inp_size = (300, 300)
target_size = (150, 150)
interpolation = "bilinear"

auto = tf.data.AUTOTUNE
batch_size = 64
epochs = 5


# load and prepare the dataset
train_ds, validation_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:40%]", "train[40%:50%]"],
    as_supervised=True,
)

def preprocess_dataset(image, label):
    image = tf.image.resize(image, (inp_size[0], inp_size[1]))
    label = tf.keras.backend.one_hot(label, 2)
    return (image, label)


train_ds = (
    train_ds.shuffle(batch_size * 100)
    .map(preprocess_dataset, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)
print(len(train_ds))

validation_ds = (
    validation_ds.map(preprocess_dataset, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)

print(len(validation_ds))

# define the learnable resizer utilites
def conv_block(x, filters, kernel_size, strides, activation=layers.LeakyReLU(0.2)):
    x = layers.Conv2D(filters, kernel_size, strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    return x


def res_block(x):
    inputs = x
    x = conv_block(x, 16, 3, 1)
    x = conv_block(x, 16, 3, 1, activation=None)
    return layers.Add()([inputs, x])

    # note user can change num_res_blocks to > 1 also if needed

def get_learnable_resizer(filters=16, num_res_blocks= 1, interpolation=interpolation):
    inputs = layers.Input(shape=[None, None, 3])
    
    # first perform naive resizing
    naive_resize = layers.Resizing(*target_size, interpolation= interpolation)(inputs)
    
    # first convolution block without with batch normalization
    x = layers.Conv2D(filters=filters, kernel_size= 7, strides =1, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)


    # second convolution block with batch normalization
    x = layers.Conv2D(filters= filters, kernel_size= 7, strides =1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    # intermediate resizing a bottleneck
    bottleneck = layers.Resizing(*target_size, interpolation=interpolation)(x)
    
    # residual passes
    # first res_block will get bottleneck output as input
    x = res_block(bottleneck)
    # remaining res blocks will get previous res_block output as input
    for _ in range(num_res_blocks - 1):
        x = res_block(x)


    # projection
    x = layers.Conv2D(filters= filters, kernel_size = 3, strides = 1, padding="same", use_bias = False)(x)

    x = layers.BatchNormalization()(x)

    # skip connection
    x = layers.Add()([bottleneck, x])

    # final resized image
    x = layers.Conv2D(filters = 3, kernel_size = 7, strides =1, padding="same", use_bias=False)(x)
    final_resize = layers.Add()([naive_resize, x])

    return keras.Model(inputs, final_resize, name="learnable_resizer")

learnable_resizer = get_learnable_resizer()

print(learnable_resizer)

# visualize the outputs of the learnable resizing module
sample_images, _ = next(iter(train_ds))
plt.figure(figsize=(16, 30))

for i, image in enumerate(sample_images[:6]):
    image = image / 255

    ax = plt.subplot(3, 4, 2 * i + 1)
    plt.title("input image")
    plt.imshow(image.numpy().squeeze())
    plt.axis("off")

    ax = plt.subplot(3, 4, 2 * i + 2)
    resized_image = learnable_resizer(image[None, ...])
    plt.title("resized image")
    plt.imshow(resized_image.numpy().squeeze())
    plt.axis("off")


# model building utility
def get_model():
    backbone = keras.applications.DenseNet121(
        weights = None,
        include_top = True,
        classes = 2,
        input_shape=((target_size[0], target_size[1], 3))
    )
    backbone.trainable = True

    inputs = layers.Input((inp_size[0], inp_size[1], 3))
    x = layers.Rescaling(scale=1.0 /255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return keras.Model(inputs, outputs)


model = get_model()
model.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), optimizer="sgd", metrics=["accuracy"])

model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
