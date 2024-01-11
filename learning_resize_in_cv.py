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
    return activation

