import os
os.environ["keras_backend"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf


# the dimensions of our input image
img_width = 180
img_height = 180

layer_name = "conv3_block4_out"

# build a feature extraction model
model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)

print(model)

# setup a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

# set up the gradient ascent process
def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # we avoid border artifacts by only involving non-border pixels in the loss
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)

    # compute gradients
    grads = tape.gradient(loss, img)
    # normalize gradients
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


# setting up the end to end filter visualization loop
def initialize_image():
    img = tf.random.uniform((1, img_width, img_height, 3))
    return (img - 0.5) * 0.25


def visualize_filter(filter_index):
    iterations = 20
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    img = img[25:-25, 25:-25, :]

    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to rgb array
    img *= 255
    img = np.clip(img, 0 , 255).astype("uint8")
    return img

from IPython.display import Image, display

loss, img = visualize_filter(0)
keras.utils.save_img("0.png", img)

display(Image("0.png"))


# visualize the first 64 filters in the target layer
all_imgs = []



