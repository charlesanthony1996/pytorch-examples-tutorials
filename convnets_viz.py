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

