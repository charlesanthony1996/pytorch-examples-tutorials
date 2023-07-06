import cv2
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Reshape, Conv2DTranspose, Add, Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Input, )
from tensorflow.keras.optimizers import Adam
import numpy as np


(x_train, _) , (x_test, _ ) = tf.keras.datasets.mnist.load_data()

mnist_digits = np.concatenate([x_train, x_test], axis=0)

mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# print(mnist_digits)

dataset = tf.data.Dataset.from_tensor_slices(mnist_digits)

batch_size = 128
latent_dim = 2

train_dataset = (
    dataset
    .shuffle(buffer_size = 1024, reshuffle_each_iteration = True)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

print(train_dataset)

# modelling

# sampling

class Sampling(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return mean + tf.math.exp(0.5 * log_var) * tf.random.normal(shape = (tf.shape(mean)[0] , tf.shape(mean)[1]))
        # print("test line 3")

    def test():
        print("test line 4")


# encoder
encoder_inputs = Input(shape=(28, 28, 1))

x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)

# flattening is needed to reshape the output tensor x
# 2d feature maps from convolutional layers have to be converted into a 1d output vector
x = Flatten()(x)

# dense layers expect 1d outputs
# so the output tensor is flattened
x = Dense(16, activation="relu")(x)


mean = Dense(latent_dim,)(x)
log_var = Dense(latent_dim,)(x)

# why is mean and log_var insider []?
# since sampling expects a list of inputs -> here its the mean and the log variance
# this generates the random sample using the reparameterization trick
# the reparameterization trick allows for efficient backpropagation during training 
z = Sampling()([mean, log_var])

encoder_model = Model(encoder_inputs, [z, mean, log_var], name="encoder")
encoder_model.summary()


# decoder here
latent_inputs = Input(shape=(latent_dim,))

x = Dense(7*7*64, activation="relu")(latent_inputs)
x = Reshape((7,7,64))(x)

x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = Conv2DTranspose(32, 2, activation="relu", strides=2, padding="same")(x)

decoder_output = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder_model = Model(latent_inputs, decoder_output, name="decoder")
decoder_model.summary()



# overall vae model
vae_input = Input(shape=(28, 28, 1), name="vae_input")
z, _, _ = encoder_model(vae_input)
output = decoder_model(z)
vae = Model(vae_input, output, name="vae")
vae.summary()

# whats happening here?
# why am i printing out 3 layers from the vae model?
for i in range(3):
    print(vae.layers[i])


# training
optimizer = Adam(learning_rate=0.01)








