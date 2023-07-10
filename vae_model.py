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
epochs = 20

def custom_loss(y_true, y_pred, mean, log_var):
    loss_rec = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(y_true, y_pred), axis=(1, 2)))

    loss_reg = -0.5 * (1+ log_var - tf.square(mean) - tf.exp(log_var))

    return loss_rec + tf.reduce_mean(tf.reduce_sum(loss_reg, axis= 1))




@tf.function
def training_block(x_batch):
    with tf.GradientTape() as recorder:
        z,mean, log_var = encoder_model(x_batch)
        y_pred = decoder_model(z)
        y_true = x_batch
        loss = custom_loss(y_true, y_pred, mean, log_var)


    partial_derivatives = recorder.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(partial_derivatives, vae.trainable_weights))
    return loss


def neuralearn(epochs):
    for epoch in range(1, epochs + 1):
        print("training starts for epoch number {}".format(epoch))

        for step, x_batch in enumerate(train_dataset):
            loss = training_block(x_batch)
        print("training loss is: ", loss)
    print("Training complete!!!")



# let this train
# neuralearn(epochs)


# overriding train_step method

class VAE(tf.keras.Model):
    def __init__(self, encoder_model, decoder_model):
        super(VAE, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]


    def train_step(self, x_batch):
        with tf.GradientTape() as recorder:
            z, mean, log_var = encoder_model(x_batch)
            y_pred = decoder_model(z)
            y_true = x_batch
            loss = custom_loss(y_true, y_pred, mean, log_var)


        partial_derivatives = recorder.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(partial_derivatives, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result() }



model = VAE(encoder_model, decoder_model)
model.compile(optimizer= optimizer)
model.fit(train_dataset, epochs = 20, batch_size=128,)


# testing
scale = 1
n = 16

grid_x = np.linspace(-scale, scale, 16)
grid_y = np.linspace(-scale, scale, 16)

print(grid_x, grid_y)

plt.figure(figsize=(12, 12))
k = 0
for i in grid_x:
    for j in grid_y:
        x = plt.subplot(n, n, k + 1)

        input = tf.constant([[i, j]])
        out = model.decoder.predict(input)[0][..., 0]
        plt.imshow(out, cmap="Greys_r")
        plt.axis("off")
        k += 1



print(vae.layers[2].predict(tf.constant([[-1,1]]))[0][..., 0].shape)

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
mnist_digits = np.expand_dims(x_train, -1).astype("float32") / 255

z, _, _ =  vae.layers[1].predict(x_train)
plt.figure(figsize=(12, 12))
plt.scatter(z[:,0] , z[:, 1], c= y_train)
plt.colorbar()
plt.show()


