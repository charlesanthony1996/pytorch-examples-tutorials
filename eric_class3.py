import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import umap

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

class_names = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("training set images shape: ", x_train.shape)
print("training set labels shape: ", y_train.shape)
print("test set images shape: ", x_test.shape)
print("test set labels shape: ", y_test.shape)

# display the first image from the training set
plt.figure(figsize=(8, 8))
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Class: {class_names[y_train[0]]}")
plt.colorbar()
plt.grid(False)
# plt.show()

class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


shape = x_test.shape[1:]

latent_dim64 = 64
autoencoder64 = Autoencoder(latent_dim64, shape)

latent_dim2 = 2
autoencoder2 = Autoencoder(latent_dim2, shape)

latent_dim_miss = 64
autoencoder_miss = Autoencoder(latent_dim_miss, shape)

# what is an the purpose of the encoder?

# the encoder part of an autoencoder learns to compress the input into a smaller encoding
# this encoding reduced representation of the input but captures its salient features.essentially
# the encoder transforms the input into a latent space representation. it learns to preserve only the aspects
# of the data that are most important for reconstructing it

# what is the purpose of the decoder?

# the decoder takes the encoding provided by the encoder and attempts to recreate the original input using
# this reduced representation. the goal of the decoder is to reverse the process of the encoder, reconstructing
# the input data as accurately as possible from the compressed code


# what type of loss are we using and what does it do?

# autoencoders typically use a reconstruction loss to train the network, which measures how well the decoders
# output matches the original input. the common types 



autoencoder64.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder2.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder64.fit(x_train, x_train,
                epochs=5,
                shuffle=True,
                validation_data=(x_test, x_test))


autoencoder2.fit(x_train, x_train,
                epochs=5,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs64 = autoencoder64.encoder(x_test).numpy()
decoded_imgs64 = autoencoder64.decoder(encoded_imgs64).numpy()

encoded_imgs2 = autoencoder2.encoder(x_test).numpy()
decoded_imgs2 = autoencoder2.decoder(encoded_imgs2).numpy()

# visualization of the results

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs2[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.suptitle('Latent Space = 2')
plt.show()


plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs64[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.suptitle('Latent Space = 64')
plt.show()




# visualization latent space dim=2

# select a random amount of points to avoid plotting all points
idx = np.random.choice(len(x_test), 1000)

images = x_test[idx]
encodings = encoded_imgs2[idx]
labels = y_test[idx]


print(encodings.shape)
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("latent space autoencoder dim 2")
plt.scatter(encodings[:, 0], encodings[:, 1], c=labels,cmap = "viridis")
plt.colorbar()
for i in range(10):
    class_center = np.mean(encodings[labels == i], axis=0)
    text = TextArea('{} ({})'.format(class_names[i], i))
    ab = AnnotationBbox(text, class_center, xycoords='data', frameon=True)
    ax.add_artist(ab)
plt.show()
