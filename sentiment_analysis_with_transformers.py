# torch is not needed here
# import torch

# https://github.com/Neuralearn/deep-learning-with-tensorflow-2/blob/main/deep%20learning%20for%20natural%20language%20processing/2-Sentiment%20Analysis%20with%20RNNs%20by%20Neuralearn.ai-.ipynb
import tensorflow as tf
import numpy as np
import sklearn
import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import datetime
import pathlib
import io
import os
import re
import string
import time
from tensorflow.keras import Input
from numpy import random
import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import (Dense, Flatten, SimpleRNN, InputLayer, Conv1D, Bidirectional, GRU, LSTM, BatchNormalization)
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TextVectorization
# from google.colab import drive
# from google.colab import files
from tensorboard.plugins import projector
import matplotlib.pyplot as plt

print("imports passed")


batch_size = 64

# data preparation
train_ds, val_ds, test_ds = tfds.load("imdb_reviews", split=["train", "test[:50%]", "test[50%:]"], as_supervised=True)

# just print out training data and checking
# print(train_ds)

for review, label in val_ds.take(2):
    print(review)
    print(label)


def standardization(input_data):
    lowercase = tf.strings.lower(input_data)

    no_tag= tf.strings.regex_replace(lowercase, "<[^>]+>", "")

    output = tf.strings.regex_replace(no_tag, "[%s]"%re.escape(string.punctuation), "")

    return output



standardization(tf.constant("In the movie?"))


# unit in tokens here
# so sentences more than 250 tokens have to be truncated or cut short

vocab_size = 10000
sequence_length = 250
embedding_dim = 300


vectorize_layer = TextVectorization(
    standardize= standardization,
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length = sequence_length
)

# text vectorization is not defined yet here
print(vectorize_layer)



training_data = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(training_data)

len(vectorize_layer.get_vocabulary())


def vectorizer(review, label):
    return vectorize_layer(review), label


train_dataset = train_ds.map(vectorizer)
val_dataset = val_ds.map(vectorizer)

# getting the 411'th word?
vectorize_layer.get_vocabulary()[411]

for review, label in train_dataset.take(1):
    print(review)
    print(label)



train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# modelling

# simpleRNN
inputs = np.random.random([32, 10, 8]).astype(np.float32)
simple_rnn = tf.keras.layers.SimpleRNN(25)
output = simple_rnn(inputs)
print(output.shape)

embedding_dim = 64
model = tf.keras.models.Sequential([
    Input(shape=(sequence_length,)),
    Embedding(vocab_size, embedding_dim),
    SimpleRNN(32),
    Dense(1, activation="relu")
])

model.summary()


checkpoint_filepath = "/users/charles/desktop/models/rnn.h5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    monitor = "val_accuracy",
    mode = "max",
    save_best_only=True
)


print(model_checkpoint_callback)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.0001),
    metrics=["accuracy"]
)



history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs= 10,
    callbacks = [model_checkpoint_callback]
)


# plotting loss against after each epoch
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model_loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()


# plotting accuracy against epoch
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

