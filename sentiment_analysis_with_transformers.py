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

# transformers
# embeddings


# positional embeddings
def positional_encoding(model_size,sequence_length):
    output = []
    for pos in range(sequence_length):
        PE = np.zeros((model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[i] = np.sin(pos/10000 **(i/model_size))
            else:
                PE[i] = np.cos(pos/ 10000 ** ((i - 1) /model_size))
    output.append(tf.expand_dims(PE, axis= 0))
    out = tf.concat(output, axis = 0)
    out = tf.expand_dims(out , axis = 1)
    return tf.cast(out, dtype=tf.float32)


class Embeddings(Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim,):
        super(Embeddings, self).__init__()
        self.token_embeddings= Embedding(
            input_dim = vocab_size, output_dim = embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim


    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = positional_encoding(
            self.embed_dim, self.sequence_length
        )
        return embedded_tokens + embedded_positions


    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "emded_dim": self.embed_dim,
        })

        return config


test_input = tf.constant([[2, 112, 10, 12, 5, 0, 0, 0,]])
emb = Embeddings(8, 20000, 256)
emb_out =emb(test_input)
print(emb_out.shape)



class TransformerEncoder(layer):
    def __init__(self, embed_dim, dense_dim, num_heads,):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim,
        )
        self.dense_proj = tf.keras.Sequential(
            [Dense(dense_dim, activation="relu"), Dense(embed_dim,)]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    
    def call(self, inputs, mask = None):
        if mask is not None:
            mask1 = mask[:, : tf.newaxis]
            mask2 = mask[:, tf.newaxis, :]
            padding_mask = tf.cast(mask1&mask2 , dtype="int32")


            attention_output = self.attention(
                query= inputs, key=inputs, value=inputs, attention_mask=padding_mask
            )

            proj_input = self.layernorm_1(inputs + attention_mask)
            proj_output = self.dense_proj(proj_input)
            return self.layernorm_2(proj_input + proj)
