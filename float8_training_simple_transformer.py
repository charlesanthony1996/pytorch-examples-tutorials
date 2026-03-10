import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re

import keras
import keras_hub
import tensorflow as tf

# hyperparameters
epochs = 3
batch_size = 32
vocabulary_size = 20000
max_sequence_length = 200
# model_kwargs = dict(

# ) 