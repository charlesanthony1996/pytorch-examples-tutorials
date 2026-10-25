import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re

import keras
import keras_hub
import tensorflow as tf

keras.config.set_dtype_policy("mixed_bfloat16")

# hyperparameters
epochs = 3
batch_size = 32
vocabulary_size = 20000
max_sequence_length = 200
model_kwargs = dict(
    vocabulary_size= vocabulary_size,
    max_sequence_length=max_sequence_length,
    hidden_dim = 32,
    num_heads=2,
    intermediate_dim=32,
    dropout=0.1
)

train_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=42
)

valid_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/test",
    batch_size=batch_size
)