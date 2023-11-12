import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


# setting seed for responsibility
seed = 42
keras.utils.set_random_seed(seed)


num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape {x_test.shape}")


# configure the hyperparameters
buffer_size = 512
batch_size = 256

# augmentation
