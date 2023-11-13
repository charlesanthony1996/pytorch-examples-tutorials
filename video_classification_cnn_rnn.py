from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os



# define hyperparameters
img_size = 224
batch_size = 64
epochs = 10

max_seq_length = 20
num_features = 2048

# data preparation
train_df = pd.read_csv("/users/charles/train.csv")
test_df = pd.read_csv("/users/charles/test.csv")

print(f"Total videos for training: ", len(train_df))
print(f"total videos for testing: ", len(test_df))

# print(train_df.head())
# print(test_df.head())

train_df.sample(10)




