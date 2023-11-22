import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

# downloading dataset
dataset_url = "https://git.io/JiY4i"

desired_directory = "/users/charles/downloads"
dataset_path = keras.utils.get_file(
    fname="shapenet2.zip",
    origin = dataset_url,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract =True,
    archive_format="auto",
    cache_dir=desired_directory,
)


# loading the dataset
with open("/tmp/.keras/datasets/PartAnnotation/metadata.json") as json_file:
    metadata = json.load(json_file)

print(metadata)

# hyperparameters
points_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points".format(metadata["Airplane"]["directory"])

labels_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points_label".format(metadata["Airplane"]["directory"])

labels = metadata["Airplane"]["lables"]
colors = metadata["Airplane"]["colors"]

val_split = 0.2
num_sample_points = 1024
batch_size = 32
epochs = 60
intial_lr = 1e-3

