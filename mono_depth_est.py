import os
import sys

import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

tf.random.set_seed(123)


annotation_folder = "/dataset/"
if not os.path.exists(os.path.abspath(".") + annotation_folder):
    annotation_zip = tf.keras.utils.get_file(
        "val.tar.gz",
        cache_subdir=os.path.abspath("."),
        origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
        extract=True,
    )


# preparing the dataset

path = "val/indoors"

filelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        filelist.append(os.path.join(root, file))
        # print(file)

filelist.sort()

data = {
    "image": [x for x in filelist if x.endswith(".png")],
    "depth": [x for x in filelist if x.endswith("_depth.npy")],
    "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
}

# print(data)

df = pd.DataFrame(data)

df = df.sample(frac = 1, random_state= 42)


# preparing hyperparameters

HEIGHT = 256
WIDTH = 256
LR = 0.0002
EPOCHS = 30
BATCH_SIZE = 32


# building a data pipeline

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size = 6, dim = (768, 1024), n_channels=3, shuffle=True):
        # init
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # generate one batch of data
        # generate indices of the batch
        index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        # find list of ids
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y


    def on_epoch_end(self):
        # updates indexes after each epoch
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
        

    def load(self, image_path, depth_map, mask):
        # load input and target image
        image_ = cv.imread(image_path)
        image_ = cv.cvtColor(image_, cv.COLOR_BGR2RGB)
        image_ = cv.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)


        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where = mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis = 2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):

        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))


        for i, batch_id in enumerate(batch):
            x[i, ], y[i, ] = self.load(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                self.data["mask"][batch_id],
            )

        return x, y


def visualize_depth_map(samples, test = False, model = None):
    input, target = samples
    cmap = plt.cm.jet
    cmap.set_bad(color="black")


    if test:
        pred = model.predict(input)
        fig, ax = plt.subplots(6, 3, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((input[i].squeeze()), cmap=cmap)
            ax[i, 2].imshow((input[i].squeeze()), cmap=cmap)

    else:
        fig, ax = plt.subplots(6, 2, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((input[i].squeeze()), cmap=cmap)

visualize_samples = next(iter(DataGenerator(data=df, batch_size= 6, dim=(HEIGHT, WIDTH))))

visualize_depth_map(visualize_samples)



