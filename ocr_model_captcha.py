import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# download the captcha images
data_dir = Path("/users/charles/downloads/captcha_images_v2/")

# print(data_dir)

# get the list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
# print(len(images))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
# print(len(labels))
charachters = set(char for label in labels for char in label)
# print(len(charachters))
characters = sorted(list(charachters))
# print(len(charachters))

# print("number of images found: ", len(images))
# print("number of labels found: ", len(labels))
# print("number of unique charachters: ", len(charachters))
# print("charachters present: ", charachters)

# batch size for training and validation
batch_size = 16

# desired image dimensions
img_width = 200
img_height = 50

# factor by which the image is going to be downsampled
# by the convolutional blocks. we will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2
# hence total downsampling factor would be 4
downsample_factor = 4

# maximum length of any captcha in the dataset
max_length = max(len(label) for label in labels)
# print(max_length)

# preprocessing
# mapping charachters to integers
char_to_num = layers.StringLookup(vocabulary = list(charachters), mask_token=None)

# mapping integers back to original characters
num_to_char = layers.StringLookup(vocabulary = char_to_num.get_vocabulary(), mask_token=None, invert =True)


def split_data(images, labels, train_size = 0.9, shuffle=True):
    # get the total size of the dataset
    size = len(images)
    # make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)

    # get the size of training samples
    train_samples = int(size * train_size)
    # split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]

    return x_train, x_valid, y_train, y_valid


# splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

def encode_single_sample(img_path, label):
    # read image
    img = tf.io.read_file(img_path)
    # decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5 transpose the image because we want the time
    # dimension to correspond to the width of the image
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6 map the charachters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))


    return {"image": img, "label": label}


# create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size = tf.data.AUTOTUNE)
)

# visualize the data
_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")

# plt.show()



class CTCLayer(layers.Layer):
    def __init__():
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost


    def call(self, y_true, y_pred):
        # compute the training time loss value and add it
        # to the layer using self.add_loss()
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # at test time , just return the computed predictions
        return y_pred

        
