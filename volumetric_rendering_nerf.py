import os

os.environ["keras_backend"] = "tensorflow"

import tensorflow as tf

tf.random.set_seed(42)

import keras
from keras import layers

import os
import glob
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# initialize global variables
auto = tf.data.AUTOTUNE
batch_size = 5
num_samples = 32
pos_encode_dims = 16
epochs = 20


print(auto)

# download and load the data
url = ("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz")

data = keras.utils.get_file(origin=url)
data = np.load(data)
images = data["images"]
im_shape = images.shape
(num_images, h, w, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

plt.imshow(images[np.random.randint(low=0, high=num_images)])
plt.show()