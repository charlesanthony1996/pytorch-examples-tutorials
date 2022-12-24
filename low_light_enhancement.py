import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


image_size = 256
batch_size = 16
max_train_images = 400

def load_data(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels = 3)
    image = tf.image.resize(images = image, size= [image_size, image_size])
    image = image/ 255.0
    return image


def data_generator(low_light_images):
    pass
    

