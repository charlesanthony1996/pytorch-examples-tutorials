import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
# from keras import layers
# from keras import ops
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

num_classes = 100
input_shape = (32, 32, 3)


patch_size = (2, 2)
dropout_rate = 0.03
num_heads = 8
embed_dim = 64
num_mlp = 256

qkc_bias = True
window_size = 2
shift_size = 1
image_dimension = 32

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[0] // patch_size[1]

learning_rate = 1e-3
batch_size = 128
num_epochs = 40
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1

# prepare the data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(x_train, num_classes)
num_train_samples = int(len(x_train) * (1 - validation_split))
num_val_samples = len(x_train) - num_train_samples
x_train, x_val = np.split(x_train, [num_train_samples])
y_train, y_val = np.split(y_train, [num_train_samples])
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test.shape: {y_test.shape}")


plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i * 1)
    plt.xticks()
    plt.yticks()
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.show()

# helper functions
def window_parition(i, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = height // window_size
    x = ops.reshape(
        x,
        (
            -1,
            patch_num_y,
            window_size,
            patch_num_x,
            window_size, 
            channels,
        ),
    )
    x = ops.transpose(x, (0, 1, 2, 3, 4, 5))
    windows = ops.reshape(x, (-1, window_size, window_size, channels))
    return windows



