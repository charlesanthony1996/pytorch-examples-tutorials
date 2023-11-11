import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox

# download and load the dataset
fpath = keras.utils.get_file("moving_mnist.npy", "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy")

dataset = np.load(fpath)

# print(dataset[0])

# swap the axes representing the number of frames and number of data samples
dataset = np.swapaxes(dataset, 0, 1)

# well pick out 1000 of the 10000 total examples and use those
dataset = dataset[:1000, ...]

# add a channel dimension since the images are grayscale
dataset = np.expand_dims(dataset, axis= -1)

# split into train and validation sets using indexing to optimize memory
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]) :]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]

# normalize the data to the 0-1 range
train_dataset = train_dataset  /  255
val_dataset = val_dataset / 255

# we'll define a helper function to shift the frames, where
# x is frames 0 to n - 1 and y is frames 1 to n
def create_shifted_items(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y

# apply the processing function to the datasets
x_train, y_train = create_shifted_items(train_dataset)
x_val, y_val = create_shifted_items(val_dataset)

# inspect the dataset
print("Training dataset shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("validation dataset shapes: " + str(x_val.shape) + ", " + str(y_val.shape))


# data visualization

# construct a figure on which we will visualize the images
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

# plot each of the sequential images for one random data example
data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]

for idx, ax in enumerate(axes.flat):
    ax.imshow(np.squeeze(train_dataset[data_choice][idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")


# print information
print(f"displaying frames for example {data_choice}")
# plt.show()

# model construction
# construct the input layer with no definite frame size
inp = layers.Input(shape=(None, *x_train.shape[2:]))

print(inp)

# we will construct 3 convlstm2d layers with batch normalization
# followed by a conv3d layer for spatiotemportal outputs

x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu")(inp)

print(x)

x = layers.BatchNormalization()(x)
print(x)
x = layers.ConvLSTM2D(filters=64,kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
print(x)

x = layers.BatchNormalization()(x)
print(x)

x = layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), padding="same", return_sequences=True, activation="relu")(x)

print(x)

x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)

print(x)

# next we will build the complete model and compile it
model = keras.models.Model(inp, x)

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())

# model training
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# define modifiable training hyperparameters
epochs = 1
batch_size = 5

# fit the model to the training data
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=(early_stopping, reduce_lr))

# frame predictive visualizations
example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

print(example)

# pick the first last ten frames from the example
frames = example[:10, ...]
original_frames = example[10:, ...]

# predict a new set of 10 frames
for _ in range(10):
    # extract the models prediction and post process it
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)


    # extend the set of prediction frames
    frames = np.concatenate((frames, predicted_frame), axis=0)


# construct a figure for the original and new frames
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
print(fig)
print(axes)


# plot the original frames
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"frame {idx + 11}")
    ax.axis("off")


# plot the new frames
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"frame {idx + 11}")
    ax.axis("off")


# display the figure
plt.show()