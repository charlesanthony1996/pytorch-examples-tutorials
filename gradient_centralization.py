from time import time
import keras
from keras import layers
from keras.optimizers import RMSprop
# from keras import ops

import tensorflow as tf
from tensorflow import data as tf_data
import tensorflow_datasets as tfds

# prepare the data
num_classes = 2
input_shape = (300, 300, 3)
dataset_name = "horses_or_humans"
batch_size = 128
AUTOTUNE = tf_data.AUTOTUNE

(train_ds, test_ds ), metadata = tfds.load(
    name=dataset_name,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
    with_info=True,
    as_supervised=True
)

# print(train_ds)
# print(test_ds)


# print(f"image shape: , {metadata.features['image'].shape}")
# print(f"training images: ,{metadata.splits['train'].num_examples}")
# print(f"test images: {metadata.splits['test'].num_examples}")

# use data augmentation

rescale = layers.Rescaling(1.0 / 255)
# print(rescale)

data_augmentation = [
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2)
]

# print(data_augmentation)


# helper to data augmentation
def apply_aug(x):
    for aug in data_augmentation:
        x = aug(x)
    return x

def prepare(ds, shuffle = False, augment = False):
    # rescale dataset
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1024)

    # batch dataset
    ds = ds.batch(batch_size)

    # use data augmentation only on the training set
    if augment:
        ds = ds.map(
            lambda x, y: (apply_aug(x), y),
            num_parallel_calls = AUTOTUNE
        )

    return ds.prefetch(buffer_size=AUTOTUNE)


# rescale and augment the data
train_ds = prepare(train_ds, shuffle=True, augment=True)
test_ds = prepare(test_ds)

# define a model
model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.Dropout(0.5),
        
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Dropout(0.5),

        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),

        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ]
)

# print(model.summary())

# implement gradient centralization
# class GCRMSProp(RMSprop):
#     def get_gradients(self, loss, params):
#         grads = []
#         gradients = super().get_gradients()
#         for grad in gadients:
#             grad_len = len(grad.shape)
#             if grad_len > 1:
#                 axis = list(range(grad_len - 1))
#                 grad -= ops.mean(grad, axis=axis, keep_dims=True)
#             grads.append(grad)

#         return grads

class GCRMSProp(RMSprop):
    def __init__(self, learning_rate=1e-4, **kwargs):
        super(GCRMSProp, self).__init__(learning_rate=learning_rate, **kwargs)

    def get_gradients(self, loss, params):
        grads = super(GCRMSProp, self).get_gradients(loss, params)
        if len(grads) > 1:
            axis = list(range(len(grads[0].shape) - 1))
            grads = [grad - tf.reduce_mean(grad, axis=axis, keepdims=True) if len(grad.shape) > 1 else grad for grad in grads]
        return grads


optimizer = GCRMSProp(learning_rate = 1e-4)


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)

# train the model without GC
time_callback_no_gc = TimeHistory()
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=["accuracy"]
)

# model.summary()


history_no_gc = model.fit(train_ds, epochs=10, verbose=1, callbacks=[time_callback_no_gc])

# train the model with GC
time_callback_gc = TimeHistory()
model.compile(loss="binary_crossentropy", optimizer=RMSprop(learning_rate=1e-4), metrics=["accuracy"])

# model.summary()

history_gc = model.fit(train_ds, epochs=10, verbose=1)


# comparing performance
print("not using gradient centralization")
print(f"loss: {history_no_gc.history['loss'][-1]}")
print(f"accuracy: {history_no_gc.history['accuracy'][-1]}")
print(f"training time: {sum(time_callback_no_gc.times)}")


print("Using Gradient Centralization")
print(f"Loss: {history_gc.history['loss'][-1]}")
print(f"Accuracy: {history_gc.history['accuracy'][-1]}")
# print(f"Training Time: {sum(time_callback_gc.times)}")

