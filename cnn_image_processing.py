import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import cifar10


# Load an example image
image = cv.imread("/users/charles/desktop/images/orange.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.summary()

# Preprocess the image
image_resized = cv.resize(image, (32, 32))
image_resized = image_resized.astype("float32") / 255.0
image_expanded = np.expand_dims(image_resized, axis=0)

prediction = model.predict(image_expanded)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0  # Corrected this line

# Convert the labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
epochs = 10
batch_size = 64
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
# evaluate it
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)
