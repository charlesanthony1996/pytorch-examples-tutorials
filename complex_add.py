import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split


# generate training and test data
def generate_data(size):
    x = np.random.randint(0, 100, (size, 2))
    y = np.sum(x**2, axis = 1)
    return x, y


size = 50000
x, y = generate_data(size)
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)


# normalize the data
x_train = x_train / 100
x_test = x_test / 100
y_train = y_train / (2 * 100 ** 2)
y_test = y_test / (2 * 100 ** 2)

# define the model
model = Sequential([
    Dense(128, activation="relu", input_shape=(2,)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="linear"),
])

# compile and train the model

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(x_train, y_train, epochs = 100,  batch_size=64, validation_data=(x_test, y_test))

# test the model

def predict_sum_of_squares(a, b):
    input_data = np.array([[a, b]]) / 100
    prediction = model.predict(input_data)
    return prediction[0][0] * (2 * 100 ** 2)


a, b = 37, 58
result = predict_sum_of_squares(a, b)
print(result)
print()
