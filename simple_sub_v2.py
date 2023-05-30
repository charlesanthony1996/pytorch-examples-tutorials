import keras
import tensorflow
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# generate train data
x_train = np.array([[i, j] for i in range(1, 100) for j in range(1, 100)])
y_train = np.array([i - j for i, j in x_train])

# scale input data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

print(x_train)


# reshape input data for the lstm
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

model = Sequential()
model.add(LSTM(64, input_shape = (2, 1) , activation="relu", return_sequences=True))
model.add(LSTM(32, activation="relu"))
model.add(Dense(1, activation="linear"))

# compile and train the model
optimizer = Adam(learning_rate= 0.0001)
model.compile(loss="mean_squared_error", optimizer= optimizer)
model.fit(x_train, y_train, epochs= 50, verbose= 1)


# test the model with a new example

x_test = scaler.transform([[5, 1]])
x_test = x_test.reshape(1,x_test.shape[1], 1)
print(model.predict(x_test))