from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2

model = Sequential()
model.add(Dense(8, input_dim =2 , activation="relu" , kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="linear"))

optimizer = Adam(learning_rate= 0.001)
model.compile(loss="mean_squared_error", optimizer = optimizer)

# generate more training data
x_train = [[i, j] for i in range(1, 10) for j in range(1, 10)]
y_train = [i - j for i, j in x_train]

# train the model with more data and for more epochs
model.fit(x_train, y_train, epochs = 500, verbose = 1)

# test the model with new examples
print(model.predict([[5, 1]]))


# import numpy as np
# import tensorflow as tf
# from tensorflow import keras


# # define the model architecture
# model = keras.Sequential([
#     keras.layers.Dense(16, input_shape=[2], activation="relu"),
#     keras.layers.Dense(8, activation="relu"),
#     keras.layers.Dense(4, activation="relu"),
#     keras.layers.Dense(1, activation="linear")
# ])

# # compile the model
# optimizer = tf.optimizers.Adam(0.001),
# model.compile(optimizer = optimizer, loss = "mean_squared_error")

# # provide the training data
# x_train = np.random.randint(1, 100, (1000, 2))
# y_train = np.array([x[0] - x[1] for x in x_train])

# # train the model
# model.fit(x_train, y_train, epochs= 100, verbose= 1)

# # use the model to make predictions
# print(model.predict([[50, 25]]))