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
model.fit(x_train, y_train, epochs = 200, verbose = 1)

# test the model with new examples
print(model.predict([[5, 1]]))