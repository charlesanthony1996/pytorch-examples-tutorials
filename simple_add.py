from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim =2, activation="linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

# test the network with some examples
x_test = [[2, 3], [4, 5], [6, 7], [8, 9]]
y_test = [5, 9, 13, 17]

model.fit(x_test, y_test, epochs=10, verbose=0)

# test the network with new examples
print(model.predict([[1, 1]]))
print(model.predict([[3, 2]]))
