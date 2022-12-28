import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

input_dim = 100
output_dim = 784

inputs = Input(shape=(input_dim,))

hidden = Dense(128, activation="relu")(inputs)
outputs = Dense(output_dim , activation="sigmoid")(hidden)

model = Model(inputs = inputs, outputs= outputs)

print(model)

model.compile(loss="binary_crossentropy", optimizer="adam")
print("After compiling")


print(model)
print()

noise = np.random.rand(1, input_dim)

noise2 = np.random.rand(1, 100)

print(noise)
print()

generated_data = model.predict(noise)

print()

generated_data = model.predict(noise2)

print()

print(generated_data)
