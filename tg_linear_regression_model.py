import numpy as np
from tinygrad.tensor import Tensor
import pandas as pd

df = pd.read_csv("CO2_emission_wrangled.csv")

# extract the data from austria
df_austria = df[df["Country_Name"] == "Austria"]
data = df_austria["CO2_emission"].values

class LinearModel:
    def __init__(self, input_dim, output_dim):
        self.weights = Tensor(np.random.normal(0, 0.01, (input_dim, output_dim)).astype(np.float32))
        self.bias = Tensor(np.zeros((output_dim)).astype(np.float32))

    def __call__(self, x):
        return x.dot(self.weights) + self.bias

# data preparation (as in your previous code)
x = data[:-1]
y = data[1:]

# normalization
x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# reshaping the data
x = x.reshape(-1, 1).astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

# define model and loss
model = LinearModel(1, 1)

def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    return (diff * diff).mean()

# train loop
for epoch in range(100):
    pass
    # y_pred = model(Tensor(x))
    # loss = mse_loss(y_pred, Tensor(y))

    # loss.backward()

    # model.weights -= model.weights.grad * 0.01
    # model.bias -= model.bias.grad * 0.01

    # reset gradients to None for the next step
    # model.weights.grad = None
    # model.bias.grad = None

    # print(f"Epoch: {epoch}, Loss: {loss.data}")

# prediction
last_value = data[-1]
last_value_normalized = (last_value - np.min(data)) / (np.max(data) - np.min(data))
prediction = model(Tensor(np.array([[last_value_normalized]]).astype(np.float32)))

print(f"Prediction: {prediction.data[0][0]}")
