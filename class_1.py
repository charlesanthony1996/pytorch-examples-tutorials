import numpy as np


# writing a model from a tensor itself
class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size 
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.random.randn(output_size) * 0.01
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)


    def forward(self, x):
        # storing x for use in backward pass
        self.x = x
        return np.dot(x, self.weights) + self.bias


    def backward(self, grad_output):
        self.grad_weights = np.dot(self.x.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis = 0)
        return np.dot(grad_output, self.weights.T)



class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.x <= 0] = 0
        return grad_input


class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = grad

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

def matmul(a, b):
    return Tensor(np.dot(a.data, b.data))

def add(a, b):
    return Tensor(np.maximum(a.data + b.data))

def relu(x):
    return Tensor(np.maximum(0, x.data))


# implement mean squared loss
def mse_loss(y_pred, y_true):
    loss = np.mean((y_pred - y_true) ** 2)
    grad_loss = 2.0 * (y_pred - y_true) / y_true.size
    return loss, grad_loss



class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        x = self.linear2.forward(x)
        return x


    def backward(self, grad_output):
        grad_output = self.linear2.backward(grad_output)
        grad_output = self.relu.backward(grad_output)
        grad_output = self.linear1.backward(grad_output)



def train(model, x, y, epochs = 1000, learning_rate= 0.01):
    for epoch in range(epochs):
        # forward pass
        y_pred = model.forward(x)
        loss, grad_loss = mse_loss(y_pred, y)

        # backward pass and optimization
        model.backward(grad_loss)

        for layer in [model.linear1, model.linear2]:
            layer.weights -= learning_rate * layer.grad_weights
            layer.bias -= learning_rate * layer.grad_bias

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")



np.random.seed(0)
x = np.random.randn(100, 5)
y = np.random.randn(100, 1)

model = Model(5, 10, 1)
train(model, x, y)



