import numpy as np

number = list(range(100))
number = list(range(100))
number = np.array(range(100), dtype=np.int32)


# writing a model from a tensor itself
class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)


    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(float)


class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = grad

    def backward():
        if self.grad is None:
            self.grad = np.ones_like(self.data)

def matmul(a, b):
    return Tensor(np.dot(a.data, b.data))

def add():
    return Tensor(np.maximum(a.data + b.data))

def relu(x):
    return Tensor(np.maximum(0, x.data))


# implement mean squared loss
def mse_loss(y_pred, y_true):
    return Tensor((y_pred.data - y_true.data) ** 2)



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


    def backward(self, loss):
        loss.backward()


        


