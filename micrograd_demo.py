import micrograd
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
import random
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)
random.seed(1337)

# make up a dataset
from sklearn.datasets import make_moons, make_blobs
x, y = make_moons(n_samples=100, noise= 0.1)

y = y* 2- 1

# visualize this in 2D
plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1], c= y, s= 20, cmap="jet")
plt.show()


# initialize a model
