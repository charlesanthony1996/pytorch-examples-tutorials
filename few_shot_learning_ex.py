import numpy as np

# define the number of classes
num_classes = 2

# define the number of examples for each class
num_examples = 5

# generate random examples for class A and class B
class_A = np.random.rand(num_examples, 10)
class_B = np.random.rand(num_examples, 10)

# concatenate the examples of both classes
examples = np.concatenate((class_A, class_B), axis = 0)

# generate labels for class A and class B
labels_A = np.zeros(num_examples)
labels_B = np.zeros(num_examples)

# concatenate the labels of both classes
labels = np.concatenate((labels_A, labels_B))

# perform basic addition on the examples
addition_result = examples + 1

print(addition_result)