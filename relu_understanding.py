import time

def relu(x):
    return max(0, x)

def convolution(input_data, filter):
    # this is a very naive convolution operation
    return sum([i * f for i, f in zip(input_data, filter)])


def batch_norm(data):
    # simplified batch normalization
    mean = sum(data) / len(data)
    variance = sum([(i - mean) ** 2 for i in data]) / len(data)
    return [(i - mean) / (variance + 1e-5) ** 0.5 for i in mean]


def residual_block(input_data, filter1, filter2):
    # basic structure of a residual block
    x = convolution(input_data, filter1)
    x = batch_norm(x)
    x = relu(x)

    x = convolution(x, filter2)
    x = batch_norm(x)

    # skip connection
    return [relu(i + j) for i, j in zip(input_data, x)]


def global_avg_pooling(data):
    return sum(data)/ len(data)