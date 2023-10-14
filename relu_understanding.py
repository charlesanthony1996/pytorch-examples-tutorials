import time

def relu(x):
    return max(0, x)

def convolution(input_data, filter):
    # this is a very naive convolution operation
    return [i * f for i, f in zip(input_data, filter)]


def batch_norm(data):
    # simplified batch normalization
    mean = sum(data) / len(data)
    variance = sum([(i - mean) ** 2 for i in data]) / len(data)
    return [(i - mean) / (variance + 1e-5) ** 0.5 for i in data]


def residual_block(input_data, filter1, filter2):
    # basic structure of a residual block
    x = convolution(input_data, filter1)
    x = batch_norm(x)
    x =[relu(xi) for xi in x]

    x = convolution(x, filter2)
    x = batch_norm(x)

    # skip connection
    return [relu(i + j) for i, j in zip(input_data, x)]


def global_avg_pooling(data):
    return sum(data)/ len(data)


# simplified resnet - 50 structure
input_data = [1.0, 2.0, 3.0, 4.0]

# intial convolution
conv1_filter = [0.5, 0.5, 0.5, 0.5]
x = convolution(input_data, conv1_filter)
x = [relu(xi) for xi in x]


print(x)

# stages (we'll just show one residual block per stage for simplicity)
for _ in range(4):
    filter1 = [0.5, 0.5, 0.5, 0.5]
    filter2 = [0.5, 0.5, 0.5, 0.5]
    x = residual_block(x, filter1, filter2)

# global average pooling
x = global_avg_pooling(x)

output = x * 10

print(output)

