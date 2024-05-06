from time import time
import keras
from keras import layers
from keras.optimizers import RMSprop
# from keras import ops
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


num_classes = 2
input_shape = (300, 300, 3)
dataset_name = "horses_or_humans"
dataset_name2 = "MNIST"
batch_size = 128
autotune = torch.utils.data.DataLoader

# (train_ds, test_ds), metadata = tfds.load(
#     name=dataset_name,
#     split=[tfds.Split.TRAIN, tfds.Split.TEST],
#     with_info=True,
#     as_supervised=True
# )

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_data = torchvision.datasets.MNIST("/users/charles", download=True, train=True, transform=transform)

metadata = DataLoader(
    mnist_data,
    batch_size= 128,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(mnist_data))

print(labels)
# denormalize the image
print("Image shape: ", images[0].shape)
train_dataset = datasets.MNIST(root="/users/charles", download=True, train=True, transform=transform)
test_dataset = datasets.MNIST(root="/users/charles", download=True, train=False, transform=transform)
print("Training examples: ",len(train_dataset))
print("Testing examples: ",len(test_dataset))




# select the first image
# image = images[0]

# # denormalize the image
# image = image * 0.5 + 0.5

# # plotting the figure
# plt.figure(figsize=(6, 6))
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(f"label: {labels}")
# plt.axis("off")
# # plt.show()

# data augmentation



