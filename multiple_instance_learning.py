import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.style.use("ggplot")

positive_class = 1
bag_count = 1000
val_bag_count = 300
bag_size = 3
plot_size = 3
ensemble_avg_count = 1

# prepare bags

def create_bags(input_data, input_labels, positive_class, bag_count, instance_count):

    # set up bags
    bags = []
    bag_labels = []

    # normalize input data
    input_data = np.divide(input_data, 255.0)

    # count positive samples
    count = 0

    for _ in range(bag_count):

        # pick a fixed size random subset of samples
        index = np.random.choice(input_data.shape[0], instance_count, replace=False)
        instances_data = input_data[index]
        instances_labels = input_labels[index]

        # by default , all bags are labeled as 0
        bag_label = 0

        # check if there is at least a positive class in the bag
        if positive_class in instances_labels:

            # positive bag will be labeled as 1
            bag_label = 1
            count += 1

        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))


    print(f"Positive bags: {count}")
    print(f"Negative steps: {bag_count - count}")

    return list(np.swapaxes(bags, 0, 1)), np.array(bag_labels)


# load the mnist dataset
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

# create training data
train_data, train_labels = create_bags(
    x_train, y_train, positive_class, bag_count, bag_size
)

# print(train_data)

# print(train_labels)

# create validation data
val_data, val_labels = create_bags(
    x_val, y_val, positive_class, val_bag_count, bag_size
)

# print(val_data)

# print(val_labels)






