import csv
import numpy as np

# get the real data from from http://www.kaggle.com/mlg-ulb/creditcardfraud/
fname = "/users/charles/downloads/creditcard.csv"

all_features = []
all_targets = []

with open(fname) as f:
    for i, line in enumerate(f):
        # print(i, line)
        if i == 0:
            print("Header: ", line.strip())
            continue
        fields = line.strip().split(",")
        all_features.append([float(v.replace('"',"")) for v in fields[:-1]])
        all_targets.append([int(fields[-1].replace('"', ""))])
        if i == 1:
            print("example features: ", all_features[-1])

features = np.array(all_features, dtype="float32")
targets = np.array(all_targets, dtype="uint8")
print("features.shape:", features.shape)

print("targets.shape:", targets.shape)


# prepare a validation set
num_val_samples = int(len(features) * 0.2)

train_features = features[:-num_val_samples]
train_targets = targets[:-num_val_samples]

val_features = features[-num_val_samples:]
val_targets = targets[-num_val_samples:]

print("number of training samples: ", len(train_features))
print("number of validating samples: ", len(val_features))

# analyze class imbalance in the targets
counts = np.bincount(train_targets[:, 0])
print("number of positive samples in training data: {} ({:.2f}% of total)".format(counts[1], 100 * float(counts[1]) / len(train_targets)))

weight_for_0 = 1.0/ counts[0]
weight_for_1 = 1.0/counts[1]

# normalize the data using training set statistics
mean = null