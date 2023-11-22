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