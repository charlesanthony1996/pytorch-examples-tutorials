import torch
import numpy as np
from sklearn.base import BiclusterMixin
from sklearn.cluster import SpectralBiclustering
from sklearn.cluster import SpectralCoclustering

# creating some dummy data
data = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])

data1 = np.array([
    [1, 2, 3, 4],
    [3, 1, 2, 4],
    [4, 3, 2, 1],
    [2, 1, 3, 4]
])


# print(data)

# creating the biclustering mixin model with 2 clusters
model = SpectralBiclustering(n_clusters= 4, random_state= 0)
model_spectralCo = SpectralCoclustering(n_clusters= 4, random_state = 0)


# print(model)

model.fit(data1)

model_spectralCo.fit(data1)
# output the row and column labels
row_labels = model.row_labels_
column_labels = model.column_labels_

row_labels1 = model_spectralCo.row_labels_
column_labels1 = model_spectralCo.column_labels_

# print("row labels: ", row_labels)
# print("column labels: ", column_labels)

print("row labels: ", row_labels1)
print("column labels: ", column_labels1)



