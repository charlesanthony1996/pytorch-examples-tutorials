from sklearn.cluster import MeanShift
import numpy as np

x = np.array([
    [1, 1], [2, 3],
    [5, 5], [3, 4]
])

clustering = MeanShift(bandwidth = 2).fit(x)

print(clustering.labels_)

print(clustering.predict([[1, 3]]))

# print(clustering)

    # [5, 5], [3, 4]    // Hill 1
    #      [1, 3]       // Your test point
    # [1, 1], [2, 3]    // Hill 0

# your ball will roll to the point of the lowest hill -> thats mean shift