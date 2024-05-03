import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import cluster
import matplotlib.pyplot as plt
import seaborn as sns
import glob


# Task 1

# generate a simple scatterplot containing
# data[:,0] are x values, data[:,1] are the y values
data,_ = datasets.make_blobs(n_samples=1000, centers=2, n_features=2,random_state=3)


# K-means uses centroids for initialization
# Since we want to find 2 clusters we use 2 centroids 
centroid_A = np.array([0,0])
centroid_B = np.array([-3.0,5.0])

# visualize the datapoints and centroids
plt.figure()
plt.scatter(data[:,0],data[:,1],color = "gray", label = "datapoints")
plt.plot(centroid_A[0],centroid_A[1],color="tab:orange",marker="v", markersize= 10, label = "centroid A")
plt.plot(centroid_B[0],centroid_B[1],color="tab:blue", marker = "v", markersize= 10, label = "centroid B")
plt.legend()
# plt.show()

# 2
diff_vec_A = data - centroid_A
diff_vec_B = data - centroid_B

# 3
dist_vec_A = np.linalg.norm(diff_vec_A, axis=1)
dist_vec_B = np.linalg.norm(diff_vec_B, axis=1)

# 4
dist_vec = np.vstack((dist_vec_A, dist_vec_B)).T
cl_indices = np.argmin(dist_vec, axis=1)

# 5
cluster_A = data[cl_indices == 0]
cluster_B = data[cl_indices == 1]

# 6
centroid_A_new = np.mean(cluster_A, axis=0)
centroid_B_new = np.mean(cluster_B, axis=0)

# 7
centroid_dist_A = np.linalg.norm(centroid_A_new - centroid_A)
centroid_dist_B = np.linalg.norm(centroid_B_new - centroid_B)

plt.figure()
plt.scatter(cluster_A[:,0],cluster_A[:,1],color = "gold", label = "Cluster A points")
plt.scatter(cluster_B[:,0],cluster_B[:,1],color = "lightblue", label = "Cluster B points")
plt.plot(centroid_A[0],centroid_A[1],color="tab:orange",marker="s", markersize= 7, label = "old centroid A")
plt.plot(centroid_B[0],centroid_B[1],color="tab:blue", marker = "s", markersize= 7, label = "old centroid B")
plt.plot(centroid_A_new[0],centroid_A_new[1],color="tab:orange",marker="v", markersize= 10, label = "new centroid A")
plt.plot(centroid_B_new[0],centroid_B_new[1],color="tab:blue", marker = "v", markersize= 10, label = "new centroid B")
plt.legend()
# plt.show()

print("below are the intermediate results for reference. Only the first 5 entries for each variable are printed")
print("\n")
print(f"data: \n{data[0:5,:]}")
print("\n")
print(f"centroid_A: \n{centroid_A}")
print(f"centroid_B: \n{centroid_B}")
print("\n")
print(f"diff_vec_A: \n{diff_vec_A[0:5,:]}")
print(f"diff_vec_B: \n{diff_vec_B[0:5,:]}")
print("\n")
print(f"dist_vec_A: \n{dist_vec_A[0:5]}")
print(f"dist_vec_B: \n{dist_vec_B[0:5]}")
print("\n")
print(f"cl_indices: \n{cl_indices[0:5]}")
print("\n")
print(f"cluster_A: \n{cluster_A[0:5]}")
print(f"cluster_B: \n{cluster_B[0:5]}")
print("\n")
print(f"centroid_A_new: \n{centroid_A_new}")
print(f"centroid_B_new: \n{centroid_B_new}")
print("\n")
print(f"centroid_dist_A: \n{centroid_dist_A}")
print(f"centroid_dist_B: \n{centroid_dist_B}")


def custom_kmeans(data, centroid_A, centroid_B):
    # Step 2: Compute difference vectors
    diff_vec_A = data - centroid_A
    diff_vec_B = data - centroid_B

    # Step 3: Compute Euclidean distances
    dist_vec_A = np.linalg.norm(diff_vec_A, axis=1)
    dist_vec_B = np.linalg.norm(diff_vec_B, axis=1)

    # Step 4: Assign clusters
    cl_indices = np.argmin(np.vstack((dist_vec_A, dist_vec_B)).T, axis=1)

    # Step 5: Find points in each cluster
    cluster_A = data[cl_indices == 0]
    cluster_B = data[cl_indices == 1]

    # Step 6: Compute new centroids
    centroid_A_new = np.mean(cluster_A, axis=0)
    centroid_B_new = np.mean(cluster_B, axis=0)

    # Step 7: Compute the movement of each centroid
    centroid_dist_A = np.linalg.norm(centroid_A_new - centroid_A)
    centroid_dist_B = np.linalg.norm(centroid_B_new - centroid_B)

    print(f"centroid A moved by: {np.round(centroid_dist_A, 3)}, centroid B by: {np.round(centroid_dist_B, 3)}")
    
    plt.figure()
    plt.scatter(cluster_A[:, 0], cluster_A[:, 1], color="gold", label="Cluster A points")
    plt.scatter(cluster_B[:, 0], cluster_B[:, 1], color="lightblue", label="Cluster B points")
    plt.plot(centroid_A[0], centroid_A[1], color="tab:orange", marker="s", markersize=7, label="Old Centroid A")
    plt.plot(centroid_B[0], centroid_B[1], color="tab:blue", marker="s", markersize=7, label="Old Centroid B")
    plt.plot(centroid_A_new[0], centroid_A_new[1], color="tab:orange", marker="v", markersize=10, label="New Centroid A")
    plt.plot(centroid_B_new[0], centroid_B_new[1], color="tab:blue", marker="v", markersize=10, label="New Centroid B")
    plt.legend()
    plt.show()

    return centroid_A_new, centroid_B_new, cluster_A, cluster_B

# Task 2
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# %matplotlib inline
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


penguins = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")

penguins.to_csv("penguins.csv", index=False)
penguins = pd.read_csv("penguins.csv")
print(penguins.head())

penguins = penguins.dropna()
penguins.species.value_counts()

sns.pairplot(penguins.drop("year", axis=1), hue='species')
# plt.show()

def gt_label_indices(df,f):
    label_list = df[f].to_list()
    unique_list = list(np.unique(label_list))
    
    index_list = []
    for l in label_list:
        index_list.append(unique_list.index(l))
        
    return index_list
        
def kmeans(df,n_clusters,f1,f2):
    labels = cluster.KMeans(n_clusters=n_clusters, n_init=10).fit_predict(df[[f1,f2]])
    return labels

def spectral(df, n_clusters,f1,f2):

    labels = cluster.SpectralClustering(
        n_clusters=n_clusters,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=0).fit_predict(df[[f1, f2]])
    
    return labels  

def birch(df, n_clusters,f1,f2):
#     df = df.sample(frac=0.5)    answer = len(df['color'].unique())
    labels = cluster.Birch(branching_factor=200, threshold=1, n_clusters=n_clusters) \
                    .fit_predict(df[[f1, f2]])
    return labels


from sklearn import cluster
f1 = "flipper_length_mm"
f2 = "bill_length_mm"

gt_labels       = gt_label_indices(penguins,"species")
kmeans_labels   = kmeans(penguins,3,f1,f2)
spectral_labels = spectral(penguins,3,f1,f2)
birch_labels    = birch(penguins,3,f1,f2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(data=penguins, x=f1, y=f2, hue = gt_labels,ax=axes[0,0])
axes[0,0].set_title("GT Labels")

sns.scatterplot(data=penguins, x=f1, y=f2, hue = kmeans_labels,ax=axes[0,1])
axes[0,1].set_title("K-Means")

sns.scatterplot(data=penguins, x=f1, y=f2, hue = spectral_labels,ax=axes[1,0])
axes[1,0].set_title("spectral")

sns.scatterplot(data=penguins, x=f1, y=f2, hue = birch_labels,ax=axes[1,1])
axes[1,1].set_title("BIRCH")

# first we scale the data to normalize the range of values

penguin_data = penguins[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values

scaled_penguin_data = StandardScaler().fit_transform(penguin_data)


from sklearn.decomposition import PCA
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py

X_reduced = PCA(n_components=2).fit_transform(scaled_penguin_data)


print(X_reduced[:, 0].shape)


pca_df = pd.DataFrame(data={"species":penguins["species"],
                       "feature1":X_reduced[:, 0],
                       "feature2":X_reduced[:, 1]})

sns.scatterplot(data=pca_df, x="feature1", y="feature2", hue = "species", s=100).set(title='PCA')

# plt.scatter(
#     X_reduced[:, 0],
#     X_reduced[:, 1],
#     c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})],
#     s= 100,
#  )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('PCA projection of the Penguin dataset', fontsize=24)


from sklearn import manifold


tsne = manifold.TSNE(n_components=3, init='pca', random_state=2)
X_tsne = tsne.fit_transform(scaled_penguin_data)

tsne_df = pd.DataFrame(data={"species":penguins["species"],
                       "feature1":X_tsne[:, 0],
                       "feature2":X_tsne[:, 1]})

sns.scatterplot(data=tsne_df, x="feature1", y="feature2", hue = "species", s=100).set(title='TSNE')
# plt.show()

import umap

reducer = umap.UMAP()

embedding = reducer.fit_transform(scaled_penguin_data)
embedding.shape

umap_df = pd.DataFrame(data={"species":penguins["species"],
                       "feature1":embedding[:, 0],
                       "feature2":embedding[:, 1]})

sns.scatterplot(data=umap_df, x="feature1", y="feature2", hue = "species", s=100).set(title='UMAP')
# plt.show()

f1 = "feature1"
f2 = "feature2"


gt_labels       = gt_label_indices(umap_df,"species")
kmeans_labels   = kmeans(umap_df,3,f1,f2)
spectral_labels = spectral(umap_df,3,f1,f2)
birch_labels    = birch(umap_df,3,f1,f2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = gt_labels,ax=axes[0,0])
axes[0,0].set_title("GT Labels")

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = kmeans_labels,ax=axes[0,1])
axes[0,1].set_title("K-Means")

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = spectral_labels,ax=axes[1,0])
axes[1,0].set_title("spectral")

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = birch_labels,ax=axes[1,1])
axes[1,1].set_title("BIRCH")

penguins_clusters = penguins.copy()
penguins_clusters["cluster_index"] = birch_labels

cluster = 0

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    # display(penguins_clusters.loc[penguins_clusters["cluster_index"]==cluster])
    print(penguins_clusters.loc[penguins_clusters["cluster_index"] == cluster])


# task 3
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import matplotlib.pyplot as plt
# plt.ion()
import seaborn as sns
import pandas as pd
# %matplotlib inline
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


fastfood = pd.read_csv("https://raw.githubusercontent.com/charlesanthony1996/billionaires_dataset/main/fastfood.csv")

fastfood.to_csv("fastfood.csv", index=False)
fastfood = pd.read_csv("fastfood.csv")
fastfood = fastfood.head(200)
print(fastfood.head())


# penguins = penguins.dropna()
# penguins.species.value_counts()
fastfood = fastfood.dropna()
fastfood.restaurant.value_counts()


# sns.pairplot(penguins.drop("year", axis=1), hue='species')
# sns.pairplot(fastfood.drop("salad", axis=1), hue='restaurant')
plt.tight_layout()
# plt.show()

from sklearn import cluster
f1 = "total_fat"
f2 = "protein"

gt_labels       = gt_label_indices(fastfood,"restaurant")
kmeans_labels   = kmeans(fastfood,3,f1,f2)
spectral_labels = spectral(fastfood,3,f1,f2)
birch_labels    = birch(fastfood,3,f1,f2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(data=fastfood, x=f1, y=f2, hue = gt_labels,ax=axes[0,0])
axes[0,0].set_title("GT Labels")

sns.scatterplot(data=fastfood, x=f1, y=f2, hue = kmeans_labels,ax=axes[0,1])
axes[0,1].set_title("K-Means")

sns.scatterplot(data=fastfood, x=f1, y=f2, hue = spectral_labels,ax=axes[1,0])
axes[1,0].set_title("spectral")

sns.scatterplot(data=fastfood, x=f1, y=f2, hue = birch_labels,ax=axes[1,1])
axes[1,1].set_title("BIRCH")
# plt.show()

# first we scale the data to normalize the range of values

fastfood_data = fastfood[
    [
        "total_fat",
        "protein",
        "calcium",
        "sugar",
    ]
].values

scaled_fastfood_data = StandardScaler().fit_transform(fastfood_data)

# print(scaled_fastfood_data)

from sklearn.decomposition import PCA
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py

X_reduced = PCA(n_components=2).fit_transform(scaled_fastfood_data)


print(X_reduced[:, 0].shape)


pca_df = pd.DataFrame(data={"restaurant":fastfood["restaurant"],
                       "feature1":X_reduced[:, 0],
                       "feature2":X_reduced[:, 1]})

sns.scatterplot(data=pca_df, x="feature1", y="feature2", hue = "restaurant", s=100).set(title='PCA')
# plt.show()

from sklearn import manifold


tsne = manifold.TSNE(n_components=3, init='pca', random_state=2)
X_tsne = tsne.fit_transform(scaled_fastfood_data)

tsne_df = pd.DataFrame(data={"restaurant":fastfood["restaurant"],
                       "feature1":X_tsne[:, 0],
                       "feature2":X_tsne[:, 1]})

sns.scatterplot(data=tsne_df, x="feature1", y="feature2", hue = "restaurant", s=100).set(title='TSNE')
# plt.show()

import umap

reducer = umap.UMAP()

embedding = reducer.fit_transform(scaled_fastfood_data)
embedding.shape

umap_df = pd.DataFrame(data={"restaurant":fastfood["restaurant"],
                       "feature1":embedding[:, 0],
                       "feature2":embedding[:, 1]})

sns.scatterplot(data=umap_df, x="feature1", y="feature2", hue = "restaurant", s=100).set(title='UMAP')
# plt.show()

f1 = "feature1"
f2 = "feature2"


gt_labels       = gt_label_indices(umap_df,"restaurant")
kmeans_labels   = kmeans(umap_df,3,f1,f2)
spectral_labels = spectral(umap_df,3,f1,f2)
birch_labels    = birch(umap_df,3,f1,f2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = gt_labels,ax=axes[0,0])
axes[0,0].set_title("GT Labels")

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = kmeans_labels,ax=axes[0,1])
axes[0,1].set_title("K-Means")

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = spectral_labels,ax=axes[1,0])
axes[1,0].set_title("spectral")

sns.scatterplot(data=umap_df, x=f1, y=f2, hue = birch_labels,ax=axes[1,1])
axes[1,1].set_title("BIRCH")


fastfood_clusters = fastfood.copy()
fastfood_clusters["cluster_index"] = birch_labels

cluster = 0

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # display(fastfood_clusters.loc[fastfood_clusters["cluster_index"]==cluster])
    print(fastfood_clusters.loc[fastfood_clusters["cluster_index"] == cluster])
