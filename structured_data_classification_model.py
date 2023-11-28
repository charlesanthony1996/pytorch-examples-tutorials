import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

# first load the data and apply preprocessing
movielens_data_file_url = ("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")

movielens_zipped_file = keras.utils.get_file("ml-latest-small.zip", movielens_data_file_url, extract=False)

keras_datasets_path = Path(movielens_zipped_file).parents[0]
movielens_dir = keras_datasets_path / "ml-latest-small"

if not movielens_dir.exists():
    with ZipFile(movielens_zipped_file, "r") as zip:
        print("extracting all the files now")
        zip.extractall(path=keras_datasets_path)
        print("Done!")


ratings_file = movielens_dir / "ratings.csv"

df = pd.read_csv(ratings_file)
# print(df.head())
# print(ratings_file)


# encoding users and movies as integer indices
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

df["rating"] = df["rating"].values.astype(np.float32)

min_rating = min(df["rating"])
max_rating = max(df["rating"])


print("Number of users: {}, number of movies: {}, min rating: {}, max rating: {}".format(num_users, num_movies, min_rating, max_rating))


# prepare training and validation data
df = df.sample(frac= 1, random_state=42)
x = df[["user", "movie"]].values

# normalize the targets between 0 and 1. makes it easy to train
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# assuming training on 90% of the data and validating on 10%
train_indices = int(0.9 * df.shape[0])

x_train, x_val, y_train, y_val = 
