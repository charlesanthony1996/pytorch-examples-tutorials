import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import keras
from keras import layers
# from keras import ops

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

x_train, x_val, y_train, y_val = (x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:])

# create the model
embedding_size = 50

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer="he_normal", embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_initializer="he_normal", embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.movie_bias = layers.Embedding(num_movies, 1)


    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias

        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_movies, embedding_size)
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001))


# train the model based on the data split
history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=5, verbose= 1, validation_data=(x_val, y_val),)

print(history)

# plot training and validation loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
# plt.show()

# show top 10 movie recommendations to a user
movie_df = pd.read_csv(movielens_dir / "movies.csv")

# let us go get a user and see the top recommendations
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]

movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

user_encoder = user2user_encoded.get(user_id)

user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))

ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]

recommend_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

print("showing recommendations for user: {}".format(user_id))
print("===" * 9)
print("movies with high ratings from user")
print("---" * 8)

top_movies_user = (movies_watched_by_user.sort_values(by="rating" , ascending=False).head(5).movieId.values)

movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)


print("----" * 8)
print("top 10 recommendations")
print("---" * 8)

recommended_movies = movie_df[movie_df["movieId"].isin(recommend_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)




