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


