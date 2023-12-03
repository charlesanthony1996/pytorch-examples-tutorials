import keras
import tensorflow as tf
from pathlib import Path
from zipfile import ZipFile
from tensorflow.keras.utils import FeatureSpace
import pandas as pd


# print(keras.__version__)

# load the data
data_url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
data_zipped_path = tf.keras.utils.get_file("bank_marketing.zip", data_url, extract=True)

keras_datasets_path = Path(data_zipped_path).parents[0]

with ZipFile(f"{keras_datasets_path}/bank-additional.zip", "r") as zip:
    # extract files
    zip.extractall(path=keras_datasets_path)


dataframe = pd.read_csv(f"{keras_datasets_path}/bank-additional/bank-additional.csv", sep=";")
print(dataframe.head())

# dropping duration to avoid target leak
dataframe.drop("duration", axis=1, inplace=True)
# creating the new feature previously contacted
dataframe["previously_contacted"] = dataframe["pdays"].map(lambda x: 0 if x == 999 else 1)

# print a preview
print(f"Dataframe shape: {dataframe.shape}")
print(dataframe.head())

# train validation split
valid_dataframe = dataframe.sample(frac=0.2, random_state=0)
train_dataframe = dataframe.drop(valid_dataframe.index)

print(f"using {len(train_dataframe)} sample dataframe" f"{len(valid_dataframe)} for validation")


# generating tf datasets
label_lookup = tf.keras.layers.StringLookup(vocabulary=["no", "yes"], num_oov_indices=0)
print(label_lookup)

def encode_labels(x, y):
    encoded_y = label_lookup(y)
    return x, encoded_y

# encode_labels()

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("y")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.map(encode_labels, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size = len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
valid_ds = dataframe_to_dataset(valid_dataframe)

print(len(train_ds))
print(len(valid_ds))

for x, y in dataframe_to_dataset(train_dataframe).take(1):
    print(f"Input: {x}")
    print(f"Target: {y}")


