import keras
import tensorflow as tf
from pathlib import Path
from zipfile import ZipFile
from tensorflow.keras.utils import FeatureSpace


# print(keras.__version__)

# load the data
data_url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
data_zipped_path = 