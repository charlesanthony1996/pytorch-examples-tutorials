from keras.applications import VGG16
import tensorflow as tf
tf.config.run_functions_eagerly(True)


conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

print(conv_base)

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/users/charles/downloads/dogs-vs-cats'
train_dir = os.path.join(base_dir, "train")
train_cats_dir = os.path.join(base_dir, "train/cat")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test1")

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode="binary"
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(labels_batch)
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)

# cats = os.listdir("/users/charles/downloads/dogs-vs-cats")
# print(len(cats))

