from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os



# define hyperparameters
img_size = 224
batch_size = 64
epochs = 10

max_seq_length = 20
num_features = 2048

# data preparation
train_df = pd.read_csv("/users/charles/train.csv")
test_df = pd.read_csv("/users/charles/test.csv")

print(f"Total videos for training: ", len(train_df))
print(f"total videos for testing: ", len(test_df))

# print(train_df.head())
# print(test_df.head())

train_df.sample(10)
test_df.sample(10)


def crop_centre_square(frame):
    y, x = frame.shape[0: 2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_x + min_dim , start_x : start_x + min_dim]


def load_video(path, max_frames = 0, resize=(img_size, img_size)):
    cap = cv2.VideoCapture(path)
    frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break

    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights = "imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3)
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((img_size, img_size, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature extraction")

feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices = 0, vocabulary=np.unique(train_df["tag"])
)

print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # frame masks and frame features are what we will feed to our sequence model
    # frame masks will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not

    frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype="bool")
    frame_features = np.zeros(
        shape = (num_samples, max_seq_length, num_features), dtype="float32"
    )

    # for each video
    for idx, path in enumerate(video_paths):
        # gather all its frames and add a batch dimension
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # initialize placeholders to store the masks and features of the current video
        temp_frame_mask = np.zeros(shape=(1, max_seq_length), dtype="bool")
        temp_frame_features = np.zeros(
            shape =(1, max_seq_length, num_features), dtype="float32"
        )

        # extract features from the frames of the current video
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(max_seq_length, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )

            # 1 
            temp_frame_mask[i, :length] = 1

        frame_features[idx] = temp_frame_features.squeeze()
        frame_masks[idx] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels



train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")


print(f"frame features in train set: ", {train_data[0].shape})
print(f"frame features in test set: ", {test_data[1].shape})





# the sequence model
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((max_seq_length, num_features))
    mask_input = keras.Input((max_seq_length,), dtype="bool")

    # refer to the following tutorial to understand he significance of using mask
    # https://keras.io/api/layers/recurrent_layers/gru/

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask = mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return rnn_model

# utility for running experiences
def run_experiment():
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose= 1
    )
    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs = epochs,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy =  seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"test accuracy :, {round(accuracy * 100, 2)}%")


    return history, seq_model


_, sequence_model = run_experiment()

# inference
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, max_seq_length), dtype="bool")
    frame_features = np.zeros(shape= (1, max_seq_length, num_features), dtype="float32")


    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(max_seq_length, video_length)

        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])

        frame_mask[i, :length] = 1

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()
    
    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print()








