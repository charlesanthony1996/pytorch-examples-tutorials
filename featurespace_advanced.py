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


# preprocessing
train_ds_with_no_labels = train_ds.map(lambda x, _: x)

def example_feature_space(dataset, feature_space, feature_names):
    feature_space.adapt(dataset)
    for x in dataset.take(1):
        inputs = {feature_name: x[feature_name] for feature_name in feature_names}
        preprocessed_x = feature_space(inputs)
        print(f"Input: {[{k: v.numpy()} for k, v in inputs.items()]}")
        print(f"preprocessed output: {[{k:v.numpy()} for k, v in preprocessed_x.items()]}")

    
# feature hashing
feature_space = FeatureSpace(features={"campaign": FeatureSpace.integer_hashed(num_bins=4, output_mode="one_hot")}, output_mode="dict",)

example_feature_space(train_ds_with_no_labels, feature_space, ["campaign"])

# feature hashing can also be used for string features
feature_space = FeatureSpace(features= {"education": FeatureSpace.string_hashed(num_bins=4, output_mode="one_hot")}, output_mode="dict",)

example_feature_space(train_ds_with_no_labels, feature_space, ["education"])

feature_space = FeatureSpace(features={"age": FeatureSpace.float_discretized(num_bins=3, output_mode="one_hot")}, output_mode="dict")

example_feature_space(train_ds_with_no_labels, feature_space, ["age"])

# feature indexing
feature_space = FeatureSpace(features={"default": FeatureSpace.string_categorical(num_oov_indices=1, output_mode="one_hot")}, output_mode="dict")

example_feature_space(train_ds_with_no_labels, feature_space, ["default"])

feature_space = FeatureSpace(features= {"previously_contacted": FeatureSpace.integer_categorical(num_oov_indices=0, output_mode="one_hot")}, output_mode="dict")

example_feature_space(train_ds_with_no_labels, feature_space, ["previously_contacted"])

feature_space = FeatureSpace(features={
    "age": FeatureSpace.integer_hashed(num_bins=6,output_mode="one_hot"), 
    "job": FeatureSpace.string_categorical(num_oov_indices=0,
    output_mode="one_hot")},
    
    crosses=[
        FeatureSpace.cross(
            feature_names=("age", "job"),
            crossing_dim=8,
            output_mode="one_hot"
        )
    ],
    output_mode="dict"
)

example_feature_space(train_ds_with_no_labels, feature_space, ["age", "job"])

# feature space using a keras preprocessing layer
custom_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")

feature_space = FeatureSpace(features= {"education": FeatureSpace.feature(preprocessor=custom_layer, dtype="string", output_mode="float")}, output_mode="dict")

example_feature_space(train_ds_with_no_labels, feature_space, ["education"])

# configuring the final feature space
feature_space = FeatureSpace(
    features= {
        "previously_contacted": FeatureSpace.integer_categorical(num_oov_indices=0),
        "marital": FeatureSpace.string_categorical(num_oov_indices= 0),
        "education": FeatureSpace.string_categorical(num_oov_indices= 0),
        "default": FeatureSpace.string_categorical(num_oov_indices=0),
        "housing": FeatureSpace.string_categorical(num_oov_indices= 0),
        "loan": FeatureSpace.string_categorical(num_oov_indices=0),
        "contact": FeatureSpace.string_categorical(num_oov_indices=0),
        "month":FeatureSpace.string_categorical(num_oov_indices=0),
        "day_of_week": FeatureSpace.string_categorical(num_oov_indices=0),
        "poutcome": FeatureSpace.string_categorical(num_oov_indices=0),
        # categorical features to hash and bin
        "job": FeatureSpace.string_hashed(num_bins=3),
        
        "pdays": FeatureSpace.integer_hashed(num_bins=4),

        "age": FeatureSpace.float_discretized(num_bins=4),

        "campaign": FeatureSpace.float_normalized(),
        "previous": FeatureSpace.float_normalized(),
        "emp.var.rate": FeatureSpace.float_normalized(),
        "cons.price.idx": FeatureSpace.float_normalized(),
        "cons.conf.idx": FeatureSpace.float_normalized(),
        "euribor3m": FeatureSpace.float_normalized(),
        "nr.employed": FeatureSpace.float_normalized()
    },
    # special feature cross with a custom crossing dim
    crosses = [
        FeatureSpace.cross(feature_names=("age", "job"), crossing_dim= 8),
        FeatureSpace.cross(
            feature_names=("default", "housing", "loan"), crossing_dim=6
        ),
        FeatureSpace.cross(
            feature_names=("poutcome", "previously_contacted"), crossing_dim=2
        )
    ],
    output_mode="concat"
)

# adapt the feature space to the training data
train_ds = train_ds.batch(32)
valid_ds = valid_ds.batch(32)

train_ds_with_no_labels = train_ds.map(lambda x, _:x)
feature_space.adapt(train_ds_with_no_labels)

for x, _ in train_ds.take(1):
    # print(f"sample data: {x}")
    # if 'age' not in x:
    preprocessed_x = feature_space(x)
    print(f"preprocessed_x shape: ", {preprocessed_x.shape})
    print(f"preprocessed_x sample: \n {preprocessed_x[0]}")


feature_space.save("myfeaturespace.keras")


# preprocessing with feature space as part of the tf.data pipeline
preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls= tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)


preprocessed_valid_ds = valid_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls= tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# model
encoded_features = feature_space.get_encoded_features()
print(encoded_features)

x = tf.keras.layers.Dense(64, activation="relu")(encoded_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

print(output)

model = tf.keras.Model(inputs = encoded_features, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(preprocessed_train_ds, validation_data=preprocessed_train_ds, epochs=20, verbose=2)

loaded_feature_space = tf.keras.models.load_model("myfeaturespace.keras")

dict_inputs = loaded_feature_space.get_inputs()
encoded_features = loaded_feature_space.get_encoded_features()
print(encoded_features)


print(dict_inputs)

outputs = model(encoded_features)
# print(outputs)
inference_model = tf.keras.Model(inputs=dict_inputs, outputs=outputs)


sample = {
    "age": 30,
    "job": "blue-collar",
    "marital": "married",
    "education": "basic.9y",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "fri",
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1,
    "previously_contacted": 0, 
}

input_dict = { name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = inference_model.predict(input_dict)

print(f"This particular client has a {100 * predictions[0][0]:.2f}% probability of subscribing a term deposit, as evaluated by our model")

