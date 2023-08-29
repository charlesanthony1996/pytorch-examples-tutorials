import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.style.use("ggplot")

positive_class = 1
bag_count = 1000
val_bag_count = 300
bag_size = 3
plot_size = 3
ensemble_avg_count = 1

# prepare bags

def create_bags(input_data, input_labels, positive_class, bag_count, instance_count):

    # set up bags
    bags = []
    bag_labels = []

    # normalize input data
    input_data = np.divide(input_data, 255.0)

    # count positive samples
    count = 0

    for _ in range(bag_count):

        # pick a fixed size random subset of samples
        index = np.random.choice(input_data.shape[0], instance_count, replace=False)
        instances_data = input_data[index]
        instances_labels = input_labels[index]

        # by default , all bags are labeled as 0
        bag_label = 0

        # check if there is at least a positive class in the bag
        if positive_class in instances_labels:

            # positive bag will be labeled as 1
            bag_label = 1
            count += 1

        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))


    print(f"Positive bags: {count}")
    print(f"Negative steps: {bag_count - count}")

    return list(np.swapaxes(bags, 0, 1)), np.array(bag_labels)


# load the mnist dataset
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

# create training data
train_data, train_labels = create_bags(
    x_train, y_train, positive_class, bag_count, bag_size
)

# print(train_data)

# print(train_labels)

# create validation data
val_data, val_labels = create_bags(
    x_val, y_val, positive_class, val_bag_count, bag_size
)

# print(val_data)

# print(val_labels)


class MILAttentionLayer(layers.Layer):
    def __init__(self,weight_params_dim,kernel_initializer="glorot_uniform",kernel_regularizer=None,use_gated=False,**kwargs):
        super().__init__(**kwargs)
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    
    def build(self, input_shape):
        # input shape
        # list of 2D tensors with shape: (batch_size, input_dim)
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer = self.v_init,
            name="v",
            regularizer = self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer = self.w_init,
            name="w",
            regularizer = self.w_regularizer,
            trainable= True
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer = self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True
            )

        else:
            self.u_weight_params = None

        self.input_built = True


    def call(self, inputs):

        # assigning variables from the number of inputs
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # apply softmax over instances such that the output summation is equal to 1
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]


    def compute_attention_scores(self, instance):

        # reserve in case "gated" mechanism used
        original_instance = instance

        # print shapes
        print("shapes:")
        print("original instances: ", original_instance.shape)
        print("v_weight_params: ", self.v_weight_params.shape)
        if self.use_gated:
            print("u_weight_params: ", self.u_weight_params.shape)


        # tanh(v * h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes= 1))

        # for learning non-linear relations efficiently
        if self.use_gated:

            gate = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes = 1)
            )
            instance = instance * gate
            attention_scores = tf.tensordot(instance , self.w_weight_params, axes= 1)
            
            # return tf.tensordot(instance, self.w_weight_params, axes= 1)
            return attention_scores


def plot(data, labels, bag_class, predictions=None, attention_weights=None):

    labels = np.array(labels).reshape(-1)

    if bag_class == "positive":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:plot_size]]


        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:plot_size]]

    elif bag_class == "negative":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:plot_size]]

        else:
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[:, labels[0:plot_size]]


    else:
        print(f"There is no class {bag_class}")
        return

    
    print(f"The bag class label is {bag_class}")
    for i in range(plot_size):
        figure = plt.figure(figsize=(8, 8))
        print(f"Bag number: {labels[i]}")
        for j in range(bag_size):
            image = bags[j][i]
            figure.add_subplot(1, bag_size, j+ 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i][j], 2]))
            plt.imshow(image)
        plt.show()


# plot some validation data bags per class
# plot(val_data, val_labels, "positive")
# plot(val_data, val_labels, "negative")


def create_model(instance_shape):
    # extract features from inputs

    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(128, activation="relu")
    shared_dense_layer_2 = layers.Dense(64, activation="relu")
    
    for _ in range(bag_size):
        inp = layers.Input(instance_shape)
        flatten = layers.Flatten()(inp)
        dense_1 = shared_dense_layer_1(flatten)
        dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
        embeddings.append(dense_2)


    # invoke the attention layer
    alpha = MILAttentionLayer(
        weight_params_dim = 256,
        kernel_regularizer= keras.regularizers.l2(0.01),
        use_gated =True,
        name="alpha",
    )(embeddings)

    # reshape alpha to match the shape of embeddings
    reshaped_alpha = [
        layers.Reshape(target_shape=(1, bag_size, 1, alpha[i].shape[-1]))(alpha[i][:, None,None, :])
        for i in range(len(alpha))
    ]

    concat_alpha = layers.concatenate(reshaped_alpha, axis= 1)



    # multiply attention weights with the input layers
    # multiply_layers = [
    #     layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    # ]

    multiply_layers = [
        layers.multiply([concat_alpha[:, i], embeddings[i]]) for i in range(len(embeddings))
    ]



    # concatenate layers
    concat = layers.concatenate(multiply_layers, axis = 2)

    # classification output node
    output = layers.Dense(2, activation="softmax")(concat)

    return keras.Model(inputs, output)



def compute_class_weights(labels):

    # count number of positive and negative bags
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[1])
    total_count = negative_count + positive_count

    # build class weight dictionary
    return {
        0: (1 / negative_count) * (total_count/ 2),
        1: (1 / positive_count) * (total_count/ 2),
    }


# build and train the model
def train(train_data, train_labels, val_data, val_labels, model):

    # train model
    # prepare callbacks
    # path where to save best weights

    # take the file name from the wrapper
    file_path = "/tmp/best_model_weights.h5"

    # initialize model checkpoint callback
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose = 0,
        mode="min",
        save_best_only=True,
        save_weights_only=True
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience= 10, mode="min"
    )

    # compile model
    model.compile(
        optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )

    
    # fit model
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=20,
        class_weight=compute_class_weights(train_labels),
        batch_size = 1,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )

    # load best weights
    model.load_weights(file_path)

    return model


# building models
instance_shape = train_data[0][0].shape
models = [create_model(instance_shape) for _ in range(ensemble_avg_count)]

# show single model architecture
print(models[0].summary())

# training models
trained_models = [
    train(train_data, train_labels, val_data , val_labels, model)
    for model in tqdm(models)
]




