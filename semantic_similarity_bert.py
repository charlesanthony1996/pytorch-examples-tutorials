import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

# configuration
max_length = 128
batch_size = 32
epochs = 1


# there are more than 550k samples in total: we will use 100k for this example
train_df = pd.read_csv("/users/charles/downloads/SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
# print(train_df.head())
valid_df = pd.read_csv("/users/charles/downloads/SNLI_Corpus/snli_1.0_dev.csv")
test_df = pd.read_csv("/users/charles/downloads/SNLI_Corpus/snli_1.0_test.csv")


# shape of the data
# print(f"total train samples: ", {train_df.shape[0]})
# print(f"total validation samples: ", {valid_df.shape[0]})
# print(f"total test samples: ", {test_df.shape[0]})


# look at one sample from the dataset
# print(f"sentence 1: {train_df.loc[1, 'sentence1']}")
# print(f"sentence 2: {train_df.loc[1, 'sentence2']}")
# print(f"sentence 3: {train_df.loc[1, 'similarity']}")



# preprocessing
# we have some nan entries in our train data, we will simply drop them
# print("Number of missing values")
# print(train_df.isnull().sum())
train_df.dropna(axis = 0,inplace=True)

# distribution of our training targets
# print("Train target distribution")
# print(train_df.similarity.value_counts())

# print("validation target distribution")
# print(valid_df.similarity.value_counts())

train_df = ( train_df[train_df.similarity!= '-'].sample(frac=1.0, random_state= 42).reset_index(drop=True))

# print(train_df)

valid_df = (valid_df[valid_df.similarity != '-'].sample(frac=1.0, random_state= 42).reset_index(drop=True))

# print(valid_df)

# one hot encoding, validation and test labels
train_df["label"] = train_df["similarity"].apply(lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2)

y_train = tf.keras.utils.to_categorical(train_df.label, num_classes = 3)

valid_df["label"] = valid_df["similarity"].apply(lambda x: 0 if x == "categorical" else 1 if x == "entailment" else 2)

y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes = 3)

test_df["label"] = test_df["similarity"].apply(lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2)

y_test = tf.keras.utils.to_categorical(test_df.label , num_classes = 3)

# create a custom data generator
class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sentence_pairs, labels, batch_size = batch_size, shuffle=True, include_targets=None):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets

        # load our bert tokenizer to encode the text
        # we will use base base uncased pretrained model
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()


    def __len__(self):
        # denotes the number of batches per epoch
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # retrieves the batch of index
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]


        # with berts tokenizers batch encode plus batch of both the sentences are
        # encoded together and seperated by [SEP] token
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens= True,
            max_length = max_length,
            # padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf"
        )

        # print(encoded)

        # convert batch of encoded features to a numpy array
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")


        # set to true if data generator is used for training/validation
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels

        else:
            return [input_ids , attention_masks, token_type_ids]


    def on_epoch_end(self):
        # shuffle indexes after each epoch if shuffle is set to true
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)

        
# create the model under a distribution strategy scope
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # encoded token ids from bert tokenizer
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")

    # attention masks indicates to the model which tokens should be attended to
    attention_masks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_masks")

    # token type ids are binary masks identifying different sequences in the model
    token_type_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="token_type_ids")

    # loading pretrained bert model
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    # freeze the bert model to reuse the pretrained features without modifying them
    bert_model.trainable = False

    bert_output = bert_model.bert(input_ids, attention_masks, token_type_ids = token_type_ids)

    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    # add trainable layers on top of frozen layers to adapt the pretrained features on the new data
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(sequence_output)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(inputs = [input_ids, attention_masks, token_type_ids], outputs=output)


    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["acc"])


print(f"Strategy: {strategy}")
model.summary()

# create train and validation data generators
train_data = BertSemanticDataGenerator(train_df[["sentence1", "sentence2"]].values.astype("str"), y_train, batch_size=batch_size, shuffle=True, include_targets=True)

valid_data = BertSemanticDataGenerator(train_df[["sentence1", "sentence2"]].values.astype("str"), y_val, batch_size=batch_size, shuffle = False, include_targets=True)



# train the model
history = model.fit(train_data, validation_data = valid_data,epochs= epochs, use_multiprocessing=True, workers=-1)

print(history)

# evaluate model on the test set
test_data = BertSemanticDataGenerator(
    test_data[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False
)

model.evaluate(test_data, verbose= 1)


# inference on custom sentences
def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size = 1, shuffle=False, include_targets=False
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f%}"
    pred = labels[idx]
    return pred, proba



sentence1 = "Two men are looking at something"
sentence2 = "The two men are observing at with their eyes closed at something"

check_similarity(sentence1, sentence2)


