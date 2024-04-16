## requirements.txt
# tensorflow
# numpy
# sentencepiece
# transformers

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
from transformers import XLNetTokenizer, TFXLNetForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
# tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


# example to training dataset
examples = [
    {"text": "I loved the movie", "label": 1},
    {"text": "I hated the movie", "label": 0}
]

# tokenization and data preparation
def encode_example(example):
    encoding = tokenizer.encode_plus(
        example['text'],
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='tf'
    )
    return {'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0]}, example['label']

# convert examples to tensorFlow dataset
dataset = tf.data.Dataset.from_generator(
    lambda: (encode_example(ex) for ex in examples),
    output_signature=(
        {'input_ids': tf.TensorSpec(shape=(128,), dtype=tf.int32),
         'attention_mask': tf.TensorSpec(shape=(128,), dtype=tf.int32)},
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# batching
dataset = dataset.batch(2)

# loading the five models here
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
# model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
# model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# model = TFAlbertForSequenceClassification.from_pretrained("albert-base-v1")
# model = TFXLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
# model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# training and params
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# training
model.fit(dataset, epochs=3)

# testing data
test_examples = [
    "This is the best movie I have ever seen!",
    "Absolutely terrible! I will never watch this again.",
    "It was okay, not great but not bad either."
]

# preprocessing
test_encodings = tokenizer(test_examples, truncation=True, padding=True, max_length=128, return_tensors='tf')

# creating a tf dataset
test_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},
    tf.constant([0] * len(test_examples))
))
test_dataset = test_dataset.batch(3)

# prediction
predictions = model.predict(test_dataset)

# decoding predictions
predicted_labels = np.argmax(predictions.logits, axis=1)
sentiments = ["Positive" if label == 1 else "Negative" for label in predicted_labels]

# logging sentiment values
for sentence, sentiment in zip(test_examples, sentiments):
    print(f"sentence: \"{sentence}\" - sentiment: {sentiment}")


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Example data
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset and dataloader
texts = ["I loved the movie", "I hated the movie"]
labels = [1, 0]  # Assuming binary classification
dataset = SentimentDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = criterion(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Predictions
model.eval()
with torch.no_grad():
    inputs = tokenizer(["what the fuck man?"], return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print("Prediction:", predictions.item())
