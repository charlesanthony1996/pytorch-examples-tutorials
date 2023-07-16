# torch is not needed
# import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import datetime
import pathlib
import io
import os
import re
import string
import time
from numpy import random
import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Flatten, InputLayer, BatchNormalization, Dropout, Input, LayerNormalization
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
# from google.colab import drive
# from google.colab import files
from datasets import load_dataset
from transformers import (DataCollatorWithPadding, create_optimizer, DebertaTokenizerFast)

# print("imports are done")

batch_size = 32
max_length = 512

dataset = load_dataset("covid_qa_deepset")

# print(dataset)

# print(dataset["train"][0])

answer = "Mother to child transmission MTCT"
# answer = "hello"
# print(len(answer) + 370)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# print(tokenizer)


model_id = "microsoft/deberta-base"
tokenizer = DebertaTokenizerFast.from_pretrained(model_id)

# print(tokenizer)

from transformers import LongformerTokenizerFast, TFLongformerForQuestionAnswering
model_id = "allenai/longformer-large-4096-finetuned-triviaqa"
tokenizer = LongformerTokenizerFast.from_pretrained(model_id)

# print(tokenizer)

tokenized_examples = tokenizer(
    dataset["train"][0]["question"],
    dataset["train"][0]["context"],
    truncation="only_second",
    max_length=max_length,
    stride=64,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)


# print(tokenized_examples)

print(len(tokenized_examples["input_ids"]))

list_token = tokenizer.tokenize("[CLS] What is the main cause of HIV-1 infection?")
print(list_token)
for i in range(len(list_token)):
    if list_token[i] == "ĠChildren":
        print(i)

tokenizer.encode("What is the main cause of HIV-1 infection?")

for ids in tokenized_examples["input_ids"]:
    print(ids)
    print("-->", tokenizer.decode(ids))
    # break

offset_mapping_list = [(0, 4), (4, 7), (7, 11), (11, 16), (16, 22), (22, 25), (25, 29), (29, 30), (30, 31), (31, 34)]
print(len(offset_mapping_list))


print(tokenized_examples)

sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
offset_mapping = tokenized_examples.pop("offset_mapping")

tokenized_examples["start_positions"] = []
tokenized_examples["end_positions"] = []

for i, offsets in enumerate(offset_mapping):
    if len(dataset["train"][0]["answers"]["answer_start"]) == 0:
        tokenized_examples["start_positions"].append(0)
        tokenized_examples["end_positions"].append(0)
    
    else:
        start_char = dataset["train"][0]["answers"]["answer_start"][0]
        end_char = start_char+ len(dataset["train"][0]["answers"]["text"][0])
        found = 0
        start_token_position = 0
        end_token_position = 0

        for j, offset in enumerate(offsets):
            if offset[0] <= start_char and offset[1]>= start_char and found == 0:
                start_token_position = j
                end_token_position = max_length

                found = 1
            if offset[1] >= end_char and found == 1:
                end_token_position = j
                break

        tokenized_examples["start_positions"].append(start_token_position)
        tokenized_examples["end_positions"].append(end_token_position)

print(tokenized_examples["start_positions"])
print(tokenized_examples["end_positions"])


print(tokenized_examples["input_ids"])

def preprocess_function(dataset):
    questions = [q.lstrip() for q in dataset["question"]]
    paragraphs = [p.lstrip() for p in dataset["context"]]

    tokenized_examples = tokenizer(
        questions,
        paragraphs,
        truncation = "only_second",
        max_length=max_length,
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]

        start_char = dataset["answers"][sample_index]["answer_start"][0]
        end_char = start_char + len(dataset["answers"][sample_index]["text"][0])

        found = 0
        start_token_position = 0
        end_token_position = 0


        for j, offset in enumerate(offsets):
            if offset[0] <= start_char and offset[1] >= start_char and found == 0:
                start_token_position = j
                end_token_position = max_length
                found = 1
            if offset[1] >= end_char and found == 1:
                end_token_position = j
                break

        tokenized_examples["start_positions"].append(start_token_position)
        tokenized_examples["end_positions"].append(end_token_position)


    return tokenized_examples


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns= dataset["train"].column_names
)


print(tokenized_dataset)

tf_dataset = tokenized_dataset["train"].to_tf_dataset(
    shuffle=True,
    batch_size=batch_size
)

for i in tf_dataset.take(10):
    print(i)


train_dataset = tf_dataset.take(int(0.9*len(tf_dataset)))

val_dataset = tf_dataset.skip(int(0.9*len(tf_dataset)))

# modelling

from transformers import LongformerTokenizerFast, TFLongformerForQuestionAnswering

model = TFLongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

print(model)

print(model.summary())


optimizer = Adam(learning_rate = 0.00001)
model.compile(optimizer=optimizer)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)


# evaluation
from evaluate import load

squad_metric = load("squad")
predictions =[{"prediction":"1999", "id": "56e10a3be3433e1400422b22"}]
references = [{"answers": {"answer_start": [97], "text":["1976"]}, "id": "56e10a3be3433e1400422b22"}]
results = squad_metric.compute(predictions= predictions , references=references)

# testing

question="how is the virus spread?"
text="We know that the disease is caused by the SARS-CoV-2 virus"
inputs = tokenizer(question, text, return_tensors= "tf")

outputs = model(**inputs)

answer_start = int(tf.math.argmax(outputs.start_logits, axis= -1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis= 1)[0])

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(predict_answer_tokens)
tokenizer.decode(predict_answer_tokens)

