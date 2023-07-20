import tensorflow as tf
import numpy as np
import io
import os
import re
import matplotlib.pyplot as plt
import string
import evaluate
import time
from numpy import random
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
# from google.colab import drive, files
from datasets import load_dataset
from transformers import create_optimizer, T5TokenizerFast, DataCollatorForSeq2Seq, TFT5ForConditionalGeneration, TFAutoModelForSeq2SeqLM,  AutoModelForSeq2SeqLM


batch_size = 64
max_length= 128

dataset_id = "leslyarun/c4_200m_gec_train100k_test25k"

dataset = load_dataset(dataset_id)

# print(dataset)

# print(dataset["train"][0])

model_id = "t5-small"

tokenizer = T5TokenizerFast.from_pretrained(model_id)

# print(tokenizer)

def preprocess_function(examples):
    inputs = [example for example in examples["input"]]
    targets = [example for example in examples["output"]]

    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)

    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# print(tokenized_dataset)

tokenized_dataset["train"][1000]

# print(tokenized_dataset["train"][1])

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_id)
data_collator = DataCollatorForSeq2Seq(tokenizer= tokenizer, model=model, return_tensors="tf")

train_dataset = tokenized_dataset["train"].to_tf_dataset(
    shuffle=True,
    batch_size = batch_size,
    collate_fn= data_collator,
)

# print(train_dataset)

val_dataset = tokenized_dataset["test"].to_tf_dataset(
    shuffle=True,
    batch_size = batch_size,
    collate_fn = data_collator,
)

# print(val_dataset)

# for i in val_dataset.take(1):
#     print(i)

# model.summary()

num_epochs = 1
num_train_steps = len(train_dataset) * num_epochs

# print(num_train_steps)

optimizer, schedule = create_optimizer(
    init_lr = 2e-5,
    num_warmup_steps = 0,
    num_train_steps = num_train_steps,
)

# print(optimizer)

# print(schedule)

model.compile(optimizer= optimizer)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=num_epochs
)

plt.plot(history.history["loss"])


