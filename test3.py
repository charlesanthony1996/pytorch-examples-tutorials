# Import necessary libraries
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import wandb
wandb.login(key="1bc2afc44c8734c5c661f6f44f3609b88903b8d8")


# Function to tokenize and pad texts
def tokenize_and_pad(text_list):
    return tokenizer(text_list, max_length=64, truncation=True, padding="max_length", return_tensors="pt")

# Load dataset
dataset = load_dataset('csv', data_files={'train': '/users/charles/downloads/labeled_data.csv'})

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization and padding of the dataset
def preprocess_function(examples):
    return tokenize_and_pad(examples['text'])

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir='./logs',
    logging_steps=10,
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Optionally, save the model
model.save_pretrained('./saved_model')

# Evaluate the model if validation data is available
# results = trainer.evaluate()
# print(results)

eval_dataset = load_dataset('csv', data_files={'validation': 'validation_data.csv'})
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_eval_dataset['validation'],  # Make sure to pass this
    data_collator=data_collator
)


if 'validation' in tokenized_datasets:
    results = trainer.evaluate()
    print("Evaluation Results:", results)
else:
    print("No validation dataset provided; skipping evaluation.")


try:
    trainer.train()
    if 'validation' in tokenized_datasets:
        results = trainer.evaluate()
        print("Evaluation Results:", results)
except Exception as e:
    print(f"An error occurred: {e}")
