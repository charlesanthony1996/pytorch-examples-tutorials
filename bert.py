from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from helpers import get_gpu

# Get the GPU device
device = get_gpu()

# Define the file path (update this to the correct path if necessary)
data_dir = Path("/Users/charles/Downloads")
file_path = data_dir / "BBC News Train.csv"

# Check if the file exists and load the CSV file into a DataFrame
if file_path.is_file():
    data = pd.read_csv(file_path)
    print("File loaded successfully!")
else:
    print(f"File not found at {file_path}. Please check the file path.")
    data = pd.DataFrame()  # Empty DataFrame as a fallback

# Define the labels
labels = {"business": 0, "entertainment": 1, "sport": 2, "tech": 3, "politics": 4}
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Print the directory for debugging
print(data_dir)

# Print the first few rows of the DataFrame (if loaded successfully)
if not data.empty:
    print(data.head())
else:
    print("Data frame is empty. Please check the file path and contents.")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.labels = [labels[category] for category in data["Category"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            for text in data["Text"]
        ]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], np.array(self.labels[idx])
    

class BERTClassifier(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.fc(self.dropout(pooled_output))
        return output
    

train_df, valid_df, test_df = np.split(
    data.sample(frac=1, random_state=1337), [int(0.8 * len(data)), int(0.9 * len(data))]
)

train_dataloader = torch.utils.data.DataLoader(Dataset(train_df), batch_size=2, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(Dataset(valid_df), batch_size=2)

criterion = nn.CrossEntropyLoss()
model = BERTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

for epoch in range(1):
    total_train_loss = 0
    total_train_correct = 0
    for train_input, train_label in tqdm(train_dataloader):
        mask = train_input["attention_mask"].squeeze(1).to(device)
        input_id = train_input["input_ids"].squeeze(1).to(device)
        train_label = train_label.to(device)

        optimizer.zero_grad()
        output = model(input_id, mask)
        
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        total_train_correct += (output.argmax(1) == train_label).sum().item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_train_loss/len(train_dataloader)}, Accuracy: {total_train_correct/len(train_df)}")

# Validation loop (optional)
