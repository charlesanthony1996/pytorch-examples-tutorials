import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm

# define the classifier model
class Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # uncomment this layer, just for testing
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        # x = self.softmax(x)
        return x


# fetch dataset here
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = data.data
labels = data.target

# split target
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels,test_size=0.2,  random_state=42)


# tokenization and vocabulary testing
tokenizer = get_tokenizer('basic_english')
# why the unk? -> 
vocab = build_vocab_from_iterator(map(tokenizer, train_texts), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# function to encode texts
def encode(texts, vocab, tokenizer):
    return [vocab(tokenizer(text)) for text in texts]

sequences = encode(texts, vocab, tokenizer)

# find the maximum sequence length after tokenization to set as max_len
max_len = 500
sequences_padded = pad_sequence([torch.tensor(seq)[:max_len] for seq in sequences], batch_first=True, padding_value=vocab["<pad>"])


# split the data
x_train_val , x_test, y_train_val, y_test = train_test_split(sequences_padded, labels, test_size = 0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=42)


# # convert to tensors and pad
y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_val_torch = torch.tensor(y_val, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)


# create datasets and loaders
train_dataset = TensorDataset(x_train, y_train_torch)
val_dataset = TensorDataset(x_val, y_val_torch)
test_dataset = TensorDataset(x_test, y_test_torch)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=True, num_workers=0)

# model, loss and optimizer
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = len(np.unique(labels))

# classifier class and using the top params
model = Classifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)


# training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss, total_correct, total_samples  = 0, 0, 0
    print("okay")
    for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
        optimizer.zero_grad()
        # print(texts.shape)
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)


    train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples
    print(f"Epoch {epoch + 1}: Train loss: {train_loss}, Train accuracy: {train_accuracy}")

    # validation
    # model.eval()
    # val_loss, val_correct, val_samples = 0, 0, 0
    # with torch.no_grad():
    #     for texts, labels in tqdm(val_loader , desc="Validation", leave=False):
    #         outputs = model(texts)
    #         loss = criterion(outputs, labels)

    #         val_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         val_correct += (predicted == labels).sum().item()
    #         val_correct += labels.size(0)

    # val_loss /= len(val_loader)
    # val_accuracy = val_correct / val_samples
    # print(f"Validation samples: {val_loss}, Validation Accuracy: {val_accuracy}")






