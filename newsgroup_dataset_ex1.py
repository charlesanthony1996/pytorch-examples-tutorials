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

# define the classifier model
class Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
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
        x = self.softmax(x)
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




# example usage
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 5

model = Classifier(vocab_size, embedding_dim, hidden_dim, output_dim)

print(model)


# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# training loop
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

