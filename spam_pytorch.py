import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader


# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Preprocess the data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
messages = df['message'].values
labels = df['label'].values

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages) # learning the vocabulary from the text data - learns unique words 
sequences = tokenizer.texts_to_sequences(messages) #converts it into a sequence of numerical indices. Each word in the text is replaced by its corresponding index in the learned vocabulary.

# Pad sequences
max_len = 100
sequences_padded = pad_sequences(sequences, maxlen=max_len) # ensure that input texts have the same length padding = adding zeros, trunctuation = removing words. Needed for the batch processing to ensure uniform input dimensions. 

# Split your data into train, validation, and test sets -> 70, 20, 10 split
# Split the data into 90% for training+validation and 10% for testing
X_train_val, X_test, y_train_val, y_test = train_test_split(sequences_padded, labels, test_size=0.1, random_state=42)

# Split the 90% training+validation into 70% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.22, random_state=42)  # ≈0.22 to make it 20% of the whole

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.int64) # Classes
y_train_torch = torch.tensor(y_train, dtype=torch.long)
X_val_torch = torch.tensor(X_val, dtype=torch.int64)
y_val_torch = torch.tensor(y_val, dtype=torch.long)
X_test_torch = torch.tensor(X_test, dtype=torch.int64)
y_test_torch = torch.tensor(y_test, dtype=torch.long)


# Create TensorDatasets for each set
train_dataset = TensorDataset(X_train_torch, y_train_torch)
val_dataset = TensorDataset(X_val_torch, y_val_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)


# Create DataLoaders for each set
batch_size = 32  # You can adjust the batch size if needed

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the model
class SMSClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SMSClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # An embedding layer that turns token indices into dense vectors of a fixed size (embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True) #batch_first=True indicates that the input tensors will have the batch size as the first dimension.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        """
        : (first colon): This means "take all elements" along the batch size dimension. It's selecting all sequences in the batch.
        -1: This specifies the index of the timestep to select. -1 is Python's way of indexing the last item, so this selects the last timestep from each sequence.
         --> the last timestep's output is often used for making a decision or prediction because it's assumed to contain information from the entire sequence.
        : (second colon): This again means "take all elements" along the hidden dimension, so it's selecting the entire hidden state vector of the last timestep"""
        
        
        x = self.fc(x)
        return x


# Define the train function with validation
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, n_epochs):
    train_losses, train_accuracy, train_f1 = [], [], []
    val_losses, val_accuracy, val_f1 = [], [], []
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss, epoch_correct, epoch_samples, epoch_f1 = 0, 0, 0, []
        for texts, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Training]', leave=False):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_samples += labels.size(0)
            epoch_f1.append(f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro'))
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracy.append(epoch_correct / epoch_samples)
        train_f1.append(np.mean(epoch_f1))

        # Validation
        model.eval()
        val_loss, val_correct, val_samples, val_f1_scores = 0, 0, 0, []
        with torch.no_grad():
            for texts, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Validation]', leave=False):
                outputs = model(texts)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples += labels.size(0)
                val_f1_scores.append(f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro'))
        val_losses.append(val_loss / len(val_loader))
        val_accuracy.append(val_correct / val_samples)
        val_f1.append(np.mean(val_f1_scores))

    return train_losses, train_accuracy, train_f1, val_losses, val_accuracy, val_f1

# Testing function
def test(model, test_loader, criterion):
    model.eval()
    test_loss, test_correct, test_samples = 0, 0, 0
    test_f1_scores = []
    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc='Testing', leave=False):
            outputs = model(texts)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_samples += labels.size(0)
            test_f1_scores.append(f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro'))
    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_samples
    test_f1 = np.mean(test_f1_scores)
    return test_loss, test_accuracy, test_f1


# Instantiate the model

#self, vocab_size, embedding_dim, hidden_dim, output_dim
model = SMSClassifier(len(tokenizer.word_index) + 1, 32, 32, 2) # +1 = empty space / no word

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# Call the train function and collect metrics

num_epochs = 10
train_losses, train_accuracy, train_f1, val_losses, val_accuracy, val_f1 = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Call the test function
test_loss, test_accuracy, test_f1 = test(model, test_loader, criterion)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}')



# Adjust the number of epochs to match your training

epochs = range(1, num_epochs + 1) # x-axis 


# Visualization of training and validation loss
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Visualization of training and validation accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracy, 'b-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Visualization of training and validation F1 Score
plt.subplot(2, 2, 3)
plt.plot(epochs, train_f1, 'b-', label='Training F1 Score')
plt.plot(epochs, val_f1, 'r-', label='Validation F1 Score')
plt.title('Training & Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()



plt.tight_layout()
plt.show()


