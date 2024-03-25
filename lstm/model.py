import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


# Concatenating the two Datasets
df1 = pd.read_json('./Dataset/Sarcasm_Headlines_Dataset.json', lines=True)
df2 = pd.read_json('./Dataset/Sarcasm_Headlines_Dataset_v2.json', lines=True)
frames = [df1, df2]
df = pd.concat(frames)

# Tokenize to vectorize and convert texts into features
max_features = 2500
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['headline'].values)
X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X)
Y = df['is_sarcastic'].values


# Splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Convert data into PyTorch tensors and move to GPU if available
X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)

# Define the model architecture
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_out, recurrent_dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.lstm = nn.LSTM(embed_dim, lstm_out, batch_first=True)
        self.fc = nn.Linear(lstm_out, 2)
        self.recurrent_dropout = nn.Dropout(recurrent_dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_dropout = self.dropout(embedded)
        lstm_input = self.recurrent_dropout(embedded_dropout)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Define hyperparameters
embed_dim = 128
lstm_out = 196
batch_size = 32
epochs = 25

# Instantiate the model and move to GPU if available
model = LSTMModel(max_features, embed_dim, lstm_out,0.1).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Lists to store accuracy and loss
train_loss_history = []
test_loss_history = []
accuracy_history = []

# Training the model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_loss_history.append(train_loss)

    # Evaluation on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    test_loss_history.append(test_loss)

    accuracy = 100 * correct / total
    accuracy_history.append(accuracy)

    print('Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f, Test Accuracy: %.2f %%' % (epoch + 1, epochs, train_loss,
                                                                                        test_loss, accuracy))
    
# Save the model
torch.save(model.state_dict(), './lstm/lstm_model.pth')