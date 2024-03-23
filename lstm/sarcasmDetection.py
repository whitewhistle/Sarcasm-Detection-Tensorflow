import streamlit as st
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer_path = "./lstm/tokenizer.pickle"
with open(tokenizer_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Load the model
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

model = LSTMModel(2500, 128, 196, 0.1).to(device)
model.load_state_dict(torch.load("./lstm/lstm_model.pth", map_location=device)['model_state_dict'])
model.eval()

# Function to make predictions
def predict_sarcasm(texts, model, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=max_length)
    sequences_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(sequences_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    return predicted

# Streamlit app
st.set_page_config(page_title="Sarcasm Detection App", page_icon="😏")  # Setting the page icon

st.title("Welcome to the Sarcasm Detection App!")
st.subheader("Detect sarcasm in your text with ease 😄")

st.write("Sarcasm can sometimes be tricky to detect in text. This app uses machine learning to predict whether a given text is sarcastic or not. Simply enter your text in the box below and let the magic happen!")

input_text = st.text_input("Enter your text here:", placeholder="Type your text here...")

predict_button = st.button("Predict")

if predict_button:
    if input_text:
        prediction = predict_sarcasm([input_text], model, loaded_tokenizer, max_length=50)
        if prediction[0] == 1:
            st.write("Prediction: 😏 Sarcastic!")
        else:
            st.write("Prediction: 😊 Not Sarcastic")


st.write("""
### How does it work?
This app uses a Long Short-Term Memory (LSTM) neural network model trained on a dataset of sarcastic and non-sarcastic headlines.
The model predicts whether the input text is sarcastic or not.
""")

st.write("""
### About the Model
The LSTM model used in this app consists of an embedding layer, an LSTM layer, and a fully connected layer.
It takes the input text, converts it into embeddings, processes it through the LSTM layer, and finally classifies it as sarcastic or not.
""")

st.write("""
### About the Data
The dataset used to train the model consists of sarcastic and non-sarcastic headlines.
It was preprocessed and tokenized before being used for training the model.
""")

