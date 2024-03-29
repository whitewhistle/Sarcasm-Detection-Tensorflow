import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.preprocessing.text import Tokenizer
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle

# Set page configuration with custom theme
st.set_page_config(
    page_title="Sarcasm Detection App",
    page_icon="😏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Customize the theme
custom_theme = """
    [theme]
    primaryColor = "#E694FF"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#DCDCDC"
    font = "sans serif"
"""
# Apply the custom theme
st.markdown(f'<style>{custom_theme}</style>', unsafe_allow_html=True)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and LSTM model
tokenizer_path = "./lstm/tokenizer.pickle"
with open(tokenizer_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

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

model_lstm = LSTMModel(2500, 128, 196, 0.1).to(device)
model_lstm.load_state_dict(torch.load("./lstm/lstm_model.pth", map_location=device)['model_state_dict'])
model_lstm.eval()

import requests
API_URL = "https://api-inference.huggingface.co/models/stein1045/bert-model2"
headers = {"Authorization": "Bearer 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"}
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	


# Function to make predictions using LSTM
def predict_sarcasm_lstm(texts, model, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=max_length)
    sequences_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(sequences_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    return predicted[0]

# Function to make predictions using BERT
def predict_text_bert(text):
    output = query({
	"inputs": text,
     })
    print(output)
    return output

# Streamlit app
st.title("Sarcasm Detection App")
st.subheader("Choose a Model to Detect Sarcasm")

model_choice = st.radio("Select Model:", ("LSTM", "BERT"))

input_text = st.text_input("Enter your text here:", placeholder="Type your text here...")

predict_button = st.button("Predict")

if predict_button:
    if input_text:
        if model_choice == "LSTM":
            prediction = predict_sarcasm_lstm([input_text], model_lstm, loaded_tokenizer, max_length=50)
        else:
            prediction = predict_text_bert(input_text)
        
        if prediction == 1:
            st.write("Prediction: 😏 Sarcastic!")
        else:
            st.write("Prediction: 😊 Not Sarcastic")

# Frontend improvements
st.write("""
### About the Models
- **LSTM Model:** This model utilizes a Long Short-Term Memory (LSTM) neural network architecture trained on a dataset of sarcastic and non-sarcastic headlines. It preprocesses the text using a tokenizer and then makes predictions based on the learned patterns.
- **BERT Model:** This model employs a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model fine-tuned on a sarcasm detection task. BERT is a powerful transformer-based model that captures contextual information from the input text, allowing it to make accurate predictions.
- **Note:** While these models can detect sarcasm with high accuracy, they may not always provide the correct answer due to the inherent complexity and ambiguity of sarcasm.
- **About the Dataset:** This model was trained on a dataset consisting of headlines. It has learned patterns from various headlines to distinguish between sarcastic and non-sarcastic statements. While it performs well on headlines, its accuracy may vary depending on the context and complexity of the text.
""")
