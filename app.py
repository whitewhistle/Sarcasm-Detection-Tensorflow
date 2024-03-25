import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
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

# Load the BERT model
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
for param in model_bert.base_model.parameters():
    param.requires_grad = False
n_fine_tune_layers = 10
for param in model_bert.base_model.encoder.layer[-n_fine_tune_layers:].parameters():
    param.requires_grad = True
model_bert = model_bert.to(device)
model_bert = torch.load('./bert/bert.pth', map_location=device)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

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
    inputs = tokenizer_bert(text, return_tensors='pt', max_length=256, truncation=True, padding=True)
    with torch.no_grad():
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model_bert(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return predicted_label

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

st.write("""
### About the Models
- **LSTM Model:** This model uses a Long Short-Term Memory (LSTM) neural network architecture trained on a dataset of sarcastic and non-sarcastic headlines. It preprocesses the text using a tokenizer and then makes predictions based on the learned patterns.
- **BERT Model:** This model utilizes a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model fine-tuned on a sarcasm detection task. BERT is a powerful transformer-based model that captures contextual information from the input text, allowing it to make accurate predictions.
""")
