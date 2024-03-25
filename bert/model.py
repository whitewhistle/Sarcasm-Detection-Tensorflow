import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

# Parameters
bert_path = 'bert-base-uncased'
max_seq_length = 256


# Concatenating the two Datasets

df1 = pd.read_json('./Dataset/Sarcasm_Headlines_Dataset.json', lines=True)
df2 = pd.read_json('./Dataset/Sarcasm_Headlines_Dataset_v2.json', lines=True)
frames = [df1, df2]
df = pd.concat(frames)


text = df['headline'].tolist()
text = [' '.join(t.split()[0:max_seq_length]) for t in text]
text = np.array(text, dtype=object)[:, np.newaxis]
text_label = df['is_sarcastic'].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(text, text_label, random_state=0)

#Tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_path)

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# Convert to PyTorch Dataset
train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, max_seq_length)
test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer, max_seq_length)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
for param in model.base_model.parameters():
    param.requires_grad = False
n_fine_tune_layers = 10
for param in model.base_model.encoder.layer[-n_fine_tune_layers:].parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()
accumulation_steps = 4

# Initialize lists to store training and testing metrics
train_losses = []
test_losses = []
accuracies = []
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/(i+1):.4f}')

    # Evaluation
    model.eval()
    total_loss_eval = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss_eval += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    avg_loss_eval = total_loss_eval / len(test_loader)
    accuracy = correct / len(test_dataset)

    # Store metrics for plotting
    train_losses.append(avg_loss)
    test_losses.append(avg_loss_eval)
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}, Train Loss: {avg_loss}, Test Loss: {avg_loss_eval}, Accuracy: {accuracy}")

    # Manually clear GPU cache
    torch.cuda.empty_cache()

# Save the entire model
torch.save(model, './bert/bert.pth')