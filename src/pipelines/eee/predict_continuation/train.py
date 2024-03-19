# python -m src.pipelines.eee.predict_continuation.train

import pandas as pd
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch

data = pd.read_csv('src/pipelines/eee/predict_continuation/training_data/training_set.csv')

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Preparing the dataset
texts = data['page_content'].tolist()
labels = data['continuation'].tolist()

# Create the dataset
dataset = TextDataset(texts, labels, tokenizer)

# Splitting the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

from transformers import DistilBertForSequenceClassification, AdamW

# Load pre-trained DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Move model to GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(f'Using device: {device}')

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3  # Number of training epochs. Adjust as needed.

from sklearn.metrics import accuracy_score

def evaluate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            loss = outputs.loss
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(true_labels, predictions)
    return avg_val_loss, val_accuracy

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss {avg_train_loss}")

    # Perform validation at the end of each epoch
    avg_val_loss, val_accuracy = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}: Validation Loss {avg_val_loss}, Validation Accuracy {val_accuracy}")

# Specify the directory to save the model and tokenizer
model_save_path = "src/pipelines/eee/predict_continuation/model"
tokenizer_save_path = "src/pipelines/eee/predict_continuation/model/tokenizer"

# Create the directory if it doesn't exist
import os
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Save the model
model.save_pretrained(model_save_path)

# Save the tokenizer
tokenizer.save_pretrained(tokenizer_save_path)
