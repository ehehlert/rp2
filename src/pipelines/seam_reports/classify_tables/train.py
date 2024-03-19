# python -m src.pipelines.seam_reports.classify_tables.train

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

directory = 'src/pipelines/seam_reports/classify_tables'
input_file = 'training_set.csv'

input_path = f"{directory}/data/{input_file}"
model_output_path = f"{directory}/model"
class_names_output_path = f"{directory}/model/saved_classes.npy"

# Load the dataset
tagged_df = pd.read_csv(input_path)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
inputs = tokenizer(tagged_df['content'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)

# Convert labels to numerical format

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tagged_df['table_class'])

# Create a TensorDataset
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is the number of batches * number of epochs.
total_steps = len(train_loader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# Function for calculating accuracy - useful for evaluation
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Move the model to the defined device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

model = model.to(device)

# Training loop
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    
    # Training
    model.train()
    
    total_loss = 0
    
    for step, batch in enumerate(train_loader):
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        model.zero_grad()        
        
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    
    print("Average training loss: {0:.2f}".format(avg_train_loss))


# Saving the fine-tuned model & tokenizer
model.save_pretrained(model_output_path)
tokenizer.save_pretrained(model_output_path)

# Saving class names for later use
np.save(class_names_output_path, label_encoder.classes_)

print(f"Training complete. Exported to {model_output_path}")