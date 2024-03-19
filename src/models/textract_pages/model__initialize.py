# python -m src.models.textract_pages.model__initialize
# predicts: page_type

from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from src.models.textract_pages.preprocess_data import preprocess_data
import json

with open('src/models/textract_pages/config.json') as f:
    config = json.load(f)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def initialize_model(tagged_csv):
    # Load the tagged data
    df = pd.read_csv(tagged_csv)

    # Preprocess data
    inputs, labels, class_names, tokenizer = preprocess_data(df, train=True)

    # Create a TensorDataset
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Load the untrained BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(class_names))

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f'Using device: {device}')

    # Number of training epochs
    epochs = 4

    # Total number of training steps is the number of batches * number of epochs
    total_steps = len(train_loader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

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

        # Validation
        model.eval()
        eval_accuracy = 0
        nb_eval_steps = 0

        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

    # Saving the fine-tuned model & tokenizer
    model.save_pretrained(config['model_path'])
    tokenizer.save_pretrained(config['tokenizer_path'])

    print(f"Training complete. Exported to {config['model_path']}")

if __name__ == '__main__':
    tagged_csv = 'src/models/textract_pages/data/training_set_tagged.csv'
    initialize_model(tagged_csv)

