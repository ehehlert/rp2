# python -m src.pipelines.seam_reports.classify_tables.evaluate

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

def evaluate_model(input_df):
    model_path = "src/pipelines/seam_reports/classify_tables/model"
    class_names = np.load("src/pipelines/seam_reports/classify_tables/model/saved_classes.npy", allow_pickle=True)

    # Load the pre-trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    input_col = input_df['content'].astype(str)

    # Tokenize the text
    inputs = tokenizer(input_col.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Create a TensorDataset
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Set the model to evaluation mode
    model.eval()

    # Determine the device to use
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Predicting table_class... Using device: {device}')
    model.to(device)

    # Initialize a list to store predictions
    predictions = []

    # Predict
    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask = batch

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
            logits = outputs.logits

            # Move logits to CPU and convert to numpy array
            logits = logits.detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            predictions.extend(pred_flat)

    # Decode the predictions using the class names
    decoded_predictions = [class_names[pred] for pred in predictions]

    # Add predictions to the DataFrame
    input_df['predicted_table_class'] = decoded_predictions

    # Save the DataFrame with predictions
    return input_df

#Example usage:
if __name__ == "__main__":
    input_df = pd.read_csv('src/pipelines/seam_reports/classify_tables/data/eval_set.csv')
    evaluated = evaluate_model(input_df)
    evaluated.to_csv('src/pipelines/seam_reports/classify_tables/data/predictions.csv', index=False)
    print(evaluated.head())