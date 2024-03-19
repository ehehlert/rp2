# python -m src.models.textract_pages.model__predict
# predicts: page_type

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import BertForSequenceClassification
import json

from src.models.textract_pages.preprocess_data import preprocess_data
from src.models.textract_pages.TemperatureScaledModel import TemperatureScaledModel
from src.functions.extract_words import extract_words
from src.functions.get_json_multiple import get_json_multiple
from src.functions.shift_pages import shift_pages

with open('src/models/textract_pages/config.json') as f:
    config = json.load(f)

def predict_page_type(input_df, temperature=1.0, confidence_threshold=0.9):
    
    model_path = config['model_path']
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Wrap the original model with temperature scaling
    temp_scaled_model = TemperatureScaledModel(model, temperature)
    temp_scaled_model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    temp_scaled_model.to(device)
    
    inputs, _, _, _ = preprocess_data(input_df, train=False)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    predictions = []
    confidence_scores = []
    softmax_scores_list = []

    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask = batch

            scaled_logits = temp_scaled_model(b_input_ids, attention_mask=b_attention_mask)
            softmax_scores = F.softmax(scaled_logits, dim=1)
            max_scores, preds = torch.max(softmax_scores, dim=1)
            
            # Ensure tensor is moved to CPU before conversion
            max_scores = max_scores.cpu().numpy()
            preds = preds.cpu().numpy()

            predictions.extend(preds)
            confidence_scores.extend(max_scores)
            softmax_scores_list.extend(softmax_scores.cpu().numpy())

    softmax_scores_cpu = np.array(softmax_scores_list)

    # Decode predictions and determine ID/OOD
    label_encoder_path = config['label_encoder_path']
    label_encoder = joblib.load(label_encoder_path)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    ood_flags = [1 if score < confidence_threshold else 0 for score in confidence_scores]

    input_df['predicted_page_type'] = decoded_predictions
    input_df['OOD'] = ood_flags
    input_df['confidence_score'] = confidence_scores

    return input_df, softmax_scores_cpu

def predict(input_df):
    predicted_df, _ = predict_page_type(input_df, temperature=2.9, confidence_threshold=0.30)
    output_df = shift_pages(predicted_df)
    print(output_df.head())
    return output_df

if __name__ == '__main__':
    json_files = get_json_multiple('textract_jobs/complete/json')
    words = extract_words(json_files)
    predict(words)
    output_df = shift_pages(words)
    print(output_df.head())

