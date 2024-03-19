# python -m src.models.textract_pages.evaluate

import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

from src.models.textract_pages.TemperatureScaledModel import TemperatureScaledModel
from src.models.textract_pages.preprocess_data import preprocess_data
from src.functions.find_optimal_temperature import find_optimal_temperature

with open('src/models/textract_pages/config.json') as f:
    config = json.load(f)

def evaluate_model_with_temp_scaling_and_ood_detection(input_df, model, temperature=1.0, confidence_threshold=0.9):
    # Wrap the original BERT with temperature scaling
    temp_scaled_model = TemperatureScaledModel(model, temperature)
    temp_scaled_model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    temp_scaled_model.to(device)
    
    inputs, _, _ = preprocess_data(input_df, train=False)
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
    label_encoder_path = 'src/models/textract_pages/artifacts/label_encoder.pkl'
    label_encoder = joblib.load(label_encoder_path)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    ood_flags = [1 if score < confidence_threshold else 0 for score in confidence_scores]

    input_df['predicted_page_type'] = decoded_predictions
    input_df['OOD'] = ood_flags
    input_df['confidence_score'] = confidence_scores

    # Assuming 'true_labels' are in 'input_df'
    y_true = input_df['true_labels'].values  # Make sure this matches your actual data

    # Calculate and print metrics
    accuracy = accuracy_score(y_true, decoded_predictions)
    precision = precision_score(y_true, decoded_predictions, average='weighted', zero_division=0)
    recall = recall_score(y_true, decoded_predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_true, decoded_predictions, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return input_df, softmax_scores_cpu

# EVALUATION SCRIPT:
if __name__ == "__main__":
    # Load the dataset (could be validation or test set, depending on your process)
    input_df = pd.read_csv('src/models/textract_pages/data/validation_set.csv')

    # Load your pretrained model
    model_path = "src/models/textract_pages/model"
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Set device for model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Define a range of temperatures to explore
    temperatures = [1.0 + 0.1 * i for i in range(20)]  # From 1.0 to 3.0 in steps of 0.1

    # Find the optimal temperature on your validation set
    # Note: You need to adapt evaluate_model_with_temp_scaling_and_ood_detection 
    # to return metrics necessary for calculate_ece, like y_true and y_prob
    best_temperature, _ = find_optimal_temperature(input_df, model, temperatures)

    print(f"Optimal temperature found: {best_temperature}")

    # Evaluate the model with the optimal temperature and OOD detection on the test set
    evaluated_df, _ = evaluate_model_with_temp_scaling_and_ood_detection(input_df, model, temperature=2.9, confidence_threshold=0.90)

    # Save the evaluated DataFrame with predictions and OOD flags
    evaluated_df.to_csv('src/models/textract_pages/data/predictions_with_temp_and_ood.csv', index=False)
    print(evaluated_df.head())



    input_df = pd.read_csv('src/models/textract_pages/data/test_set.csv')
    evaluated = evaluate_model_with_ood_detection(input_df, confidence_threshold=0.9)
    evaluated.to_csv('src/models/textract_pages/data/predictions2.csv', index=False)
    print(evaluated.head())