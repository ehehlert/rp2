# python -m src.pipelines.eee.pred_continuation

import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.functions.shift_pages import shift_pages_continuation, group_pages_by_continuation

# Paths where the model and tokenizer are saved
model_path = "src/pipelines/eee/predict_continuation/model"
tokenizer_path = "src/pipelines/eee/predict_continuation/model/tokenizer"

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

# Assuming MPS is available, move the model to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(f'Using device: {device}')
model.eval()

def predict_continuation(df, tokenizer, model, device, batch_size=8):
    # Assuming 'page_content' is the column with text to classify
    texts = df['page_content'].tolist()

    predictions = []
    probs_list = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize texts
        encodings = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Move encodings to the same device as model
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        # Calculate probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        batch_predictions = torch.argmax(probs, dim=1)
        
        predictions.extend(batch_predictions.cpu().numpy())
        probs_list.extend(probs[:, 1].cpu().numpy())  # Probability of being a continuation
    
    # Add predictions to the DataFrame
    df['predicted_continuation'] = predictions
    # df['continuation_probability'] = probs_list
    
    return df


def predict_shift_and_group(df):
    df_with_predictions = predict_continuation(df, tokenizer, model, device)
    df_shifted = shift_pages_continuation(df_with_predictions)
    df_grouped = group_pages_by_continuation(df_shifted)
    return df_grouped

# Example usage
df_new = pd.read_csv("src/pipelines/eee/predict_continuation/training_data/test_set.csv")
df_grouped = predict_shift_and_group(df_new)
df_grouped.to_csv('output20.csv', index=False)
