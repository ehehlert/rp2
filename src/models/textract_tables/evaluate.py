import pandas as pd
import torch
import joblib
from src.models.textract_tables.model import TableClassifier
from src.models.textract_tables.preprocessing import load_preprocessing_artifacts
from src.models.textract_tables.preprocessing import preprocess_data
import json

def evaluate_model(dataframe):
    # Load configuration for model and artifact paths
    with open('src/models/textract_tables/config.json', 'r') as f:
        config = json.load(f)
    
    # Load the preprocessing artifacts
    scaler, pca, imputer, training_columns = load_preprocessing_artifacts()

    # Preprocess the dataset using the preprocess_data function
    dataframe_processed, _, _, _, _ = preprocess_data(dataframe, training_columns, scaler, pca, imputer)
    
    # Load the label encoder  
    label_encoder = joblib.load(config['label_encoder_path'])

    # Ensure the device is set correctly
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(pca.n_components_)

    # Initialize and load the model
    model = TableClassifier(input_size=pca.n_components_, output_size=len(label_encoder.classes_), hidden_layers=[128, 64])
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.to(device)
    model.eval()

    # Make predictions
    input_tensor = torch.tensor(dataframe_processed, dtype=torch.float).to(device)

    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_indices = torch.argmax(predictions, dim=1).cpu().numpy()

    # Decode predictions
    decoded_labels = label_encoder.inverse_transform(predicted_indices)

    # Attach predictions to the DataFrame and return
    dataframe['predicted_label'] = decoded_labels
    return dataframe