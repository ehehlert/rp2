import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from preprocessing import preprocess_data, save_preprocessing_artifacts
from model import TableClassifier

# Load the dataset
df = pd.read_csv('src/models/textract_tables/data/tagged_set.csv')

# Separate the features and the labels
labels_df = df['table_class']
features_df = df.drop(['table_class'], axis=1)

# Preprocess data with PCA
features_scaled, training_columns, scaler, pca, imputer = preprocess_data(features_df, fit_scaler=True, fit_pca=True, n_components=0.95)
print(type(imputer))  # It is: <class 'sklearn.impute._base.SimpleImputer'>

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_df)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and create DataLoader for training and testing
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Save preprocessing artifacts, now including PCA
save_preprocessing_artifacts(scaler, pca, training_columns, imputer)

# Save the LabelEncoder
label_encoder_save_path = 'src/models/textract_tables/artifacts/label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_save_path)

# Model setup
input_size = X_train.shape[1]
output_size = len(set(labels_encoded))
hidden_layers = [128, 64]
model = TableClassifier(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')
model = model.to(device)

# Class weights
class_weights_named = {
    "grid": 9.2,
    "kvp": 0.449,
    "tabular": 1.508
}

# Loss function and optimizer
#weights_tensor = torch.tensor([class_weights_named[class_name] for class_name in label_encoder.classes_], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 150
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Save the model's state dictionary
model_save_path = 'src/models/textract_tables/artifacts/table_classifier_model.pth'
torch.save(model.state_dict(), model_save_path)