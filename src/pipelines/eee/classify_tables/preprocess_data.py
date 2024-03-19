import pandas as pd
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import json

with open('src/pipelines/eee/classify_tables/config.json') as f:
    config = json.load(f)

def preprocess_data(df, train=False, retrain=False):

    # Ensure all page_content is a string
    df['content'] = df['content'].fillna('').astype(str)
    
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the text
    inputs = tokenizer(df['content'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Convert labels to numerical format
    if train:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['table_role'])
        class_names = label_encoder.classes_
        np.save(config['class_names_path'], label_encoder.classes_)
        joblib.dump(label_encoder, config['label_encoder_path'])
        return inputs, labels, class_names, tokenizer
    elif retrain:
        label_encoder = joblib.load(config['label_encoder_path'])
        labels = label_encoder.transform(df['table_role'])
        return inputs, labels, None, tokenizer
    else:
        return inputs, None, None, tokenizer
    

