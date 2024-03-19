import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def preprocess_data(df, training_columns=None, scaler=None, pca=None, imputer=None, fit_scaler=False, fit_pca=False, n_components=None):
    # Drop unnecessary columns
    df_processed = df.drop(['table_id', 'table_page', 'source', 'content', 'entities'], axis=1, errors='ignore')

    # Apply one-hot encoding
    df_processed = pd.get_dummies(df_processed)

    # If a list of training columns is provided, this function makes sure the test data has the same columns
    if training_columns:
        missing_cols = set(training_columns) - set(df_processed.columns)
        for col in missing_cols:
            df_processed[col] = 0
        df_processed = df_processed[training_columns]
    else:
        # This block is primarily for the training phase
        training_columns = df_processed.columns.tolist()

    # Impute missing values
    if fit_scaler:
        imputer = SimpleImputer(strategy='mean')
        df_processed = imputer.fit_transform(df_processed)
    else:
        # Apply imputation to test data based on training fit
        df_processed = imputer.transform(df_processed)

    # Normalize the features
    if fit_scaler:
        scaler = StandardScaler()
        df_processed = scaler.fit_transform(df_processed)
    else:
        # Apply scaling to test data based on training fit
        df_processed = scaler.transform(df_processed)

    # Apply PCA
    if fit_pca:
        pca = PCA(n_components=n_components)
        df_processed = pca.fit_transform(df_processed)
    elif pca is not None:
        df_processed = pca.transform(df_processed)

    return df_processed, training_columns, scaler, pca, imputer

def save_preprocessing_artifacts(scaler, pca, training_columns, imputer, scaler_path='src/models/textract_tables/artifacts/scaler.pkl', pca_path='src/models/textract_tables/artifacts/pca.pkl', imputer_path='src/models/textract_tables/artifacts/imputer.pkl', columns_path='src/models/textract_tables/artifacts/training_columns.pkl'):
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(training_columns, columns_path)

def load_preprocessing_artifacts():
    scaler = joblib.load('src/models/textract_tables/artifacts/scaler.pkl')
    pca = joblib.load('src/models/textract_tables/artifacts/pca.pkl')
    imputer = joblib.load('src/models/textract_tables/artifacts/imputer.pkl')
    training_columns = joblib.load('src/models/textract_tables/artifacts/training_columns.pkl')
    return scaler, pca, imputer, training_columns


