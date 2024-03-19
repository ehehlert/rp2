# python -m src.models.textract_tables.classify_textract_tables

from src.functions.get_json_multiple import get_json_multiple
from src.functions.extract_and_process_cells import extract_and_process_cells
from src.functions.extract_tables import extract_tables
from src.models.textract_tables.evaluate import evaluate_model

def classify_textract_tables(json_filepath):
    # Get the JSON files
    json_files = get_json_multiple(json_filepath)

    # Extract and process the cells
    cells_df = extract_and_process_cells(json_files)

    # Extract the tables
    tables_df = extract_tables(json_files, cells_df)

    # Classify the tables
    classified_tables_df = evaluate_model(tables_df)

    return classified_tables_df

if __name__ == "__main__":
    classified_tables_df = classify_textract_tables('textract_validate')
    print(classified_tables_df['predicted_label'].value_counts())

