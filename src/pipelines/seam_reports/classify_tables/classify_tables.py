# python -m src.pipelines.seam_reports.classify_tables.classify_tables

from src.functions.extract_and_process_cells import extract_and_process_cells
from src.functions.extract_tables import extract_tables
from src.functions.get_json_multiple import get_json_multiple
from src.pipelines.seam_reports.classify_tables.evaluate import evaluate_model

json_dir = 'textract'
model_dir = 'src/pipelines/seam_reports/classify_tables'

def classify_tables(json_dir):
    json_files = get_json_multiple(json_dir)
    cells_df = extract_and_process_cells(json_files)
    tables_df = extract_tables(json_files, cells_df)
    return evaluate_model(tables_df)

if __name__ == "__main__":
    classified_blocks = classify_tables(json_dir)
    print(classified_blocks.head(20))
