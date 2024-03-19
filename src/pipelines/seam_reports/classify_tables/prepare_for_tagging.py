# python -m src.pipelines.seam_reports.classify_tables.prepare_for_tagging

from src.functions.get_json_multiple import get_json_multiple
from src.functions.extract_and_process_cells import extract_and_process_cells
from src.functions.extract_tables import extract_tables

output_path = 'src/pipelines/seam_reports/classify_tables/data'
output_file = 'training_set_untagged.csv'

def prepare_for_tagging(output_path, output_file):
    json_files = get_json_multiple('textract_train')
    cells_df = extract_and_process_cells(json_files)
    tables_df = extract_tables(json_files, cells_df)
    tables_df.to_csv(f"{output_path}/{output_file}", index=False)

prepare_for_tagging(output_path, output_file)