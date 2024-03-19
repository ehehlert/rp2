# python -m src.pipelines.seam_reports.classify_blocks.prepare_for_tagging

from src.functions.get_json_multiple import get_json_multiple
from src.functions.extract_blocks import extract_blocks

output_path = 'src/pipelines/seam-reports/classify-blocks/data'
output_file = 'training_set_untagged.csv'

def prepare_for_tagging(output_path, output_file):
    json_files = get_json_multiple('textract')
    blocks_df = extract_blocks(json_files, 10)
    blocks_df.to_csv(f"{output_path}/{output_file}", index=False)

prepare_for_tagging(output_path, output_file)