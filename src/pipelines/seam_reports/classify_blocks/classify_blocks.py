# python -m src.pipelines.seam_reports.classify_blocks.classify_blocks

from src.functions.extract_blocks import extract_blocks
from src.functions.get_json_multiple import get_json_multiple
from src.pipelines.seam_reports.classify_blocks.evaluate import evaluate_model

json_dir = 'textract'
model_dir = 'src/pipelines/seam_reports/classify_blocks'

def classify_blocks(json_dir):
    json_files = get_json_multiple(json_dir)
    blocks_df = extract_blocks(json_files, 10)
    return evaluate_model(blocks_df)

if __name__ == "__main__":
    classified_blocks = classify_blocks(json_dir)
    print(classified_blocks.head(20))
