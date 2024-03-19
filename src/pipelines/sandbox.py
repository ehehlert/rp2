# run with python -m src.pipeline.preprocess_json

### THE POINT OF THIS SCRIPT IS TO SIMPLIFY THE PROCESS OF EXPLORING THE JSON FILES GENERATED FROM TEXTRACT
## You can process specific files (get_json) or all files in a directory (get_json_multiple) - output for both is json_files
## You can extract the first 10 blocks from each file (extract_blocks: input is json_files)
## You can extract the cells from each file (extract_cells: input is json_files)
## You can extract the tables from each file (extract_tables: input is json_files)

## in order to specify a json file, import get_json from src.components.json instead, and specify a filepath inside of an array (you can process multiple)
from src.functions.get_json_multiple import get_json_multiple
from src.functions.extract_blocks import extract_blocks
from src.functions.extract_and_process_cells import extract_and_process_cells
from src.functions.extract_tables import extract_tables

## if you want to process all files in a directory, use the following:
directory = 'textract'
json_files = get_json_multiple(directory)

## if you want to specify file(s), use the following:
#directory = 'textract'
#json_files = ['textract_results_f40aa4b27ddfbf97b8a152a34d27c94b24b943916637c79375b4310b168460a0.json']

## grab the first n blocks from each file, or put 'all'
blocks_df = extract_blocks(json_files, 10)

## grab the cells from each file and process the dataframe
#cells_df = extract_and_process_cells(json_files)

## grab the tables from each file
#tables_df = extract_tables(json_files, cells_df)


## save the dataframes to csv
blocks_df.to_csv('./output/csvs/blocks.csv', index=False)
#cells_df.to_csv('./output/csvs/cells.csv', index=False)
#tables_df.to_csv('./output/csvs/tables.csv', index=False)


## NOTE: There is also an option to extract and process cells separately
#from src.components.extract_cell_blocks import extract_cell_blocks
#from src.components.process_cells_df import process_cells_dataframe
#cells_df = extract_cell_blocks(json_files)
#cells_df = process_cells_dataframe(cells_df)