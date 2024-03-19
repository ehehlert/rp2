from src.functions.extract_cells import extract_cell_blocks
from src.functions.process_cells_df import process_cells_dataframe

def extract_and_process_cells(json_files):
    """
    Extract and process the cells from the JSON files.

    Parameters:
    json_files (list): List of file paths to JSON files.

    Returns:
    pd.DataFrame: Processed DataFrame containing information about spreadsheet cells.
    """
    # Extract the cells from the JSON files
    cells_df = extract_cell_blocks(json_files)

    # Process the cells DataFrame
    cells_df = process_cells_dataframe(cells_df)

    return cells_df

