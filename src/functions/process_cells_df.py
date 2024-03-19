import pandas as pd

def process_cells_dataframe(cells_df):
    """
    Process the cells DataFrame to handle merged cells and ensure correct data types for columns.

    Parameters:
    cells_df (pd.DataFrame): DataFrame containing information about spreadsheet cells.

    Returns:
    pd.DataFrame: Processed DataFrame with merged cells handled and necessary columns in correct data types.
    """

    # Ensure the necessary columns are in the correct data type
    cells_df['row_index'] = cells_df['row_index'].fillna(0).astype(int)
    cells_df['column_index'] = cells_df['column_index'].fillna(0).astype(int)
    cells_df['row_span'] = cells_df['row_span'].fillna(1).astype(int)  # Default span of 1 if missing
    cells_df['column_span'] = cells_df['column_span'].fillna(1).astype(int)

    # Sort the DataFrame as required
    cells_df.sort_values(by=['table_id', 'column_index', 'row_index'], inplace=True)

    # Isolate merged cells
    merged_cells = cells_df[cells_df['cell_type'] == 'MERGED_CELL']

    # Initialize a column for tracking merged cell parent ID and merge status
    # Explicitly initialize as object type to avoid dtype incompatibility
    cells_df['merged_parent_cell_id'] = pd.NA
    cells_df['has_merged_parent'] = 0

    for cell in merged_cells.itertuples():
        # Calculate the affected range of rows and columns
        affected_rows = range(cell.row_index, cell.row_index + cell.row_span)
        affected_columns = range(cell.column_index, cell.column_index + cell.column_span)

        # Find the cells that are affected
        affected_cells = cells_df[
            (cells_df['table_id'] == cell.table_id) &
            (cells_df['cell_type'] == 'CHILD') &  # Targeting only child cells
            (cells_df['row_index'].isin(affected_rows)) &
            (cells_df['column_index'].isin(affected_columns))
        ]

        # Aggregate text content of affected cells, stripping to remove leading/trailing spaces
        aggregated_text_content = " ".join(filter(None, affected_cells['cell_content'].astype(str))).strip()

        if aggregated_text_content:
            # Update the affected cells with the aggregated text content and merge-related information
            cells_df.loc[affected_cells.index, 'cell_content'] = aggregated_text_content
            cells_df.loc[affected_cells.index, 'merged_parent_cell_id'] = cell.cell_id
            cells_df.loc[affected_cells.index, 'has_merged_parent'] = 1

    # Fill missing values for new columns
    cells_df['has_merged_parent'] = cells_df['has_merged_parent'].fillna(0).astype(int)
    # Do not convert merged_parent_cell_id to int; leave it as is or ensure it's treated as a string/object
    cells_df['merged_parent_cell_id'] = cells_df['merged_parent_cell_id'].fillna('None')

    # Ensure 'merged_parent_cell_id' is explicitly recognized as a string/object column
    cells_df['merged_parent_cell_id'] = cells_df['merged_parent_cell_id'].astype(str)

    print(f"Processed Cells DataFrame has {cells_df.shape[0]} rows and {cells_df.shape[1]} columns")

    return cells_df