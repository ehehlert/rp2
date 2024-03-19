# python -m src.pipelines.eee.extract_kvps

import pandas as pd
import pandas as pd
from difflib import SequenceMatcher
from src.functions.get_json_multiple import get_json_multiple
# from src.functions.extract_and_process_cells import extract_and_process_cells
from src.functions.process_cells_df import process_cells_dataframe
from src.functions.extract_cells_new import extract_cell_blocks
from src.functions.extract_tables import extract_tables
from src.functions.extract_words import extract_words
from src.pipelines.eee.classify_tables.model__predict import predict
from src.pipelines.eee.pred_continuation import predict_shift_and_group

def filter_and_export_tables(tables_df, cells_df, predicted_table_role):
    """
    Merges tables_df and cells_df on table_id, filters rows where predicted_table_role is "METADATA_1",
    and returns the resulting dataframe.
    
    Parameters:
    - tables_df: DataFrame with columns 'table_id' and 'predicted_table_role'
    - cells_df: DataFrame with columns 'cell_id', 'cell_content', 'table_id'
    """
    
    # Merge the dataframes on 'table_id'
    merged_df = pd.merge(cells_df, tables_df, on='table_id', how='left')
    
    # Filter the dataframe to include only rows where predicted_table_role is "METADATA_1"
    filtered_df = merged_df[merged_df['predicted_table_role'] == predicted_table_role]
    
    return filtered_df

def find_key_value_pairs(df, keywords):
    """
    Searches the dataframe for rows whose cell_content contains any of the keywords (case-insensitive),
    and returns the cell_content of the cell immediately to the right of the found keyword cell,
    along with the related page_id and table_id.

    Parameters:
    - df: DataFrame with columns 'table_id', 'row_index', 'column_index', 'page_id', and 'cell_content'
    - keywords: List of keywords to search for

    Returns:
    A dictionary where each keyword is a key, and the value is a list of dictionaries. Each dictionary in the list
    contains the 'cell_content', 'page_id', and 'table_id' of the corresponding value found.
    """
    # Normalize the cell_content for case-insensitive search
    df['cell_content_lower'] = df['cell_content'].str.lower()
    
    # Initialize a dictionary to hold the results
    results = {keyword: [] for keyword in keywords}
    
    # Iterate through the keywords to search the dataframe
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Find matching cells
        matches = df[df['cell_content_lower'].str.contains(keyword_lower)]
        
        # Iterate through matches to find the cell to the right
        for _, match in matches.iterrows():
            # Identify the cell to the right
            right_cell = df[(df['table_id'] == match['table_id']) &
                            (df['row_index'] == match['row_index']) &
                            (df['column_index'] == match['column_index'] + 1)]
            
            # If there is a right cell, add its content and metadata to the results
            if not right_cell.empty:
                right_cell_info = right_cell.iloc[0]
                results[keyword].append({
                    'cell_content': right_cell_info['cell_content'],
                    'page_id': right_cell_info['page_id'],
                    'table_id': right_cell_info['table_id']
                })
    
    # Remove the temporary lowercase column
    df.drop(columns=['cell_content_lower'], inplace=True)
    return results

def convert_results_to_dataframe(results):
    """
    Converts the results dictionary from find_key_value_pairs into a DataFrame for exporting.

    Parameters:
    - results: The dictionary returned by find_key_value_pairs
    
    Returns:
    A pandas DataFrame suitable for saving to CSV.
    """
    records = []
    for keyword, items in results.items():
        for item in items:
            records.append({
                'keyword': keyword,
                **item
            })
    return pd.DataFrame(records)

def similarity(a, b):
    """Return the ratio of similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def filter_similar_key_value_pairs(df, threshold=0.8):
    """
    Filters out rows where the key and value are similar beyond a specified threshold from a DataFrame.

    Parameters:
    - df: The DataFrame containing key-value pairs.
    - threshold: Similarity threshold beyond which pairs are considered too similar and thus filtered out.

    Returns:
    A DataFrame with the filtered key-value pairs.
    """
    # Filter out rows based on the similarity threshold
    filtered_df = df[df.apply(lambda row: similarity(row['keyword'], row['cell_content']) < threshold, axis=1)]

    return filtered_df

def extract_kvps(json_files, predicted_table_role, search_terms):
    """
    Extracts key-value pairs from json tables

    Parameters:
    - predicted_table_type: The type of table to extract key-value pairs from.
    - search_terms: The search terms to use to extract the key-value pairs.
    - json_files: The list of JSON files to extract the key-value pairs from.

    Returns:
    A DataFrame containing the extracted key-value pairs.
    """

    # Extract and process tables and cells
    files = get_json_multiple(json_files)
    cells = extract_cell_blocks(files)
    cells_df = process_cells_dataframe(cells)
    tables_df = extract_tables(files, cells_df)

    # Apply ML model to classify tables
    tables_with_role = predict(tables_df)

    # Filter and export tables
    filtered_df = filter_and_export_tables(tables_with_role, cells_df, predicted_table_role)

    # Find key-value pairs
    key_value_pairs = find_key_value_pairs(filtered_df, search_terms)

    # Convert the results to a DataFrame and then save to CSV
    results_df = convert_results_to_dataframe(key_value_pairs)

    # Filter similar key-value pairs
    output = filter_similar_key_value_pairs(results_df, threshold=0.8)

    return output
    
# Example usage
if __name__ == "__main__":
    json_filepath = 'textract_jobs/staging'
    json_files = get_json_multiple(json_filepath)
    
    # for some reason this isn't working here --- fix it tomorrow
    pages = extract_words(json_files)
    groupings = predict_shift_and_group(pages)
    print(groupings)

    predicted_table_role = 'NAMEPLATE_INFORMATION'
    search_terms = ['Manufacturer']
    output = extract_kvps(json_filepath, predicted_table_role, search_terms)
    output.to_csv('key_value_pairs_final.csv', index=False)
    print(output)

#### As of now, the page_ids are coming in duplicate, even though table_ids seem to be correct
#### The issue is with the extract_cells function, and it may be worth simply creating a page/table map to avoid this issue
#### I will try to create a page/table map to avoid this issue tomorrow
#### Check key_value_pairs_final.csv for the most up-to-date output