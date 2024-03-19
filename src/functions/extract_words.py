import json
import pandas as pd
import numpy as np

def extract_words(json_files, n_blocks='all'):
    """
    Extracts and concatenates word data from JSON files for each page, compiling them into a pandas DataFrame.

    Parameters:
    - json_files: A list of paths to JSON files containing block data.
    - n_blocks: The number of blocks to process from each file. Can be an integer or 'all' to process all blocks.

    Returns:
    - A pandas DataFrame containing the source file name, Page Id, and the concatenated text of all "WORD" blocks within each unique page.
    """

    concatenated_texts = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            source_file_name = json_file.split('/')[-1]  # Extracting only the filename

            # Find Page Ids and numbers
            page_info = {block['Page']: {'page_id': block['Id'], 'page_number': block['Page']} for block in data if block['BlockType'] == 'PAGE'}

            # Initialize a dictionary to hold concatenated text for each page
            text_by_page = {page: '' for page in page_info.keys()}

            # Process and concatenate text of WORD blocks
            for block in data:
                if block['BlockType'] == 'WORD':
                    page = block['Page']
                    text_by_page[page] += block.get('Text', '') + ' '

            # Create records for the DataFrame
            for page, text in text_by_page.items():
                concatenated_texts.append({
                    'source': source_file_name,
                    'page_id': page_info[page]['page_id'],
                    'page_number': page_info[page]['page_number'],
                    'page_content': text.strip()  # Remove the trailing space
                })

    # Convert the list of dictionaries into a DataFrame
    concatenated_texts_df = pd.DataFrame(concatenated_texts)

    print(f'Extracted {len(concatenated_texts_df)} pages from {len(json_files)} files.')

    return concatenated_texts_df
