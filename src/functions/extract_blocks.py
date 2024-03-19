import json
import pandas as pd
import numpy as np

def extract_blocks(json_files, n_blocks=10):
    """
    Extracts block data from JSON files and compiles them into a pandas DataFrame.

    Parameters:
    - json_files: A list of paths to JSON files containing block data.
    - n_blocks: The number of blocks to process from each file. Can be an integer or 'all' to process all blocks.

    Returns:
    - A pandas DataFrame containing the extracted data for each block.
    """
        
    all_blocks_data = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            blocks = json.load(f)
            source_file_name = json_file.split('/')[-1]  # Assuming you want to keep only the filename

            # Process only the specified number of blocks if n_blocks is not 'all'
            if n_blocks != 'all':
                try:
                    n_blocks_int = int(n_blocks)  # Convert n_blocks to integer if possible
                    blocks = blocks[:n_blocks_int]  # Limit the number of blocks
                except ValueError:
                    print(f"Warning: The value of n_blocks ('{n_blocks}') is not valid. Processing all blocks.")

            blocks_data = []
            
            # Extract the texts of the blocks, processing either all or a limited number
            for block in blocks:
                text = block.get('Text', 'No text') 
                block_data = {
                    'block_id': block['Id'],
                    'block_type': block['BlockType'],
                    'block_page': block['Page'],
                    'block_content': text,
                    'block_width': block['Geometry']['BoundingBox']['Width'],
                    'block_height': block['Geometry']['BoundingBox']['Height'],
                    'block_left': block['Geometry']['BoundingBox']['Left'],
                    'block_top': block['Geometry']['BoundingBox']['Top'],
                    'confidence': block.get('Confidence', np.nan),
                    'source': source_file_name
                }
                blocks_data.append(block_data)
            
            all_blocks_data.extend(blocks_data)
    
    blocks_df = pd.DataFrame(all_blocks_data)
    print(f"Extracted {n_blocks} from each file. Blocks DataFrame has {blocks_df.shape[0]} rows and {blocks_df.shape[1]} columns")
    
    return blocks_df

def add_page_id(df):
    # Create a copy of the DataFrame to not modify the original in place
    result_df = df.copy()
    
    # Create a dictionary mapping from (source, block_page) to block_id for PAGE blocks
    page_id_map = result_df[result_df['block_type'] == 'PAGE'].set_index(['source', 'block_page'])['block_id'].to_dict()
    
    # Function to retrieve page_id based on a row's source and block_page
    def get_page_id(row):
        return page_id_map.get((row['source'], row['block_page']))
    
    # Apply the function to each row to assign the page_id
    result_df['page_id'] = result_df.apply(get_page_id, axis=1)
    
    # Return the modified DataFrame
    return result_df

