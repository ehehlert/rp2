# python -m src.functions.eee_extract_metadata

import json
import pandas as pd
import numpy as np

import pandas as pd
import json

from src.functions.extract_blocks import extract_blocks
from src.functions.get_json_multiple import get_json_multiple

def filter_blocks_by_terms(blocks_df, df_first_page_ids, search_terms=['Identification', 'Location']):
    """
    Filters blocks for search terms and first_page_ids, compiling results into a DataFrame.

    Parameters:
    - blocks_df: DataFrame containing block data, output from extract_blocks.
    - df_first_page_ids: DataFrame with 'first_page_id' to filter blocks.
    - search_terms: Terms to search for within blocks.

    Returns:
    - DataFrame with filtered blocks based on search terms and first_page_ids.
    """
    # Convert first_page_ids to a set for efficient lookups
    first_page_ids = set(df_first_page_ids['first_page_id'])
    
    # Filter the blocks DataFrame for relevant pages
    filtered_blocks = blocks_df[blocks_df['page_id'].isin(first_page_ids)]
    
    # Further filter for LINE blocks containing any of the search terms
    results = []
    for term in search_terms:
        term_matches = filtered_blocks[
            (filtered_blocks['block_type'] == 'LINE') &
            (filtered_blocks['block_content'].str.contains(term, case=False, na=False))
        ]
        for _, row in term_matches.iterrows():
            results.append({
                'first_page_id': row['page_id'],
                'search_term': term,
                'text': row['block_content']
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df



