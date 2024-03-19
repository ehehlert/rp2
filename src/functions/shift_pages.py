import pandas as pd
import numpy as np

def shift_pages(df):
    # Step 1: Sort the DataFrame
    df_sort = df.sort_values(by=['source', 'page_number'])

    # Step 2: Shift the 'page_id' and 'predicted_page_type' within each 'source' group
    df_sort['previous_page_id'] = df_sort.groupby('source')['page_id'].shift(1)
    df_sort['previous_predicted_page_type'] = df_sort.groupby('source')['predicted_page_type'].shift(1)

    # Step 3: Set the 'previous_page_id' and 'previous_predicted_page_type' to null for the first page of each source
    df_sort.loc[df['page_number'] == 1, ['previous_page_id', 'previous_predicted_page_type']] = np.nan

    return df_sort

def shift_pages_continuation(df):
    # Step 1: Sort the DataFrame
    df_sort = df.sort_values(by=['source', 'page_number'])

    # Step 2: Shift the 'page_id' and 'predicted_page_type' within each 'source' group
    df_sort['previous_page_id'] = df_sort.groupby('source')['page_id'].shift(1)
    df_sort['previous_predicted_continuation'] = df_sort.groupby('source')['predicted_continuation'].shift(1)

    # Step 3: Set the 'previous_page_id' and 'previous_predicted_page_type' to null for the first page of each source
    df_sort.loc[df['page_number'] == 1, ['previous_page_id', 'previous_predicted_continuation']] = np.nan

    return df_sort

import pandas as pd

def group_pages_by_continuation(df):
    # Initialize a list to store group information
    groups = []
    
    # Variables to track the first page ID in the current group and continuation page IDs
    first_page_id = None
    continuation_page_ids = []
    
    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        page_id = row['page_id']
        predicted_continuation = row['predicted_continuation']
        
        # Handle the start of a new group
        if predicted_continuation == 0:
            if first_page_id is not None:
                # If it's not the first group, add the previous group to the list
                groups.append({
                    'first_page_id': first_page_id,
                    'continuation_page_ids': continuation_page_ids
                })
                continuation_page_ids = []  # Reset for the next group
            first_page_id = page_id  # Set the new first page ID
        else:
            # For continuation pages, just add the page ID to the list
            continuation_page_ids.append(page_id)
    
    # Don't forget to add the last group to the list
    if first_page_id is not None:
        groups.append({
            'first_page_id': first_page_id,
            'continuation_page_ids': continuation_page_ids
        })
    
    # Convert the list of groups to a DataFrame
    df_grouped = pd.DataFrame(groups)
    
    # Convert the list of continuation_page_ids to a comma-separated string
    df_grouped['continuation_page_ids'] = df_grouped['continuation_page_ids'].apply(lambda x: ','.join(x) if x else '')
    
    return df_grouped

# Example usage:
# Assuming df is your DataFrame with columns 'page_id', 'predicted_continuation', 'previous_page_id'
# df_grouped = group_pages_by_continuation(df)
# print(df_grouped.head())

