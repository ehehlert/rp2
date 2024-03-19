import json
import pandas as pd
import numpy as np

def extract_tables(json_files, cells_df):
    """
    Process document layout titles from multiple JSON files and aggregate cell contents and child entities
    for each table in the provided cells DataFrame. Then, calculate page titles based on confidence scores,
    and merge with tables information.

    Parameters:
    - json_files: List of paths to JSON files containing document layout titles.
    - cells_df: DataFrame containing detailed cells information.

    Returns:
    - tables_df: DataFrame containing tables information merged with page titles.
    """

    def aggregate_contents(group):
        sorted_group = group.sort_values(by=['row_index', 'column_index'])
        contents_by_row = sorted_group.groupby('row_index')['cell_words'].apply(list).tolist()
        return contents_by_row
    
    def identify_all_unique_entity_types(cells_df):
        """Identify all unique entity types across the dataset."""
        all_entities = cells_df['entity_type'].fillna('normal')
        return set(all_entities)

    def aggregate_child_entities(group):
        child_cells = group[group['cell_type'] == 'CHILD']
        child_cells['entity_type'] = child_cells['entity_type'].replace({np.nan: 'normal', '': 'normal'})
        sorted_group = child_cells.sort_values(by=['row_index', 'column_index'])
        entities_by_row = sorted_group.groupby('row_index')['entity_type'].apply(list).tolist()
        return entities_by_row
    
    def aggregate_child_entity_types(group, all_entity_types):

        # Define the specific entity types of interest
        entity_types_of_interest = [
            'COLUMN_HEADER', 
            'TABLE_TITLE', 
            'TABLE_SECTION_TITLE', 
            'TABLE_FOOTER', 
            'TABLE_SUMMARY'
        ]

        # Initialize counts for all entity types to zero
        entity_counts = {'entity_' + entity_type.lower(): 0 for entity_type in entity_types_of_interest}    
       
        group['entity_type'] = group['entity_type'].replace({np.nan: 'normal', '': 'normal'})
        
        # Iterate over each row, counting occurrences of the specified entity types
        for _, row in group.iterrows():
            entity_type = row['entity_type']
            if entity_type in entity_types_of_interest:
                # Adjust the key to match the modified naming convention and count it
                key = 'entity_' + entity_type.lower()
                entity_counts[key] += 1
        
        return entity_counts
    
    def process_layout_titles(data, source_file_name):
        layout_titles = []
        for item in data:
            if item.get('BlockType') == 'LAYOUT_TITLE':
                children_texts = [data[child_id]['Text'] for child_id in item.get('Relationships', [{}])[0].get('Ids', []) if child_id in data and 'Text' in data[child_id]] if 'Relationships' in item else []
                layout_title_cell = {
                    'layout_title_id': item['Id'],
                    'layout_title_text': ' '.join(children_texts),
                    'layout_title_page': item.get('Page', 0),
                    'layout_title_confidence': item.get('Confidence', 0),
                    'source': source_file_name,
                }
                layout_titles.append(layout_title_cell)
        return layout_titles

    def process_layout_titles_from_files(json_files):
        all_titles = []
        for file_path in json_files:
            with open(file_path) as file:
                data = json.load(file)
                source_file_name = file_path.split('/')[-1]
                all_titles.extend(process_layout_titles(data, source_file_name))
        titles_df = pd.DataFrame(all_titles)
        if not titles_df.empty:
            titles_df['max_confidence_per_page'] = titles_df.groupby(['source', 'layout_title_page'])['layout_title_confidence'].transform('max')
            titles_df['is_max_confidence'] = titles_df['layout_title_confidence'] == titles_df['max_confidence_per_page']
            titles_df = titles_df[titles_df['is_max_confidence']]  # Filter to only include rows with max confidence
            titles_df.drop(columns=['max_confidence_per_page', 'is_max_confidence'], inplace=True, errors='ignore')
        return titles_df

    all_unique_entity_types = identify_all_unique_entity_types(cells_df)
    titles_df = process_layout_titles_from_files(json_files)

    def table_aggregator(group):
        # This function now includes a call to aggregate_child_entity_types and incorporates its output.
        contents = aggregate_contents(group)
        entities = aggregate_child_entities(group)
        entity_types_counts = aggregate_child_entity_types(group, all_unique_entity_types)

        return pd.Series({
            'table_width': group['table_width'].max(),
            'table_height': group['table_height'].max(),
            'table_left': group['table_left'].max(),
            'table_top': group['table_top'].max(),
            'table_page': group['table_page'].max(),
            'source': group['source'].iloc[0],
            'cell_count': group['cell_words'].count(),
            'row_count': int(group['row_index'].max()),
            'column_count': int(group['column_index'].max()),
            'content': contents,
            'entities': entities,
            # Incorporate entity_types_counts directly into the output Series
            **entity_types_counts,
            'child_count': group[group['cell_type'] == 'CHILD']['cell_type'].count(),
            'merged_cell_count': group[group['cell_type'] == 'MERGED_CELL']['cell_type'].count(),
            'table_title_count': group[group['cell_type'] == 'TABLE_TITLE']['cell_type'].count(),
            'table_footer_count': group[group['cell_type'] == 'TABLE_FOOTER']['cell_type'].count(),
            'table_type': group['table_type'].max()
        })

    tables_df_without_titles = cells_df.groupby('table_id').apply(table_aggregator).reset_index()

    # Merge titles with the tables information
    tables_df = pd.merge(tables_df_without_titles, titles_df[['source', 'layout_title_page', 'layout_title_text']],
                         left_on=['source', 'table_page'], right_on=['source', 'layout_title_page'], how='left').drop(columns=['layout_title_page'], errors='ignore')

    print(f'Derived Tables DataFrame has {tables_df.shape[0]} rows and {tables_df.shape[1]} columns')
  
    return tables_df
