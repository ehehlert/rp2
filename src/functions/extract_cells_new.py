import json
import pandas as pd

from src.functions.table_to_page import find_table_page_relationships

import json
import pandas as pd

def extract_cell_blocks(json_files):
    all_cell_records = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            source_file_name = json_file.split('/')[-1]

            id_to_item = {item['Id']: item for item in data}

            # Mapping to keep track of page numbers for each page ID
            page_id_to_number = {item['Id']: item['Page'] for item in data if item.get('BlockType') == 'PAGE'}

            parent_to_all_children = {}
            for item in data:
                if 'Relationships' in item:
                    for relationship in item['Relationships']:
                        if item['Id'] not in parent_to_all_children:
                            parent_to_all_children[item['Id']] = {}
                        parent_to_all_children[item['Id']].setdefault(relationship['Type'], []).extend(relationship['Ids'])

            table_to_page = {}
            for item in data:
                if item.get('BlockType') == 'PAGE':
                    page_id = item['Id']
                    table_page = page_id_to_number.get(page_id)
                    child_table_ids = parent_to_all_children.get(page_id, {}).get('CHILD', [])
                    for table_id in child_table_ids:
                        if table_id in id_to_item and id_to_item[table_id]['BlockType'] == 'TABLE':
                            table_to_page[table_id] = page_id

            cell_records = []

            for item in data:
                if item.get('BlockType') == 'TABLE':
                    table_id = item['Id']
                    page_id = table_to_page.get(table_id)

                    # Additional properties for table geometry
                    table_geometry = item.get('Geometry', {}).get('BoundingBox', {})
                    table_polygon = item.get('Geometry', {}).get('Polygon', [])

                    relationships = parent_to_all_children.get(table_id, {})

                    aggregated_cells = []
                    for rel_type in ['CHILD', 'MERGED_CELL', 'TABLE_FOOTER', 'TABLE_TITLE']:
                        cell_ids = relationships.get(rel_type, [])
                        for cell_id in cell_ids:
                            aggregated_cells.append((cell_id, rel_type))

                    for cell_id, cell_type in aggregated_cells:
                        cell_block = id_to_item.get(cell_id)
                        if not cell_block:
                            continue

                        entity_type = cell_block.get('EntityTypes', [None])[0] if 'EntityTypes' in cell_block else None
                        cell_geometry = cell_block['Geometry']['BoundingBox']

                        child_ids = parent_to_all_children.get(cell_id, {}).get('CHILD', [])
                        cell_words = [id_to_item[child_id]['Text'] for child_id in child_ids if child_id in id_to_item and 'Text' in id_to_item[child_id]]
                        cell_content = ' '.join(cell_words)

                        # Including all table geometry details and other related properties in the record
                        cell_records.append({
                            'cell_id': cell_id,
                            'cell_type': cell_type,  
                            'entity_type': entity_type,
                            'cell_words': cell_words,
                            'cell_content': cell_content,
                            'cell_width': cell_geometry['Width'],
                            'cell_height': cell_geometry['Height'],
                            'cell_left': cell_geometry['Left'],
                            'cell_top': cell_geometry['Top'],
                            'row_index': cell_block.get('RowIndex', None),
                            'column_index': cell_block.get('ColumnIndex', None),
                            'row_span': cell_block.get('RowSpan', 1),
                            'column_span': cell_block.get('ColumnSpan', 1),
                            'table_id': table_id,
                            'table_type': item.get('EntityTypes', [None])[0],
                            'table_width': table_geometry['Width'],
                            'table_height': table_geometry['Height'],
                            'table_left': table_geometry['Left'],
                            'table_top': table_geometry['Top'],
                            'table_page': table_page,  # Added table polygon
                            'page_id': page_id,
                            'source': source_file_name,
                        })

            all_cell_records.extend(cell_records)

    blocks_df = pd.DataFrame(all_cell_records)

    print(f"Extracted Cells DataFrame has {blocks_df.shape[0]} rows and {blocks_df.shape[1]} columns")
    
    return blocks_df

