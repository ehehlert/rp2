import json
import pandas as pd

def extract_cell_blocks(json_files):
    all_cell_records = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            source_file_name = json_file.split('/')[-1]  # Assuming you want to keep only the filename

            id_to_item = {item['Id']: item for item in data}

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
                    child_table_ids = parent_to_all_children.get(page_id, {}).get('CHILD', [])
                    for table_id in child_table_ids:
                        if table_id in id_to_item and id_to_item[table_id]['BlockType'] == 'TABLE':
                            table_to_page[table_id] = page_id

            cell_records = []

            for item in data:
                if item.get('BlockType') == 'TABLE':
                    table_id = item['Id']
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
                            'table_type': item['EntityTypes'][0] if 'EntityTypes' in item else None,
                            'table_width': item['Geometry']['BoundingBox']['Width'],
                            'table_height': item['Geometry']['BoundingBox']['Height'],
                            'table_left': item['Geometry']['BoundingBox']['Left'],
                            'table_top': item['Geometry']['BoundingBox']['Top'],
                            'table_page': item['Page'],
                            'page_id': page_id,
                            'source': source_file_name,
                        })

            all_cell_records.extend(cell_records)

    blocks_df = pd.DataFrame(all_cell_records)

    print(f"Extracted Cells DataFrame has {blocks_df.shape[0]} rows and {blocks_df.shape[1]} columns")
    
    return blocks_df