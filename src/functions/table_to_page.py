def find_table_page_relationships(json_data):
    # Step 1: Identify all tables and their IDs
    table_ids = [item['Id'] for item in json_data if item['BlockType'] == 'TABLE']

    # Initialize a dictionary to hold the mapping of table IDs to page IDs
    table_to_page = {}

    # Step 2: For each table, find the page it belongs to
    for table_id in table_ids:
        for item in json_data:
            # Check if the item is a page and has relationships
            if item.get('BlockType') == 'PAGE' and 'Relationships' in item:
                for relationship in item['Relationships']:
                    # If the table ID is found in the CHILD section of a page block
                    if relationship['Type'] == 'CHILD' and table_id in relationship['Ids']:
                        # Map the table ID to the page ID
                        table_to_page[table_id] = item['Id']
                        break  # Stop searching once the parent page is found for this table

    return table_to_page
