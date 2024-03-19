import os

def get_json(path, files_list):
    # define json_files one at a time
    json_files_names = files_list

    # Construct full paths to the JSON files
    json_files = [os.path.join(path, file_name) for file_name in json_files_names]

    return json_files

