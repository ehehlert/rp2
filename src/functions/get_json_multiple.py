import os
import glob

def get_json_multiple(path):
    # Define the path to your directory containing the JSON files
    directory_path = path

    # Construct the pattern to match all .json files in the directory
    pattern = os.path.join(directory_path, '*.json')

    # Use glob.glob() to find all files in the directory that match the pattern
    json_files = glob.glob(pattern)

    return json_files

