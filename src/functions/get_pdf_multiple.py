import os
import glob

def get_pdf_multiple(path):
    # Define the path to your directory containing the JSON files
    directory_path = path

    # Construct the pattern to match all .json files in the directory
    pattern = os.path.join(directory_path, '*.pdf')

    # Use glob.glob() to find all files in the directory that match the pattern
    pdf_files = glob.glob(pattern)

    return pdf_files

