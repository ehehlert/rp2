# python -m src.functions.trigger_textract_job

# This script is used to trigger textract jobs for all the pdfs in the staging folder
# It will upload the pdf to s3, trigger the textract job, and then delete the pdf from s3
# JSON output will be saved in the textract_jobs/complete/json folder
# The original PDFs will also be moved to the textract_jobs/complete/pdfs folder

import boto3
import logging
from botocore.exceptions import ClientError
from src.functions.textract import main as process_document_with_textract_async
from utils.helpers import configure_logging
from dotenv import load_dotenv
from src.functions.get_pdf_multiple import get_pdf_multiple
import os
import shutil

s3_client = boto3.client('s3')
configure_logging()
load_dotenv()

bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
file_paths = get_pdf_multiple('textract_jobs/staging')

def upload_file_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def delete_document_from_s3(bucket, document):
    try:
        s3_client.delete_object(Bucket=bucket, Key=document)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def move_files(file_paths, target_dir):
    # Ensure the target directory exists; if not, create it.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop through all file paths and move each file.
    for file_path in file_paths:
        try:
            # Construct the full target path. The file retains its original name.
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            
            # Move the file.
            shutil.move(file_path, target_path)
            print(f"Moved: {file_path} -> {target_path}")
        except Exception as e:
            print(f"Error moving {file_path} to {target_dir}: {e}")

def process_and_delete_files(bucket_name, file_paths):
    for file_path in file_paths:
        object_name = file_path.split('/')[-1]  # Extract filename for S3 object name

        # Step 1: Upload file to S3
        if upload_file_to_s3(file_path, bucket_name, object_name):
            logging.info(f"Uploaded {object_name} to S3 bucket {bucket_name}.")
            
            # Step 2: Process the document with Textract
            try:
                process_document_with_textract_async(object_name)
                logging.info(f"Processing completed for {object_name}.")
                
                # Step 3: Delete the document from S3
                if delete_document_from_s3(bucket_name, object_name):
                    logging.info(f"Deleted {object_name} from S3 bucket {bucket_name}.")
                else:
                    logging.error(f"Failed to delete {object_name} from S3 bucket {bucket_name}.")
            
            except Exception as e:
                logging.error(f"Failed to process {object_name} due to {e}")
                continue  # Proceed with next file or handle as needed
        else:
            logging.error(f"Failed to upload {object_name} to S3 bucket {bucket_name}.")

if __name__ == "__main__":
    process_and_delete_files(bucket_name, file_paths)
    move_files(file_paths, 'textract_jobs/complete/pdfs')

