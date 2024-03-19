# python -m src.functions.upload_to_s3

import logging
import boto3
from botocore.exceptions import ClientError
import os
from utils.helpers import configure_logging
from src.functions.get_pdf_multiple import get_pdf_multiple

from dotenv import load_dotenv

load_dotenv()
configure_logging()

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_multiple_files(files, bucket):
    for file in files:
        if upload_file(file, bucket):
            logging.info(f"File {file} uploaded to {bucket}")
        else:
            logging.error(f"File {file} failed to upload to {bucket}")

if __name__ == "__main__":
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    files = get_pdf_multiple('pdfs/mixed')
    upload_multiple_files(files, bucket)