# run as a module: ~ python -m src.functions.textract_multiple

import logging
from src.functions.textract import main
from utils.helpers import configure_logging
from config.documents_config import documents_config

configure_logging()

for document_name in documents_config:
    try:
        logging.info(f"Processing document: {document_name}")
        print(f"Processing document: {document_name}")
        main(document_name)
    except Exception as e:
        logging.error(f"Failed to process document {document_name}: {e}")
        print(f"Failed to process document {document_name}. Check the log for details.")
