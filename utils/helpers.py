import logging

def configure_logging():
    logging.basicConfig(filename='logs/processing.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

import torch 

def check_gpu_status():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (f"{x}; MPS device found.")
    else:
        print ("MPS device not found.")

