import os
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import textwrap
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def to_binary(data):
    """Convert data to binary format as a string of bits"""
    if isinstance(data, str):
        return ''.join([format(ord(i), '08b') for i in data])
    elif isinstance(data, bytes) or isinstance(data, bytearray):
        return ''.join([format(i, '08b') for i in data])
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, '08b')
    else:
        raise TypeError("Type not supported.")


def encode_lsb(image, secret_data, delimiter='#####'):

    secret_data += delimiter
    
    binary_secret_data = to_binary(secret_data)
    
    data_len = len(binary_secret_data)
    
    height, width, channels = image.shape
    
    if height * width * channels < data_len:
