"""
Data generation script for steganography detection model training.

This script:
1. Takes clean images
2. Embeds hidden data using LSB and DCT steganography methods
3. Saves the steganographically modified images for model training
"""
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

# ---- LSB Steganography Implementation ----

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
    """
    Encode data into image using Least Significant Bit steganography
    
    Args:
        image: NumPy array of image
        secret_data: String to hide in the image
        delimiter: String delimiter to mark the end of hidden data
        
    Returns:
        Modified image with hidden data
    """
    # Add delimiter to the secret data
    secret_data += delimiter
    
    # Convert secret data to binary
    binary_secret_data = to_binary(secret_data)
    
    # Calculate required pixels
    data_len = len(binary_secret_data)
    
    # Get image dimensions
    height, width, channels = image.shape
    
    # Check if the image has enough pixels
    if height * width * channels < data_len:
        raise ValueError("Image too small to hold this data")
    
    # Create copy of image to modify
    encoded_image = np.copy(image)
    
    # Counter for binary data index
    data_index = 0
    
    # Encode data into image
    for row in range(height):
        for col in range(width):
            for channel in range(channels):
                if data_index < data_len:
                    # Get the pixel value
                    pixel = encoded_image[row, col, channel]
                    
                    # Convert pixel to binary
                    binary_pixel = to_binary(pixel)
                    
                    # Replace LSB with current bit of secret data
                    new_binary = binary_pixel[:-1] + binary_secret_data[data_index]
                    
                    # Convert binary back to integer
                    encoded_image[row, col, channel] = int(new_binary, 2)
                    
                    # Move to next bit of secret data
                    data_index += 1
                else:
                    # All data has been hidden
                    break
    
    return encoded_image

