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
        raise ValueError("Image too small to hold this data")
    
    encoded_image = np.copy(image)
    
    data_index = 0
    
    for row in range(height):
        for col in range(width):
            for channel in range(channels):
                if data_index < data_len:
                    pixel = encoded_image[row, col, channel]
                    
                    binary_pixel = to_binary(pixel)
                    
                    new_binary = binary_pixel[:-1] + binary_secret_data[data_index]
                    
                    encoded_image[row, col, channel] = int(new_binary, 2)
                    
                    data_index += 1
                else:
                    break
    
    return encoded_image

def generate_random_text(min_length=100, max_length=500):
    """Generate random text for hiding in images"""
    words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", 
             "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", 
             "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", 
             "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", 
             "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", 
             "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", 
             "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", 
             "also", "back", "after", "use", "two", "how", "our", "work", 