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
             "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", 
             "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"]
    
    length = random.randint(min_length, max_length)
    text = " ".join(random.choice(words) for _ in range(length))
    return text


def dct_encode(image, secret_data, alpha=0.1):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    h = h - (h % 8)
    w = w - (w % 8)
    gray = gray[:h, :w]
    
    binary_data = to_binary(secret_data)
    data_len = len(binary_data)
    
    max_capacity = (h // 8) * (w // 8)
    if data_len > max_capacity:
        raise ValueError(f"Data too large for this image. Max capacity: {max_capacity} bits")
    
    data_index = 0
    modified_image = gray.copy().astype(np.float32)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if data_index < data_len:
                block = modified_image[i:i+8, j:j+8]
                
                dct_block = cv2.dct(block)
 
                bit = int(binary_data[data_index])
                
                if bit == 0:
                    if int(dct_block[2, 1]) % 2 == 1:
                        dct_block[2, 1] -= alpha
                else:
                    if int(dct_block[2, 1]) % 2 == 0:
                        dct_block[2, 1] += alpha
                
                modified_block = cv2.idct(dct_block)
                
                modified_image[i:i+8, j:j+8] = modified_block
                
                data_index += 1
    
    modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
    
    return modified_image


def process_image(args):
    image_path, output_dir, delimiter = args
    
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        basename = os.path.splitext(os.path.basename(image_path))[0]
        
        secret_data = generate_random_text(200, 800)
     