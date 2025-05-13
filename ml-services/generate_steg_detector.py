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

# ---- DCT Steganography Implementation ----

def dct_encode(image, secret_data, alpha=0.1):
    """
    Encode data using DCT (Discrete Cosine Transform) method
    
    Args:
        image: Grayscale image as numpy array
        secret_data: Binary string to hide
        alpha: Strength factor for embedding
        
    Returns:
        Image with hidden data using DCT coefficients
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Ensure image dimensions are multiples of 8
    h, w = gray.shape
    h = h - (h % 8)
    w = w - (w % 8)
    gray = gray[:h, :w]
    
    # Prepare secret message
    binary_data = to_binary(secret_data)
    data_len = len(binary_data)
    
    # Check capacity
    max_capacity = (h // 8) * (w // 8)
    if data_len > max_capacity:
        raise ValueError(f"Data too large for this image. Max capacity: {max_capacity} bits")
    
    # Split image into 8x8 blocks and apply DCT
    data_index = 0
    modified_image = gray.copy().astype(np.float32)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if data_index < data_len:
                # Extract 8x8 block
                block = modified_image[i:i+8, j:j+8]
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Modify DCT coefficient (2,1) based on secret bit
                # We choose mid-frequency coefficient for better hiding
                bit = int(binary_data[data_index])
                
                # Embed data by modifying coefficient
                if bit == 0:
                    # Make coefficient even
                    if int(dct_block[2, 1]) % 2 == 1:
                        dct_block[2, 1] -= alpha
                else:
                    # Make coefficient odd
                    if int(dct_block[2, 1]) % 2 == 0:
                        dct_block[2, 1] += alpha
                
                # Apply inverse DCT
                modified_block = cv2.idct(dct_block)
                
                # Replace block in image
                modified_image[i:i+8, j:j+8] = modified_block
                
                data_index += 1
    
    # Clip values to valid range [0, 255]
    modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
    
    return modified_image

# ---- Main Data Generation Functions ----

def process_image(args):
    """Process a single image (to be used with parallel processing)"""
    image_path, output_dir, delimiter = args
    
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Get basename for output files
        basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create LSB steganography
        secret_data = generate_random_text(200, 800)
        lsb_img = encode_lsb(img_array, secret_data, delimiter)
        lsb_output = os.path.join(output_dir, 'lsb', f"{basename}_lsb.png")
        Image.fromarray(lsb_img).save(lsb_output)
        
        # Create DCT steganography
        # DCT works on grayscale images
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        secret_data = generate_random_text(50, 200)  # Smaller text as DCT has less capacity
        dct_img = dct_encode(gray_img, secret_data)
        
        # Convert back to RGB for consistent output
        dct_rgb = cv2.cvtColor(dct_img, cv2.COLOR_GRAY2RGB)
        dct_output = os.path.join(output_dir, 'dct', f"{basename}_dct.png")
        Image.fromarray(dct_rgb).save(dct_output)
        
        # Also save clean copy
        clean_output = os.path.join(output_dir, 'clean', f"{basename}_clean.png")
        img.save(clean_output)
        
        return image_path, True
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return image_path, False


def generate_dataset(input_dir, output_dir, num_workers=4, delimiter='#####'):
    """
    Generate steganography dataset from clean images
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory to save generated dataset
        num_workers: Number of parallel workers
        delimiter: Delimiter for LSB steganography
    """
    # Create output directories
    for subdir in ['clean', 'lsb', 'dct']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(input_dir).glob(ext)))
    
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Prepare arguments for parallel processing
    args_list = [(str(img_path), output_dir, delimiter) for img_path in image_files]
    
    # Process images in parallel
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating dataset"):
            img_path, success = future.result()
            if success:
                successful += 1
            else:
                failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    logger.info(f"Dataset generated at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate steganography dataset")
    parser.add_argument("--input", required=True, help="Directory with input images")
    parser.add_argument("--output", required=True, help="Directory to save generated dataset")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--delimiter", default="#####", help="Delimiter for LSB steganography")
    
    args = parser.parse_args()
    
    generate_dataset(args.input, args.output, args.workers, args.delimiter)