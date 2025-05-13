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
