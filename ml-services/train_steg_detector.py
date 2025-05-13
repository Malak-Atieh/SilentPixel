"""
Training script for the steganography detection CNN model.
"""
import os
import time
import logging
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Import model from app.py
from app import SteganographyCNN

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Steganography dataset
class SteganographyDataset(Dataset):
    """Dataset for steganography detection training."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory with subdirectories: 'clean', 'lsb', 'dct'
            transform: Optional transform to be applied to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all image files in each class directory
        self.clean_files = list((self.data_dir / 'clean').glob('*.png')) + \
                         list((self.data_dir / 'clean').glob('*.jpg'))
        self.lsb_files = list((self.data_dir / 'lsb').glob('*.png')) + \
                       list((self.data_dir / 'lsb').glob('*.jpg'))
        self.dct_files = list((self.data_dir / 'dct').glob('*.png')) + \
                       list((self.data_dir / 'dct').glob('*.jpg'))
        
        # Create list of (file_path, class_label) pairs
        self.files = [(file, 0) for file in self.clean_files] + \
                   [(file, 1) for file in self.lsb_files] + \
                   [(file, 2) for file in self.dct_files]
        
        logger.info(f"Found {len(self.clean_files)} clean images, "
                   f"{len(self.lsb_files)} LSB steganography images, "
                   f"{len(self.dct_files)} DCT steganography images")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path, class_label = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_label

