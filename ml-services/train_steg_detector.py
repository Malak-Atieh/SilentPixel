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

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device):
    """
    Train the steganography detection model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
    
    Returns:
        Trained model, training history
    """
    model = model.to(device)
    
    # For tracking best model
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    
    # History
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Batch loop
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model.state_dict()
                
        logger.info('')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Evaluate the model
def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Metrics dictionary, predictions, true labels
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics, all_preds, all_labels
