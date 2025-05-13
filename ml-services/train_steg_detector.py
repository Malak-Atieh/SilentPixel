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


# Plot training history
def plot_history(history, save_path=None):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


# Main function
def main():
    parser = argparse.ArgumentParser(description='Train steganography detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory with clean, lsb, dct subdirectories')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--train_val_test_split', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Proportions for train/val/test split')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = SteganographyDataset(args.data_dir, transform=data_transforms)
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * args.train_val_test_split[0])
    val_size = int(dataset_size * args.train_val_test_split[1])
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Override transforms for test dataset
    test_dataset.dataset.transform = test_transforms
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Initialize model
    model = SteganographyCNN()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    logger.info('Starting training...')
    start_time = time.time()
    
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler, args.num_epochs, device
    )
    
    total_time = time.time() - start_time
    logger.info(f'Training complete in {total_time / 60:.2f} minutes')
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, 'steg_model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f'Model saved to {model_path}')
    
    # Evaluate model on test set
    logger.info('Evaluating model on test set...')
    metrics, predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # Print metrics
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
    
    # Plot and save training history
    history_path = os.path.join(args.output_dir, 'training_history.png')
    plot_history(history, save_path=history_path)
    
    # Plot and save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        classes=['Clean', 'LSB Steg', 'DCT Steg'],
        save_path=cm_path
    )
    
    logger.info('Evaluation complete!')

if __name__ == '__main__':
    main()