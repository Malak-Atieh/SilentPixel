"""
Optimized training script for the steganography detection CNN model.
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
from torchvision import transforms, models
from PIL import Image
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler, autocast  # For mixed precision training

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

# Build pretrained model with improved architecture
def build_pretrained_model(num_classes=3, freeze_layers=7):
    model = models.resnet18(weights='IMAGENET1K_V1') 
    
    # Freeze only early layers for better transfer learning
    if freeze_layers > 0:
        ct = 0
        for child in model.children():
            ct += 1
            if ct <= freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False
    
    # Replace final fully connected layer with improved classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.7),  # Increase dropout
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

# Training function with mixed precision and early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=7, use_amp=True):
    model = model.to(device)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None  # Only use scaler when AMP is enabled
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stop_counter = 0

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}\n----------')
        start_epoch_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            dataloader = train_loader if phase == 'train' else val_loader
            running_loss = 0.0
            running_corrects = 0
            batch_count = 0

            for inputs, labels in dataloader:
                batch_count += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                if phase == 'train':
                    if use_amp:
                        # Forward pass with mixed precision
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        # Scale gradients and optimize
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard forward and backward pass without AMP
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Print progress every 20 batches
                if batch_count % 20 == 0:
                    logger.info(f'  {phase} batch {batch_count}/{len(dataloader)}, loss: {loss.item():.4f}')

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            epoch_time = time.time() - start_epoch_time
            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s')

            # Save best validation model
            if phase == 'val':
                if epoch_acc > best_val_acc:
                    logger.info(f'Validation accuracy improved from {best_val_acc:.4f} to {epoch_acc:.4f}')
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    logger.info(f'Validation accuracy did not improve. Counter: {early_stop_counter}/{patience}')
                
                # Update learning rate based on validation metrics
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_acc)
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f'Current learning rate: {current_lr:.2e}')

        # Check early stopping
        if early_stop_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
            
        # Step scheduler if not ReduceLROnPlateau
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
            
            
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Evaluate the function
def evaluate_model(model, test_loader, device):
    model.to(device).eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }, all_preds, all_labels

# Plot functions
def plot_history(history, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Main function
def main():
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--train_val_test_split', type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--freeze_layers', type=int, default=6, help='Number of initial layers to freeze')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--quick_mode', action='store_true', help='Enable quick testing mode with smaller dataset')
    args = parser.parse_args()

    set_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device with error handling
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
        use_amp = True  # Use Automatic Mixed Precision for CUDA
    else:
        device = torch.device('cpu')
        logger.info('CUDA not available. Using CPU - disabling mixed precision')
        use_amp = False  # Disable Automatic Mixed Precision for CPU

    # Data transformations - smaller image size for faster training
    data_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    logger.info(f'Loading dataset from {args.data_dir}')
    full_dataset = SteganographyDataset(args.data_dir, transform=data_transforms)
    dataset_size = len(full_dataset)
    
    # Option for quick testing with smaller dataset
    if args.quick_mode:
        reduced_size = min(500, dataset_size)  # Limit to 500 samples for quick testing
        logger.info(f'Quick mode enabled: using {reduced_size} samples instead of {dataset_size}')
        indices = torch.randperm(dataset_size)[:reduced_size]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        dataset_size = reduced_size
    
    train_size = int(dataset_size * args.train_val_test_split[0])
    val_size = int(dataset_size * args.train_val_test_split[1])
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply different transforms to test dataset
    test_dataset.dataset.transform = test_transforms

    # Create data loaders with appropriate settings for CPU/GPU
    # Adjust num_workers to 0 for Windows if encountering issues
    if os.name == 'nt' and args.num_workers > 0:  # Windows platform
        logger.info(f'Windows detected, consider setting --num_workers=0 if you encounter issues')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=device.type=='cuda'  # Only use pinned memory with CUDA
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=device.type=='cuda'
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=device.type=='cuda'
    )

    # Initialize model with selective layer freezing
    logger.info(f'Initializing model with {args.freeze_layers} frozen layers')
    model = build_pretrained_model(num_classes=3, freeze_layers=args.freeze_layers)
    
    # Loss function with class weights if needed
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with parameter filtering
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate * 0.1, 
        weight_decay=2e-4
    )
    
    # Learning rate scheduler 
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
        pct_start=0.2  # Warm up for first 20% of training
    )
    # Start training
    logger.info('Starting training with optimized configuration...')
    start_time = time.time()
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        device,
        patience=args.patience,
        use_amp=use_amp
    )
    logger.info(f'Training complete in {(time.time() - start_time)/60:.2f} minutes')

    # Save model
    model_path = os.path.join(args.output_dir, 'steg_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.num_epochs,
        'hyperparams': vars(args)
    }, model_path)
    logger.info(f'Model saved to {model_path}')

    # Evaluate model
    logger.info('Evaluating model on test set...')
    metrics, predictions, true_labels = evaluate_model(model, test_loader, device)
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1']:.4f}")

    # Save evaluation metrics
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                f.write(f"{k}: {v}\n")

    # Plot and save results
    plot_history(history, os.path.join(args.output_dir, 'training_history.png'))
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        classes=['Clean', 'LSB', 'DCT'], 
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    logger.info('Evaluation complete!')
    
if __name__ == '__main__':
    main()