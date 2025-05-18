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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SteganographyDataset(Dataset):
    """Optimized dataset with caching"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.cache = {} 
        
        self.clean_files = list((self.data_dir / 'clean').glob('*.[pj][np]g'))
        self.lsb_files = list((self.data_dir / 'lsb').glob('*.[pj][np]g'))
        self.dct_files = list((self.data_dir / 'dct').glob('*.[pj][np]g'))
        
        self.files = ([(f, 0) for f in self.clean_files] +
                     [(f, 1) for f in self.lsb_files] +
                     [(f, 2) for f in self.dct_files])
        
        logger.info(f"Dataset: {len(self.clean_files)} clean, "
                   f"{len(self.lsb_files)} LSB, {len(self.dct_files)} DCT")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
            
        img_path, label = self.files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            self.cache[idx] = (img, label)
            return img, label
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            return self[np.random.randint(len(self))] 

def build_model(num_classes=3):
    """Lightweight MobileNetV3 model"""
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        
    return total_loss / len(loader.dataset), correct.double() / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            
    return total_loss / len(loader.dataset), correct.double() / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='./models')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simplified transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load dataset with automatic class weights
    full_dataset = SteganographyDataset(args.data_dir)
    class_counts = torch.bincount(torch.tensor([label for _, label in full_dataset.files]))
    class_weights = (1. / class_counts.float()).to(device)
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           num_workers=args.num_workers)

    # Initialize model
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    # Training loop
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs} | '
                   f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                   f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
                   f'Time: {time.time()-start_time:.1f}s')
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        elif (epoch - np.argmax(history['val_acc'])) >= args.patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    # Final evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f'Final Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()