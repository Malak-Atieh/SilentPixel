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
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SteganographyDataset(Dataset):
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

def build_model(num_classes=3, model_name="mobilenet_v3", freeze_features=True):
    if model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        in_features = 576
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    elif model_name == "efficientnet_b0":

    elif model_name == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
        in_features = model.fc.in_features
        if freeze_features:
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
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


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_metrics(history, output_dir):
    """Plot and save training history metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='./models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--model', type=str, default='mobilenet_v3',
                        choices=['mobilenet_v3', 'efficientnet_b0', 'resnet18'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau'])
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = SteganographyDataset(args.data_dir)
    labels = [label for _, label in full_dataset.files]

    # Stratified split: train/val/test = 70/15/15
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        list(range(len(full_dataset))), labels, test_size=0.3, stratify=labels, random_state=42)

    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # Assign transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Handle class imbalance
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = (1. / class_counts.float()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(num_classes=3, model_name=args.model, freeze_features=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
                 if args.scheduler == 'cosine'
                 else optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2))

    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        logger.info(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {time.time()-start_time:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
        elif (epoch - np.argmax(history['val_acc'])) >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    plot_metrics(history, args.output_dir)

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss, test_acc, all_preds, all_labels = validate(model, test_loader, criterion, device)
    logger.info(f'Final Test Accuracy: {test_acc:.4f}')
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logger.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    plot_confusion_matrix(all_labels, all_preds, ['Clean', 'LSB', 'DCT'], args.output_dir)

    torch.save(model.state_dict(), os.path.join(args.output_dir,
                                                f'steg_detector_{args.model}_acc{test_acc:.4f}.pth'))
if __name__ == '__main__':
    main()