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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split

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
    def __init__(self, data_dir, transform=None, max_cache_size=1000):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.cache = {}
        self.max_cache_size = max_cache_size
        
        # Find all image files
        self.clean_files = list((self.data_dir / 'clean').glob('*.[pj][np]g'))
        self.lsb_files = list((self.data_dir / 'lsb').glob('*.[pj][np]g'))
        self.dct_files = list((self.data_dir / 'dct').glob('*.[pj][np]g'))
        
        self.files = ([(f, 0) for f in self.clean_files] +
                     [(f, 1) for f in self.lsb_files] +
                     [(f, 2) for f in self.dct_files])
        
        # Shuffle files
        random.shuffle(self.files)
        
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
            
            # Manage cache size
            if len(self.cache) >= self.max_cache_size:
                # Remove a random item from cache
                remove_key = random.choice(list(self.cache.keys()))
                self.cache.pop(remove_key)
                
            self.cache[idx] = (img, label)
            return img, label
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return a different random sample instead
            return self[np.random.randint(len(self))]

class StegDetectorCNN(nn.Module):
    def __init__(self, num_classes=3, base_model="mobilenet_v3", dropout_rate=0.5):
        super(StegDetectorCNN, self).__init__()
        
        # Choose base model
        if base_model == "mobilenet_v3":
            self.base = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            in_features = self.base.classifier[0].in_features
            self.base.classifier = nn.Identity()
        elif base_model == "efficientnet_b0":
            self.base = models.efficientnet_b0(weights='IMAGENET1K_V1')
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Identity()
        elif base_model == "resnet18":
            self.base = models.resnet18(weights='IMAGENET1K_V1')
            in_features = self.base.fc.in_features
            self.base.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown model name: {base_model}")
            
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                
    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)

def get_transforms(img_size, augment_level='strong'):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment_level == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    elif augment_level == 'light':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize
        ])
    elif augment_level == 'medium':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else: 
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),  
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2)
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, test_transform

def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.2):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)
            
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += batch_size
        
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += batch_size
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return total_loss / total, correct / total, all_preds, all_labels

def plot_results(history, all_preds, all_labels, class_names, output_dir):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def get_lr_scheduler(optimizer, scheduler_type, num_epochs, steps_per_epoch=None, min_lr=1e-6):
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
    elif scheduler_type == 'onecycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.param_groups[0]['lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.2
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5, min_lr=min_lr
        )
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='./models')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--model', type=str, default='mobilenet_v3',
                        choices=['mobilenet_v3', 'efficientnet_b0', 'resnet18'])
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['cosine', 'onecycle', 'plateau'])
    parser.add_argument('--augmentation', type=str, default='strong',
                        choices=['none', 'light', 'medium', 'strong'])
    parser.add_argument('--mixup', type=float, default=0.2,
                        help='Mixup alpha, 0 = disabled')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_folds', type=int, default=1, 
                        help='Number of cross-validation folds, 1 = disabled')
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    full_dataset = SteganographyDataset(args.data_dir)
    
    labels = np.array([label for _, label in full_dataset.files])
    
    train_transform, test_transform = get_transforms(args.img_size, args.augmentation)
    
    ensemble_test_preds = []
    ensemble_test_labels = []
    
    if args.use_folds > 1:
        kfold = StratifiedKFold(n_splits=args.use_folds, shuffle=True, random_state=42)
        fold_results = []
    else:
        kfold = [(np.where(np.random.rand(len(labels)) < 0.8)[0], 
                 np.where(np.random.rand(len(labels)) >= 0.8)[0])]
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels) 
                                              if args.use_folds > 1 else enumerate(kfold)):
        logger.info(f"{'=' * 20} Fold {fold+1}/{args.use_folds if args.use_folds > 1 else 1} {'=' * 20}")
        
        if args.use_folds > 1:
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.2, 
                stratify=labels[train_idx], random_state=42+fold
            )
        else:
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.2,
                stratify=labels[train_idx], random_state=42
            )
        
        logger.info(f"Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = test_transform
        test_dataset.dataset.transform = test_transform
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        model = StegDetectorCNN(
            num_classes=3, 
            base_model=args.model,
            dropout_rate=args.dropout
        ).to(device)
        
        freeze_epochs = min(5, args.num_epochs // 4)
        
        for stage in range(2):
            if stage == 0:
                logger.info("Stage 1: Training classifier only")
                for param in model.base.parameters():
                    param.requires_grad = False
                    
                optimizer = optim.AdamW(
                    model.classifier.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay
                )
                num_epochs = freeze_epochs
            else:
                logger.info("Stage 2: Fine-tuning entire model")
                for param in model.base.parameters():
                    param.requires_grad = True
                    
                optimizer = optim.AdamW([
                    {'params': model.base.parameters(), 'lr': args.learning_rate/10},
                    {'params': model.classifier.parameters(), 'lr': args.learning_rate}
                ], weight_decay=args.weight_decay)
                num_epochs = args.num_epochs - freeze_epochs
            
            if args.scheduler == 'onecycle':
                scheduler = get_lr_scheduler(
                    optimizer, args.scheduler, num_epochs,
                    steps_per_epoch=len(train_loader)
                )
            else:
                scheduler = get_lr_scheduler(optimizer, args.scheduler, num_epochs)
            
            best_val_acc = 0
            patience_counter = 0
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
            
            for epoch in range(num_epochs):
                # Train
                start_time = time.time()
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device,
                    mixup_alpha=args.mixup
                )
                
                # Validate
                val_loss, val_acc, val_preds, val_labels = validate(
                    model, val_loader, criterion, device
                )
                
                # Log learning rate
                if args.scheduler == 'onecycle':
                    history['lr'].append(optimizer.param_groups[0]['lr'])
                    scheduler.step()
                elif args.scheduler == 'cosine':
                    history['lr'].append(optimizer.param_groups[0]['lr'])
                    scheduler.step()
                else:  # plateau
                    history['lr'].append(optimizer.param_groups[0]['lr'])
                    scheduler.step(val_loss)
                
                # Save history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Log results
                logger.info(f"Stage {stage+1}, Epoch {epoch+1}/{num_epochs} | "
                           f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                           f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                           f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                           f"Time: {time.time()-start_time:.1f}s")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = os.path.join(
                        args.output_dir, f'best_model_fold{fold+1}_stage{stage+1}.pth'
                    )
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        best_model_path = os.path.join(args.output_dir, f'best_model_fold{fold+1}_stage2.pth')
        model.load_state_dict(torch.load(best_model_path))
        
        test_loss, test_acc, test_preds, test_labels = validate(
            model, test_loader, criterion, device
        )
        
        ensemble_test_preds.append(test_preds)
        ensemble_test_labels.append(test_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='weighted'
        )
        
        logger.info(f"Fold {fold+1} Results:")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        plot_results(
            history, test_preds, test_labels,
            ['Clean', 'LSB', 'DCT'],
            os.path.join(args.output_dir, f'fold{fold+1}')
        )
        
        if args.use_folds > 1:
            fold_results.append({
                'fold': fold+1,
                'test_acc': test_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        torch.save(
            model.state_dict(),
            os.path.join(
                args.output_dir,
                f'steg_detector_{args.model}_fold{fold+1}_acc{test_acc:.4f}.pth'
            )
        )
    
    if args.use_folds > 1:
        logger.info("\nCross-validation Summary:")
        accs = [r['test_acc'] for r in fold_results]
        f1s = [r['f1'] for r in fold_results]
        logger.info(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        logger.info(f"Mean F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        
        ensemble_preds = np.zeros_like(ensemble_test_preds[0])
        for preds in ensemble_test_preds:
            for i, pred in enumerate(preds):
                ensemble_preds[i] += pred
                
        ensemble_preds = np.argmax(ensemble_preds, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ensemble_test_labels[0], ensemble_preds, average='weighted'
        )
        
        logger.info("\nEnsemble Results:")
        logger.info(f"Ensemble Accuracy: {np.mean(ensemble_preds == ensemble_test_labels[0]):.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(ensemble_test_labels[0], ensemble_preds,
                                       target_names=['Clean', 'LSB', 'DCT']))

if __name__ == '__main__':
    main()