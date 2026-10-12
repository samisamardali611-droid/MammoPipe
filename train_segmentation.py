"""
Training Script for BCDU-Net Segmentation
==========================================

This script trains the BCDU-Net model for mammography tumor segmentation
on the preprocessed CBIS-DDSM dataset.

Features:
---------
- Dice + BCE combined loss function
- Data augmentation pipeline
- Learning rate scheduling (cosine annealing)
- Checkpoint saving (best model + periodic)
- Comprehensive metrics logging (Dice, IoU, accuracy)
- Mixed precision training support
- Early stopping
- TensorBoard logging (optional)

Usage:
------
    python train_segmentation.py \\
        --data-dir ../data/segmentation \\
        --epochs 100 \\
        --batch-size 8 \\
        --lr 1e-4 \\
        --save-dir ../checkpoints/segmentation

Author: MammoPipe Project
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Local imports
from bcdu_net import BCDUNet, BCDUNetLite, DiceBCELoss, DiceLoss
from segmentation_dataset import SegmentationDataset, create_dataloaders

# Try to import Parquet dataset (optional)
try:
    from segmentation_dataset_parquet import ParquetSegmentationDataset, create_parquet_dataloaders
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class SegmentationTrainer:
    """
    Trainer class for BCDU-Net segmentation model.
    
    Handles the complete training pipeline including:
    - Model initialization and loading
    - Training loop with validation
    - Loss computation and optimization
    - Metrics calculation (Dice, IoU, accuracy)
    - Checkpoint saving and loading
    - Learning rate scheduling
    - Early stopping
    
    Attributes:
        model (nn.Module): BCDU-Net model
        device (torch.device): Training device (CPU/GPU)
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler: Learning rate scheduler
        scaler (GradScaler): For mixed precision training
        config (dict): Training configuration
    """
    
    def __init__(self, config):
        """
        Initialize the trainer.
        
        Args:
            config (dict): Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing SegmentationTrainer")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Initialize model
        self._init_model()
        
        # Initialize loss function
        self.criterion = DiceBCELoss(
            dice_weight=config.get('dice_weight', 0.5),
            bce_weight=config.get('bce_weight', 0.5)
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['lr'] * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Training state
        self.start_epoch = 0
        self.best_dice = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'lr': []
        }
        
        # Create output directories
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)
        
        print(f"   Save directory: {self.save_dir}")
    
    def _init_model(self):
        """Initialize the BCDU-Net model."""
        model_type = self.config.get('model_type', 'bcdu_net')
        
        if model_type == 'bcdu_net_lite':
            self.model = BCDUNetLite(
                in_channels=1,
                out_channels=1
            )
        else:
            self.model = BCDUNet(
                in_channels=1,
                out_channels=1,
                base_features=self.config.get('base_features', 64),
                use_bconvlstm=self.config.get('use_bconvlstm', True)
            )
        
        self.model = self.model.to(self.device)
        
        # Load pretrained weights if specified
        if self.config.get('pretrained_path'):
            self._load_checkpoint(self.config['pretrained_path'])
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_dice = checkpoint.get('best_dice', 0.0)
            
            if 'optimizer_state_dict' in checkpoint and self.config.get('resume', False):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"   Loaded from epoch {self.start_epoch - 1}, best dice: {self.best_dice:.4f}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"   Loaded model weights")
    
    def _save_checkpoint(self, epoch, dice_score, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'config': self.config,
            'history': self.history
        }
        
        # Save periodic checkpoint
        if epoch % self.config.get('save_every', 10) == 0:
            path = self.save_dir / 'checkpoints' / f'bcdu_seg_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
            print(f"   💾 Saved checkpoint: {path.name}")
        
        # Save best model
        if is_best:
            path = self.save_dir / 'best_bcdu_segmentation.pth'
            torch.save(checkpoint, path)
            print(f"   🏆 Saved best model: {path.name} (Dice: {dice_score:.4f})")
        
        # Save latest model
        path = self.save_dir / 'latest_bcdu_segmentation.pth'
        torch.save(checkpoint, path)
    
    def _compute_metrics(self, pred, target, threshold=0.5):
        """
        Compute segmentation metrics.
        
        Args:
            pred (Tensor): Predicted logits
            target (Tensor): Ground truth masks
            threshold (float): Threshold for binary prediction
        
        Returns:
            dict: Dictionary with dice, iou, accuracy metrics
        """
        pred_prob = torch.sigmoid(pred)
        pred_binary = (pred_prob > threshold).float()
        
        # Flatten for computation
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        
        # Dice coefficient
        dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
        
        # IoU (Intersection over Union)
        iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
        
        # Pixel accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'accuracy': accuracy.item()
        }
    
    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch (int): Current epoch number
        
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]} [Train]')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = self._compute_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_dice += metrics['dice']
            total_iou += metrics['iou']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice"]:.4f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
            'iou': total_iou / num_batches
        }
    
    @torch.no_grad()
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation dataloader
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            metrics = self._compute_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_dice += metrics['dice']
            total_iou += metrics['iou']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
            'iou': total_iou / num_batches
        }
    
    def train(self, train_loader, val_loader=None):
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
        
        Returns:
            dict: Training history
        """
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['lr']}")
        print(f"   Mixed precision: {self.config.get('mixed_precision', True)}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = {'loss': 0, 'dice': 0, 'iou': 0}
            if val_loader:
                val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['lr'].append(current_lr)
            
            # Check for improvement
            is_best = False
            if val_loader and val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                is_best = True
                self.patience_counter = 0
            elif val_loader:
                self.patience_counter += 1
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics['dice'], is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n📊 Epoch {epoch + 1}/{self.config['epochs']} ({epoch_time:.1f}s)")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
            if val_loader:
                print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
            print(f"   LR: {current_lr:.2e}, Best Dice: {self.best_dice:.4f}")
            
            # Early stopping
            patience = self.config.get('patience', 20)
            if self.patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered (patience={patience})")
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training complete!")
        print(f"   Total time: {total_time / 3600:.2f} hours")
        print(f"   Best Dice: {self.best_dice:.4f}")
        
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"   History saved to: {history_path}")
        
        return self.history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train BCDU-Net for mammography segmentation')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='../data/segmentation',
                        help='Path to preprocessed segmentation data')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size (default: 512)')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='bcdu_net',
                        choices=['bcdu_net', 'bcdu_net_lite'],
                        help='Model architecture (default: bcdu_net)')
    parser.add_argument('--base-features', type=int, default=64,
                        help='Base feature channels (default: 64)')
    parser.add_argument('--use-bconvlstm', action='store_true', default=True,
                        help='Use BConvLSTM in decoder (default: True)')
    parser.add_argument('--no-bconvlstm', action='store_false', dest='use_bconvlstm',
                        help='Disable BConvLSTM in decoder')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    
    # Loss arguments
    parser.add_argument('--dice-weight', type=float, default=0.5,
                        help='Weight for Dice loss (default: 0.5)')
    parser.add_argument('--bce-weight', type=float, default=0.5,
                        help='Weight for BCE loss (default: 0.5)')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='../checkpoints/segmentation',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    # Other arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use-parquet', action='store_true',
                        help='Use Parquet format for faster data loading')
    parser.add_argument('--cache-in-memory', action='store_true',
                        help='Cache entire dataset in RAM (requires ~10GB+ RAM)')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'data_dir': args.data_dir,
        'image_size': args.image_size,
        'model_type': args.model_type,
        'base_features': args.base_features,
        'use_bconvlstm': args.use_bconvlstm,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'gradient_clip': args.gradient_clip,
        'patience': args.patience,
        'dice_weight': args.dice_weight,
        'bce_weight': args.bce_weight,
        'save_dir': args.save_dir,
        'save_every': args.save_every,
        'num_workers': args.num_workers,
        'mixed_precision': not args.no_mixed_precision,
        'pretrained_path': args.resume,
        'resume': args.resume is not None,
        'use_parquet': args.use_parquet,
        'cache_in_memory': args.cache_in_memory
    }
    
    # Print configuration
    print("=" * 60)
    print("BCDU-Net Segmentation Training")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create dataloaders
    print("\n Loading datasets...")
    
    if config['use_parquet']:
        if not PARQUET_AVAILABLE:
            print(" Error: Parquet support not available!")
            print("   Install pyarrow: pip install pyarrow")
            sys.exit(1)
        
        print("   Using Parquet format (fast loading)")
        dataloaders = create_parquet_dataloaders(
            root_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size'],
            cache_in_memory=config['cache_in_memory']
        )
    else:
        print("   Using PNG format")
        dataloaders = create_dataloaders(
            root_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size']
        )
    
    if dataloaders['train'] is None:
        print(" Error: No training data found!")
        if config['use_parquet']:
            print(f"   Run convert_to_parquet.py first to create Parquet files.")
        else:
            print(f"   Run preprocess_segmentation.py first to create the dataset.")
        sys.exit(1)
    
    # Initialize trainer
    trainer = SegmentationTrainer(config)
    
    # Start training
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('val')
    )


if __name__ == '__main__':
    main()

