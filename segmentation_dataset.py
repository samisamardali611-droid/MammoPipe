"""
Segmentation Dataset for CBIS-DDSM Mammography
===============================================

PyTorch Dataset implementation for loading preprocessed mammography images
and their corresponding segmentation masks for tumor segmentation training.

This dataset assumes data has been preprocessed using `preprocess_segmentation.py`.

Expected Directory Structure:
-----------------------------
    data/segmentation/
    ├── train/
    │   ├── images/
    │   │   ├── mass_train_001.png
    │   │   └── ...
    │   └── masks/
    │       ├── mass_train_001.png
    │       └── ...
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

Usage:
------
    from segmentation_dataset import SegmentationDataset
    
    dataset = SegmentationDataset(
        root_dir='../data/segmentation',
        split='train',
        augment=True
    )
    
    sample = dataset[0]
    image = sample['image']  # (1, 512, 512) tensor
    mask = sample['mask']    # (1, 512, 512) tensor

Author: MammoPipe Project
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for mammography segmentation.
    
    Loads preprocessed images and masks from directory structure created by
    the preprocessing script. Supports data augmentation for training.
    
    Augmentation Pipeline (when enabled):
    ------------------------------------
    - Horizontal flip (50% probability)
    - Vertical flip (50% probability)
    - Random rotation (±15 degrees)
    - Random brightness/contrast adjustment
    - Random Gaussian noise
    
    Attributes:
        root_dir (Path): Root directory of preprocessed data
        split (str): Data split ('train', 'val', 'test')
        augment (bool): Whether to apply data augmentation
        image_size (int): Expected image size (for verification)
        samples (list): List of (image_path, mask_path) tuples
    
    Args:
        root_dir (str): Path to preprocessed segmentation data
        split (str): Which split to load ('train', 'val', 'test')
        augment (bool): Enable data augmentation (default: False)
        image_size (int): Expected image size (default: 512)
    """
    
    def __init__(self, root_dir, split='train', augment=False, image_size=512):
        """
        Initialize the segmentation dataset.
        
        Args:
            root_dir (str): Root directory containing train/val/test subdirs
            split (str): Data split to load
            augment (bool): Whether to apply augmentation
            image_size (int): Expected image/mask size
        """
        self.root_dir = Path(root_dir).absolute()
        self.split = split
        self.augment = augment and split == 'train'  # Only augment training data
        self.image_size = image_size
        
        # Build sample list
        self.samples = self._load_samples()
        
        print(f"📂 SegmentationDataset initialized:")
        print(f"   Root: {self.root_dir}")
        print(f"   Split: {split}")
        print(f"   Samples: {len(self.samples)}")
        print(f"   Augmentation: {self.augment}")
    
    def _load_samples(self):
        """
        Load list of image-mask pairs from directory.
        
        Returns:
            list: List of (image_path, mask_path) tuples
        """
        split_dir = self.root_dir / self.split
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
        
        samples = []
        
        # Find all images and match with masks
        for img_file in sorted(images_dir.glob('*.png')):
            mask_file = masks_dir / img_file.name
            
            if mask_file.exists():
                samples.append((str(img_file), str(mask_file)))
        
        if len(samples) == 0:
            print(f"⚠ Warning: No samples found in {split_dir}")
        
        return samples
    
    def _apply_augmentation(self, image, mask):
        """
        Apply data augmentation to image and mask.
        
        Augmentations are applied identically to both image and mask
        to maintain alignment.
        
        Args:
            image (np.ndarray): Grayscale image
            mask (np.ndarray): Binary mask
        
        Returns:
            tuple: Augmented (image, mask)
        """
        # Horizontal flip (50% chance)
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Vertical flip (50% chance)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Random rotation (±15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, 
                                  flags=cv2.INTER_NEAREST)
        
        # Random brightness adjustment (image only)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment (image only)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Random Gaussian noise (image only)
        if random.random() > 0.3:
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return image, mask
    
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
        
        Returns:
            dict: Dictionary containing:
                - 'image': Tensor (1, H, W) normalized to [0, 1]
                - 'mask': Tensor (1, H, W) binary (0 or 1)
                - 'image_path': Original image path
                - 'mask_path': Original mask path
        """
        image_path, mask_path = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors (add channel dimension)
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'image_path': image_path,
            'mask_path': mask_path
        }
    
    def get_statistics(self):
        """
        Compute dataset statistics.
        
        Returns:
            dict: Dictionary with statistics:
                - total_samples: Number of samples
                - positive_pixels: Average % of mask pixels that are positive
                - image_size: Image dimensions
        """
        total_positive = 0
        total_pixels = 0
        
        for img_path, mask_path in self.samples[:min(100, len(self.samples))]:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                total_positive += (mask > 127).sum()
                total_pixels += mask.size
        
        return {
            'total_samples': len(self.samples),
            'positive_pixel_ratio': total_positive / max(total_pixels, 1),
            'image_size': self.image_size
        }


def create_dataloaders(root_dir, batch_size=8, num_workers=4, image_size=512):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir (str): Path to preprocessed segmentation data
        batch_size (int): Batch size (default: 8)
        num_workers (int): Number of data loading workers (default: 4)
        image_size (int): Expected image size (default: 512)
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoader objects
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = SegmentationDataset(
                root_dir=root_dir,
                split=split,
                augment=(split == 'train'),
                image_size=image_size
            )
            
            if len(dataset) > 0:
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=(split == 'train')
                )
            else:
                dataloaders[split] = None
                print(f"⚠ Empty {split} dataset, skipping dataloader creation")
                
        except FileNotFoundError as e:
            print(f"⚠ Could not load {split} split: {e}")
            dataloaders[split] = None
    
    return dataloaders


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    """Test dataset loading."""
    import sys
    
    print("Testing SegmentationDataset...")
    
    # Default path - adjust as needed
    root_dir = '../data/segmentation'
    
    if not Path(root_dir).exists():
        print(f"⚠ Test data directory not found: {root_dir}")
        print("  Run preprocess_segmentation.py first to create the dataset.")
        sys.exit(1)
    
    # Test dataset creation
    try:
        dataset = SegmentationDataset(
            root_dir=root_dir,
            split='train',
            augment=True
        )
        
        if len(dataset) > 0:
            # Test sample loading
            sample = dataset[0]
            print(f"\n✓ Sample loaded successfully")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Mask shape: {sample['mask'].shape}")
            print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            print(f"  Mask unique values: {torch.unique(sample['mask'])}")
            
            # Test statistics
            stats = dataset.get_statistics()
            print(f"\n📊 Dataset statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Positive pixel ratio: {stats['positive_pixel_ratio']:.4f}")
            
            # Test dataloader
            dataloaders = create_dataloaders(root_dir, batch_size=4, num_workers=0)
            
            if dataloaders['train']:
                batch = next(iter(dataloaders['train']))
                print(f"\n✓ Dataloader working")
                print(f"  Batch image shape: {batch['image'].shape}")
                print(f"  Batch mask shape: {batch['mask'].shape}")
        else:
            print("⚠ Dataset is empty")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    print("\n✓ All tests passed!")

