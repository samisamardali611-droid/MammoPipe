"""
Dataset Module for CBIS-DDSM Mammography Images
================================================

This module provides a PyTorch Dataset implementation for loading and preprocessing
mammography images from the CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset.

Dataset Overview:
-----------------
CBIS-DDSM is a curated subset of the Digital Database for Screening Mammography (DDSM),
containing mammography images with verified pathology labels.

Key Features:
- Binary classification: Benign vs Malignant
- Two lesion types: Mass and Calcification
- High-resolution mammogram images

Preprocessing Pipeline:
-----------------------
The dataset applies the following preprocessing steps:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) - Always applied
    2. Optional: Median filtering for noise reduction
    3. Optional: Unsharp masking for edge enhancement
    4. Resize to target size (default: 256x256)
    5. Normalize to [0, 1] range
    6. Convert to 3-channel tensor (for pretrained model compatibility)

Classes:
--------
- CBISDDSMDataset: Main dataset class with full preprocessing pipeline

Usage:
------
    from dataset import CBISDDSMDataset
    from config import config
    
    # Create test dataset
    dataset = CBISDDSMDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        config=config
    )
    
    # Get a sample
    sample = dataset[0]
    image = sample['image']      # Tensor (3, 256, 256)
    label = sample['label']      # Tensor (0 or 1)
    metadata = sample['metadata']  # Dict with patient info
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import os


class CBISDDSMDataset(Dataset):
    """
    PyTorch Dataset for CBIS-DDSM mammography images with CLAHE preprocessing.
    
    This dataset handles the complex CBIS-DDSM directory structure, resolves
    DICOM-style paths to actual image files, and applies a medical imaging
    preprocessing pipeline optimized for mammography.
    
    Directory Structure Expected:
    -----------------------------
        root_dir/
        ├── csv/
        │   ├── calc_case_description_train_set.csv
        │   ├── calc_case_description_test_set.csv
        │   ├── mass_case_description_train_set.csv
        │   └── mass_case_description_test_set.csv
        └── jpeg/
            └── [UID folders containing .jpg images]
    
    Preprocessing Pipeline:
    -----------------------
        1. Load grayscale image
        2. Apply CLAHE for contrast enhancement (always on)
        3. Optional median filter for noise reduction
        4. Optional unsharp mask for edge enhancement
        5. Resize to target size
        6. Normalize to [0, 1]
        7. Convert to 3-channel tensor
    
    Attributes:
        root_dir (Path): Root directory of the CBIS-DDSM dataset
        split (str): Data split - 'train', 'test', or 'val' (val maps to test)
        transform: Optional additional transforms to apply
        config (dict): Configuration dictionary with preprocessing settings
        image_size (int): Target image size (default: 256)
        samples (DataFrame): Loaded metadata with resolved image paths
        uid_to_path (dict): Mapping from DICOM UIDs to actual file paths
    
    Args:
        root_dir (str): Path to CBIS-DDSM dataset root directory
        split (str): Which split to load ('train', 'test', or 'val')
        transform: Optional torchvision transforms
        config (dict): Optional configuration dictionary
    
    Example:
        >>> dataset = CBISDDSMDataset(
        ...     root_dir='/path/to/cbis-ddsm',
        ...     split='test',
        ...     config={'data': {'image_size': 256}}
        ... )
        >>> 
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Image shape: {sample['image'].shape}")
        >>> print(f"Label: {sample['label']} ({sample['metadata']['pathology']})")
    """
    
    def __init__(self, root_dir, split='test', transform=None, config=None):
        """
        Initialize the CBIS-DDSM dataset.
        
        Args:
            root_dir (str): Root directory containing 'csv/' and 'jpeg/' subdirectories
            split (str): Data split to load:
                        - 'train': Training set
                        - 'test': Test set
                        - 'val': Validation set (maps to test CSV in CBIS-DDSM)
            transform: Optional torchvision transforms for data augmentation
            config (dict): Configuration dictionary. If None, uses defaults.
                          Expected keys: config['data']['image_size']
        """
        self.root_dir = Path(root_dir).absolute()
        self.split = split
        self.transform = transform
        self.config = config or {}
        
        # Get image size from config or use default
        self.image_size = self.config.get('data', {}).get('image_size', 256)
        
        print(f"📂 Loading dataset from: {self.root_dir}")
        print(f"   Split: {split}")
        
        # Load metadata and build file path mappings
        self.samples = self._load_metadata()
        print(f"✓ Loaded {len(self.samples)} samples")
    
    def _load_metadata(self):
        """
        Load CBIS-DDSM metadata from CSV files.
        
        This method:
            1. Loads both calcification and mass CSV files for the split
            2. Builds a UID-to-filepath lookup table
            3. Resolves DICOM-style paths to actual image files
            4. Filters out samples with missing images
        
        Returns:
            DataFrame: Filtered metadata with 'resolved_path' column added
        
        Raises:
            ValueError: If no CSV files found for the specified split
        """
        csv_dir = self.root_dir / 'csv'
        
        # CBIS-DDSM only has train/test CSVs, so map 'val' to 'test'
        csv_split = 'test' if self.split == 'val' else self.split
        
        dfs = []
        
        # Load both calcification and mass lesion types
        for lesion_type in ['calc', 'mass']:
            csv_file = csv_dir / f'{lesion_type}_case_description_{csv_split}_set.csv'
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df['lesion_type'] = lesion_type  # Track lesion type for analysis
                dfs.append(df)
                print(f"  ✓ Loaded {len(df)} {lesion_type} samples")
        
        if not dfs:
            raise ValueError(f"No CSV files found for split={csv_split} in {csv_dir}")
        
        # Combine calcification and mass samples
        metadata = pd.concat(dfs, ignore_index=True)
        
        # Build UID lookup table for resolving paths
        self.uid_to_path = self._build_uid_lookup()
        
        # Resolve DICOM-style paths to actual file paths
        metadata['resolved_path'] = metadata['image file path'].apply(self._resolve_path)
        
        # Filter out samples where images couldn't be found
        valid_count = metadata['resolved_path'].notna().sum()
        metadata = metadata[metadata['resolved_path'].notna()].reset_index(drop=True)
        print(f"  ✓ Resolved {valid_count}/{len(metadata)} image paths")
        
        return metadata
    
    def _build_uid_lookup(self):
        """
        Build a lookup table mapping DICOM UIDs to actual file paths.
        
        CBIS-DDSM uses DICOM UIDs (e.g., '1.3.6.1.4.1.9590...') as folder names.
        This method walks the jpeg/ directory to build a mapping from these
        UIDs to the actual image file paths.
        
        Returns:
            dict: Mapping from UID string to full file path
        """
        uid_lookup = {}
        jpeg_dir = self.root_dir / 'jpeg'
        
        if not jpeg_dir.exists():
            print(f"⚠ Warning: JPEG directory not found at {jpeg_dir}")
            return uid_lookup
        
        # Walk through all subdirectories to find image files
        for root, dirs, files in os.walk(jpeg_dir):
            for file in files:
                # Only process image files
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    parts = Path(root).parts
                    
                    # Look for DICOM UID pattern in path
                    for part in parts:
                        if part.startswith('1.3.6.1.4.1.9590'):  # DICOM UID prefix
                            uid_lookup[part] = os.path.join(root, file)
                            break
        
        print(f"  ✓ Built lookup for {len(uid_lookup)} UID folders")
        return uid_lookup
    
    def _resolve_path(self, dicom_path):
        """
        Resolve a DICOM-style path to an actual image file path.
        
        CBIS-DDSM CSV files contain paths like:
            'Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.../1.3.6.1.4.1.9590...'
        
        This method extracts the UID and looks it up in the uid_to_path mapping.
        
        Args:
            dicom_path (str): DICOM-style path from CSV file
        
        Returns:
            str or None: Actual file path if found, None otherwise
        """
        if pd.isna(dicom_path):
            return None
        
        path_str = str(dicom_path)
        parts = path_str.split('/')
        
        # The second-to-last part typically contains the UID we need
        if len(parts) >= 2:
            last_uid = parts[-2]
            if last_uid.startswith('1.3.6.1.4.1.9590'):
                return self.uid_to_path.get(last_uid)
        
        return None
    
    def _apply_clahe(self, image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE is essential for mammography because:
            - Mammograms often have low contrast
            - Different tissue types need local contrast enhancement
            - It reveals subtle calcifications and masses
        
        How CLAHE Works:
            1. Divides image into 8x8 tiles
            2. Applies histogram equalization to each tile
            3. Clips histograms to limit contrast (clipLimit=2.0)
            4. Interpolates tile boundaries for smooth result
        
        Args:
            image (np.ndarray): Grayscale image (0-255)
        
        Returns:
            np.ndarray: Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _apply_unsharp_mask(self, image):
        """
        Apply unsharp masking for edge enhancement.
        
        Unsharp masking enhances edges by:
            1. Creating a blurred version of the image
            2. Subtracting the blur to get edges
            3. Adding edges back to original with amplification
        
        Formula: output = original * 1.5 - gaussian * 0.5
        
        This can help highlight:
            - Mass boundaries
            - Calcification clusters
            - Architectural distortions
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Edge-enhanced image
        
        Note:
            Disabled by default (config['preprocessing']['use_unsharp_mask'])
            Can introduce noise if image quality is poor
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return unsharp
    
    def _apply_median_filter(self, image, kernel_size=3):
        """
        Apply median filter for noise reduction.
        
        Median filtering removes salt-and-pepper noise while preserving edges.
        This is useful for mammography to:
            - Remove sensor noise
            - Smooth minor artifacts
            - Prepare image for further processing
        
        Args:
            image (np.ndarray): Input image
            kernel_size (int): Filter kernel size (must be odd: 3, 5, 7...)
        
        Returns:
            np.ndarray: Denoised image
        
        Note:
            Disabled by default (config['preprocessing']['use_median_filter'])
        """
        return cv2.medianBlur(image, kernel_size)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        This method:
            1. Loads the image from disk
            2. Applies the full preprocessing pipeline
            3. Converts to tensor format
            4. Extracts label from pathology column
            5. Collects metadata for analysis
        
        Args:
            idx (int): Sample index
        
        Returns:
            dict: Dictionary containing:
                - 'image': Tensor of shape (3, H, W), normalized to [0, 1]
                - 'label': Tensor with value 0 (benign) or 1 (malignant)
                - 'metadata': Dict with patient_id, image_view, laterality, etc.
        
        Raises:
            ValueError: If image file cannot be loaded
        """
        row = self.samples.iloc[idx]
        
        # ----- Step 1: Load grayscale image -----
        image = cv2.imread(row['resolved_path'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load: {row['resolved_path']}")
        
        # ----- Step 2: Apply preprocessing pipeline -----
        
        # 2a. CLAHE (always applied for contrast enhancement)
        image = self._apply_clahe(image)
        
        # 2b. Optional: Median filter for noise reduction
        if self.config.get('preprocessing', {}).get('use_median_filter', False):
            kernel_size = self.config.get('preprocessing', {}).get('median_kernel_size', 3)
            image = self._apply_median_filter(image, kernel_size)
        
        # 2c. Optional: Unsharp masking for edge enhancement
        if self.config.get('preprocessing', {}).get('use_unsharp_mask', False):
            image = self._apply_unsharp_mask(image)
        
        # ----- Step 3: Resize to target size -----
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # ----- Step 4: Normalize to [0, 1] range -----
        image = image.astype(np.float32) / 255.0
        
        # ----- Step 5: Convert to 3-channel tensor -----
        # Models pretrained on ImageNet expect 3-channel input
        # Repeat grayscale to RGB: (H, W) -> (3, H, W)
        image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        
        # ----- Step 6: Apply optional augmentations -----
        if self.transform:
            image = self.transform(image)
        
        # ----- Step 7: Extract label from pathology -----
        # Labels: 0 = Benign, 1 = Malignant
        pathology = str(row.get('pathology', 'BENIGN')).upper()
        label = 1 if 'MALIGNANT' in pathology else 0
        
        # ----- Step 8: Collect metadata for analysis -----
        metadata = {
            'patient_id': row.get('patient_id', 'unknown'),
            'image_view': row.get('image view', 'unknown'),      # CC or MLO
            'laterality': row.get('left or right breast', 'unknown'),
            'lesion_type': row.get('lesion_type', 'unknown'),    # calc or mass
            'pathology': pathology,
            'image_path': row['resolved_path']
        }
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': metadata
        }
    
    def get_class_weights(self):
        """
        Calculate balanced class weights for handling class imbalance.
        
        Medical datasets often have class imbalance (more benign than malignant).
        This method computes weights inversely proportional to class frequency,
        which can be used with weighted loss functions or samplers.
        
        Formula: weight[c] = total_samples / (2 * count[c])
        
        Returns:
            Tensor: Class weights of shape (2,) for [benign, malignant]
        
        Example:
            >>> weights = dataset.get_class_weights()
            >>> criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        """
        labels = self.samples['pathology'].apply(
            lambda x: 1 if 'MALIGNANT' in str(x).upper() else 0
        )
        counts = labels.value_counts()
        total = len(labels)
        
        # Inverse frequency weighting
        weights = torch.tensor([
            total / (2 * counts.get(0, 1)),  # Benign weight
            total / (2 * counts.get(1, 1)),  # Malignant weight
        ], dtype=torch.float32)
        
        return weights
    
    def get_class_distribution(self):
        """
        Get class distribution statistics.
        
        Returns:
            dict: Dictionary containing:
                - 'benign': Count of benign samples
                - 'malignant': Count of malignant samples
                - 'total': Total sample count
                - 'ratio': String representation (e.g., "200:150")
        
        Example:
            >>> dist = dataset.get_class_distribution()
            >>> print(f"Benign: {dist['benign']}, Malignant: {dist['malignant']}")
        """
        labels = self.samples['pathology'].apply(
            lambda x: 1 if 'MALIGNANT' in str(x).upper() else 0
        )
        counts = labels.value_counts()
        
        return {
            'benign': int(counts.get(0, 0)),
            'malignant': int(counts.get(1, 0)),
            'total': len(labels),
            'ratio': f"{counts.get(0, 0)}:{counts.get(1, 0)}"
        }


# =============================================================================
# TESTING / DEBUGGING
# =============================================================================

if __name__ == '__main__':
    """
    Test script to verify dataset loading and preprocessing.
    
    Run with: python dataset.py
    
    Expected output:
        - Dataset loading summary
        - Class distribution
        - Sample image details
    """
    from config import config
    
    print("Testing dataset loading...")
    
    # Create test dataset
    dataset = CBISDDSMDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        config=config
    )
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    
    # Print class distribution
    dist = dataset.get_class_distribution()
    print(f"\n Class distribution:")
    print(f"  Benign: {dist['benign']}")
    print(f"  Malignant: {dist['malignant']}")
    print(f"  Ratio: {dist['ratio']}")
    
    # Test loading one sample
    print(f"\n Testing sample loading...")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label'].item()} ({'Malignant' if sample['label'].item() == 1 else 'Benign'})")
    print(f"  Patient ID: {sample['metadata']['patient_id']}")
    print(f"  View: {sample['metadata']['image_view']}")
