"""
Configuration Module - Inference System
====================================================

This module centralizes all configuration settings for the MammoPipe mammography
classification system. It provides a single source of truth for data paths,
model parameters, training hyperparameters, and inference settings.

Configuration Sections:
-----------------------
- data: Dataset paths, image size, batch settings, data loading parameters
- model: Architecture settings (num_classes, dropout, pretrained)
- training: Hyperparameters for training (learning rate, epochs, etc.)
- loss: Loss function settings (focal gamma, label smoothing)
- optimization: Performance settings (mixed precision, compilation)
- preprocessing: Image preprocessing options (CLAHE, filters)
- inference: Inference-specific settings (checkpoint path, output directory)

Usage:
------
    from config import config, verify_paths
    
    # Access configuration values
    batch_size = config['data']['batch_size']
    checkpoint = config['inference']['checkpoint_path']
    
    # Verify all paths exist before running
    if verify_paths():
        print("Ready to run!")
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Get project root directory (parent of src/)
# This ensures paths work regardless of where the script is run from
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Dataset root directory
# CBIS-DDSM dataset should be extracted here
DATA_ROOT = PROJECT_ROOT / "data" / "cbis-ddsm-breast-cancer-image-dataset"


# =============================================================================
# MAIN CONFIGURATION DICTIONARY
# =============================================================================

config = {
    # -------------------------------------------------------------------------
    # DATA CONFIGURATION
    # -------------------------------------------------------------------------
    # Settings for dataset loading and data pipeline
    'data': {
        'root_dir': str(DATA_ROOT),    # Path to CBIS-DDSM dataset
        'image_size': 256,              # Input image size (256x256 pixels)
        'batch_size': 32,               # Images per batch (reduce if OOM)
        'num_workers': 4,               # Parallel data loading workers
        'val_split': 0.2,               # Validation set fraction (20%)
        'prefetch_factor': 2,           # Batches to prefetch per worker
        'persistent_workers': True,     # Keep workers alive between epochs
    },
    
    # -------------------------------------------------------------------------
    # MODEL CONFIGURATION
    # -------------------------------------------------------------------------
    # Architecture and regularization settings
    'model': {
        'num_classes': 2,               # Binary: Benign (0) vs Malignant (1)
        'dropout': 0.3,                 # Dropout rate for regularization
        'pretrained': False,            # False for inference (weights loaded)
    },
    
    # -------------------------------------------------------------------------
    # TRAINING CONFIGURATION
    # -------------------------------------------------------------------------
    # Hyperparameters used during model training
    # (Not used during inference, but documented for reference)
    'training': {
        'epochs': 50,                   # Maximum training epochs
        'learning_rate': 1e-3,          # Initial learning rate
        'weight_decay': 0.01,           # L2 regularization strength
        'patience': 25,                 # Early stopping patience
        'gradient_clip': 1.0,           # Max gradient norm (prevents explosion)
        'warmup_epochs': 3,             # LR warmup period
        'use_cosine_annealing': True,   # Use cosine LR schedule
    },
    
    # -------------------------------------------------------------------------
    # LOSS FUNCTION CONFIGURATION
    # -------------------------------------------------------------------------
    # Settings for focal loss with label smoothing
    'loss': {
        'focal_gamma': 2.0,             # Focal loss focusing parameter
                                        # Higher gamma = more focus on hard examples
        'label_smoothing': 0.2,         # Prevents overconfident predictions
    },
    
    # -------------------------------------------------------------------------
    # OPTIMIZATION CONFIGURATION
    # -------------------------------------------------------------------------
    # Performance and memory optimization settings
    'optimization': {
        'mixed_precision': True,        # Use FP16 for faster inference
        'compile_model': False,         # PyTorch 2.0 model compilation
        'channels_last': False,         # Memory layout optimization
    },
    
    # -------------------------------------------------------------------------
    # PREPROCESSING CONFIGURATION
    # -------------------------------------------------------------------------
    # Image preprocessing pipeline settings
    # Note: CLAHE is always applied (not configurable here)
    'preprocessing': {
        'use_unsharp_mask': False,      # Edge enhancement filter
        'use_median_filter': False,     # Noise reduction filter
        'median_kernel_size': 3,        # Kernel size for median filter
    },
    
    # -------------------------------------------------------------------------
    # INFERENCE CONFIGURATION
    # -------------------------------------------------------------------------
    # Settings specific to model inference
    'inference': {
        # Path to best model checkpoint (V15 model, AUC 0.813)
        'checkpoint_path': str(PROJECT_ROOT / 'checkpoints' / 'best_resnet50_swinv2.ptrom'),
        'output_dir': str(PROJECT_ROOT / 'outputs'),  # Results output directory
        'save_predictions': True,        # Save predictions to CSV
        'save_visualizations': True,     # Save confusion matrix, ROC curve
    }
}


# =============================================================================
# PATH VERIFICATION
# =============================================================================

def verify_paths():
    """
    Verify that all required paths exist before running inference.
    
    This function checks:
        1. Project root directory
        2. Dataset directory and subdirectories (csv/, jpeg/)
        3. Model checkpoint file
    
    Returns:
        bool: True if all required paths exist, False otherwise
    
    Example:
        >>> from config import config, verify_paths
        >>> 
        >>> if not verify_paths():
        ...     print("Please check your paths in config.py")
        ...     sys.exit(1)
        >>> 
        >>> # Proceed with inference...
    
    Output:
        Prints detailed path verification status to console with emoji indicators.
    """
    print("Verifying paths...")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Dataset root: {DATA_ROOT}")
    print(f"  Dataset exists: {DATA_ROOT.exists()}")
    
    # Check dataset subdirectories if main directory exists
    if DATA_ROOT.exists():
        csv_dir = DATA_ROOT / 'csv'      # Contains metadata CSV files
        jpeg_dir = DATA_ROOT / 'jpeg'    # Contains mammogram images
        print(f"  CSV directory exists: {csv_dir.exists()}")
        print(f"  JPEG directory exists: {jpeg_dir.exists()}")
    
    # Check checkpoint file
    checkpoint_path = Path(config['inference']['checkpoint_path'])
    print(f"  Checkpoint exists: {checkpoint_path.exists()}")
    
    # Print checkpoint size if it exists (for verification)
    if checkpoint_path.exists():
        size_gb = checkpoint_path.stat().st_size / (1024**3)
        print(f"  Checkpoint size: {size_gb:.2f} GB")
    
    # Return True only if both dataset and checkpoint exist
    return DATA_ROOT.exists() and checkpoint_path.exists()


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == '__main__':
    """
    Run path verification when executed directly.
    
    Usage: python config.py
    """
    verify_paths()
