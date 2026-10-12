"""
Convert Segmentation Dataset to Parquet/Arrow Format
=====================================================

Converts preprocessed PNG images and masks to Parquet format for faster
data loading during training. Arrow/Parquet provides:
- Faster I/O (columnar storage, compression)
- Memory-mapped reading
- Better compatibility with distributed training

Output Structure:
-----------------
    data/segmentation_parquet/
    ├── train.parquet
    ├── val.parquet
    └── test.parquet

Each Parquet file contains columns:
- image: numpy array as bytes (512x512 uint8)
- mask: numpy array as bytes (512x512 uint8)
- filename: original filename
- pathology: benign/malignant label

Usage:
------
    python convert_to_parquet.py --input-dir ../data/segmentation --output-dir ../data/segmentation_parquet
"""

import os
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import argparse


def load_image_as_bytes(image_path):
    """Load image and convert to bytes for storage."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.tobytes()


def convert_split_to_parquet(input_dir, output_path, split_name):
    """
    Convert a single split (train/val/test) to Parquet format.
    
    Args:
        input_dir (Path): Directory containing images/ and masks/ subdirs
        output_path (Path): Output Parquet file path
        split_name (str): Name of the split for logging
    
    Returns:
        int: Number of samples converted
    """
    images_dir = input_dir / 'images'
    masks_dir = input_dir / 'masks'
    
    if not images_dir.exists():
        print(f"  Skipping {split_name}: images directory not found")
        return 0
    
    # Collect all samples
    image_files = sorted(images_dir.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"  Skipping {split_name}: no images found")
        return 0
    
    print(f"  Converting {split_name}: {len(image_files)} samples...")
    
    # Prepare data lists
    images_data = []
    masks_data = []
    filenames = []
    pathologies = []
    image_shapes = []
    
    for img_path in tqdm(image_files, desc=f"    {split_name}"):
        mask_path = masks_dir / img_path.name
        
        if not mask_path.exists():
            continue
        
        # Load image and mask
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        # Store as bytes (compressed numpy)
        images_data.append(img.tobytes())
        masks_data.append(mask.tobytes())
        filenames.append(img_path.name)
        image_shapes.append(f"{img.shape[0]}x{img.shape[1]}")
        
        # Extract pathology from filename (e.g., mass_train_P_00001_1.png)
        # Filename format: {type}_{split}_{patient}_{abnorm}.png
        fname = img_path.stem
        if 'MALIGNANT' in fname.upper():
            pathologies.append('MALIGNANT')
        else:
            pathologies.append('BENIGN')
    
    if len(images_data) == 0:
        print(f"  Skipping {split_name}: no valid samples")
        return 0
    
    # Create PyArrow table
    table = pa.table({
        'image': pa.array(images_data, type=pa.binary()),
        'mask': pa.array(masks_data, type=pa.binary()),
        'filename': pa.array(filenames, type=pa.string()),
        'pathology': pa.array(pathologies, type=pa.string()),
        'shape': pa.array(image_shapes, type=pa.string()),
    })
    
    # Write to Parquet with compression
    pq.write_table(
        table, 
        output_path,
        compression='zstd',  # Fast compression with good ratio
        use_dictionary=True,
    )
    
    # Print stats
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {output_path.name} ({len(images_data)} samples, {file_size_mb:.1f} MB)")
    
    return len(images_data)


def convert_dataset_to_parquet(input_dir, output_dir):
    """
    Convert entire segmentation dataset to Parquet format.
    
    Args:
        input_dir (str): Path to preprocessed segmentation data
        output_dir (str): Path to save Parquet files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print(f"Converting segmentation dataset to Parquet")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    
    # Convert each split
    for split in ['train', 'val', 'test']:
        split_input = input_dir / split
        split_output = output_dir / f'{split}.parquet'
        
        if split_input.exists():
            count = convert_split_to_parquet(split_input, split_output, split)
            total_samples += count
    
    print(f"\nConversion complete!")
    print(f"  Total samples: {total_samples}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert segmentation data to Parquet')
    parser.add_argument('--input-dir', type=str, default='../data/segmentation',
                        help='Input directory with preprocessed PNG data')
    parser.add_argument('--output-dir', type=str, default='../data/segmentation_parquet',
                        help='Output directory for Parquet files')
    
    args = parser.parse_args()
    convert_dataset_to_parquet(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

