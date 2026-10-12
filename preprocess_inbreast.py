"""
INbreast Dataset Preprocessing for Tumor Segmentation
======================================================

Converts INbreast DICOM images and XML annotations to PNG images with tumor masks.

Input:
    - AllDICOMs/: DICOM mammography images
    - AllXML/: XML files with Mass/Calcification polygon annotations

Output:
    - segmentation_inbreast/
        - train/images/, train/masks/
        - val/images/, val/masks/
        - test/images/, test/masks/

Usage:
    python preprocess_inbreast.py --input-dir ../data/INbreast --output-dir ../data/segmentation_inbreast
"""

import os
import re
import plistlib
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import pydicom


def parse_point_string(point_str):
    """Parse point string like '(2462.929932, 995.062988)' to (x, y)."""
    match = re.match(r'\(([^,]+),\s*([^)]+)', point_str)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return (x, y)
    return None


def load_xml_annotations(xml_path):
    """
    Load annotations from INbreast XML file.
    
    Returns:
        list: List of ROI dicts with 'name', 'points', 'type'
    """
    with open(xml_path, 'rb') as f:
        plist = plistlib.load(f)
    
    rois = []
    
    if 'Images' in plist:
        for image_data in plist['Images']:
            if 'ROIs' in image_data:
                for roi in image_data['ROIs']:
                    name = roi.get('Name', '')
                    roi_type = roi.get('Type', 0)
                    points_px = roi.get('Point_px', [])
                    
                    # Parse points
                    points = []
                    for pt_str in points_px:
                        pt = parse_point_string(pt_str)
                        if pt:
                            points.append(pt)
                    
                    if points:
                        rois.append({
                            'name': name,
                            'type': roi_type,
                            'points': points,
                            'area': roi.get('Area', 0)
                        })
    
    return rois


def create_mask_from_contours(rois, image_shape, include_masses=True, include_calcifications=False):
    """
    Create binary mask from ROI contours.
    
    Args:
        rois: List of ROI dicts
        image_shape: (height, width) of output mask
        include_masses: Include Mass annotations
        include_calcifications: Include Calcification annotations
    
    Returns:
        numpy array: Binary mask
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for roi in rois:
        name = roi['name']
        points = roi['points']
        
        # Filter by type
        if name == 'Mass' and not include_masses:
            continue
        if name == 'Calcification' and not include_calcifications:
            continue
        
        # Skip single-point annotations (need at least 3 points for polygon)
        if len(points) < 3:
            continue
        
        # Convert to numpy array of integer points
        pts = np.array(points, dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(mask, [pts], 255)
    
    return mask


def load_dicom_image(dicom_path):
    """Load and normalize DICOM image."""
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)
    
    # Normalize to 0-255
    image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
    image = image.astype(np.uint8)
    
    return image


def preprocess_inbreast(
    input_dir,
    output_dir,
    image_size=512,
    include_masses=True,
    include_calcifications=False,
    test_size=0.15,
    val_size=0.15,
    random_seed=42
):
    """
    Preprocess INbreast dataset for segmentation.
    
    Args:
        input_dir: Path to INbreast directory
        output_dir: Output directory for processed data
        image_size: Target image size
        include_masses: Include Mass annotations in masks
        include_calcifications: Include Calcification annotations
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_seed: Random seed for splitting
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    dicom_dir = input_dir / 'AllDICOMs'
    xml_dir = input_dir / 'AllXML'
    
    print("=" * 60)
    print("INbreast Preprocessing")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Image size: {image_size}")
    print(f"Include masses: {include_masses}")
    print(f"Include calcifications: {include_calcifications}")
    
    # Get all XML files
    xml_files = list(xml_dir.glob('*.xml'))
    print(f"\nFound {len(xml_files)} XML annotation files")
    
    # Match DICOM files with XML files
    pairs = []
    
    for xml_path in tqdm(xml_files, desc="Matching files"):
        # XML filename is the image ID (e.g., 20586908.xml)
        image_id = xml_path.stem
        
        # Find corresponding DICOM
        dicom_matches = list(dicom_dir.glob(f'{image_id}_*.dcm'))
        
        if dicom_matches:
            dicom_path = dicom_matches[0]
            
            # Load annotations
            rois = load_xml_annotations(xml_path)
            
            # Check if we have relevant annotations
            has_mass = any(r['name'] == 'Mass' for r in rois)
            has_calc = any(r['name'] == 'Calcification' for r in rois)
            
            if (include_masses and has_mass) or (include_calcifications and has_calc):
                pairs.append({
                    'image_id': image_id,
                    'dicom_path': dicom_path,
                    'xml_path': xml_path,
                    'rois': rois,
                    'has_mass': has_mass,
                    'has_calc': has_calc
                })
    
    print(f"\nFound {len(pairs)} images with annotations")
    print(f"  - With masses: {sum(1 for p in pairs if p['has_mass'])}")
    print(f"  - With calcifications: {sum(1 for p in pairs if p['has_calc'])}")
    
    if len(pairs) == 0:
        print("ERROR: No valid pairs found!")
        return
    
    # Split into train/val/test
    train_pairs, test_pairs = train_test_split(
        pairs, test_size=test_size, random_state=random_seed
    )
    train_pairs, val_pairs = train_test_split(
        train_pairs, test_size=val_size / (1 - test_size), random_state=random_seed
    )
    
    print(f"\nSplit: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = [
        ('train', train_pairs),
        ('val', val_pairs),
        ('test', test_pairs)
    ]
    
    stats = {'total': 0, 'with_tumor': 0, 'tumor_pixels': []}
    
    for split_name, split_pairs in splits:
        print(f"\nProcessing {split_name} set...")
        
        for pair in tqdm(split_pairs, desc=f"  {split_name}"):
            try:
                # Load DICOM image
                image = load_dicom_image(pair['dicom_path'])
                original_shape = image.shape
                
                # Create mask
                mask = create_mask_from_contours(
                    pair['rois'],
                    original_shape,
                    include_masses=include_masses,
                    include_calcifications=include_calcifications
                )
                
                # Resize
                image_resized = cv2.resize(image, (image_size, image_size))
                mask_resized = cv2.resize(mask, (image_size, image_size), 
                                          interpolation=cv2.INTER_NEAREST)
                
                # Binarize mask
                mask_resized = (mask_resized > 127).astype(np.uint8) * 255
                
                # Apply CLAHE to image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image_enhanced = clahe.apply(image_resized)
                
                # Save
                image_path = output_dir / split_name / 'images' / f'{pair["image_id"]}.png'
                mask_path = output_dir / split_name / 'masks' / f'{pair["image_id"]}.png'
                
                cv2.imwrite(str(image_path), image_enhanced)
                cv2.imwrite(str(mask_path), mask_resized)
                
                # Stats
                stats['total'] += 1
                tumor_ratio = (mask_resized > 0).mean()
                if tumor_ratio > 0:
                    stats['with_tumor'] += 1
                    stats['tumor_pixels'].append(tumor_ratio)
                    
            except Exception as e:
                print(f"Error processing {pair['image_id']}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total processed: {stats['total']}")
    print(f"With tumor mask: {stats['with_tumor']}")
    if stats['tumor_pixels']:
        print(f"Avg tumor coverage: {np.mean(stats['tumor_pixels'])*100:.2f}%")
        print(f"Min tumor coverage: {np.min(stats['tumor_pixels'])*100:.2f}%")
        print(f"Max tumor coverage: {np.max(stats['tumor_pixels'])*100:.2f}%")
    
    # Count files per split
    for split in ['train', 'val', 'test']:
        img_count = len(list((output_dir / split / 'images').glob('*.png')))
        print(f"{split}: {img_count} images")


def main():
    parser = argparse.ArgumentParser(description='Preprocess INbreast dataset')
    parser.add_argument('--input-dir', type=str, 
                        default='../data/INbreast',
                        help='Path to INbreast directory')
    parser.add_argument('--output-dir', type=str,
                        default='../data/segmentation_inbreast',
                        help='Output directory')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Target image size')
    parser.add_argument('--include-calcifications', action='store_true',
                        help='Include calcification annotations')
    parser.add_argument('--masses-only', action='store_true', default=True,
                        help='Only include mass annotations (default)')
    
    args = parser.parse_args()
    
    preprocess_inbreast(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        include_masses=True,
        include_calcifications=args.include_calcifications
    )


if __name__ == '__main__':
    main()

