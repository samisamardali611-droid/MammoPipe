"""
Multi-Task Inference Pipeline: Classification + Segmentation
=============================================================

Combined inference using:
- Classification: ResNet-101 + SwinV2 (benign vs malignant)
- Segmentation: BCDU-Net (tumor mask)

Usage:
------
    python inference_multitask.py --image path/to/mammogram.png
    
    # Or batch inference
    python inference_multitask.py --image-dir path/to/images/ --output-dir results/
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import ImprovedHybridV15, load_model as load_classifier
from bcdu_net import BCDUNet, BCDUNetLite


class MultiTaskInference:
    """
    Multi-task inference pipeline for mammography analysis.
    
    Performs both:
    - Classification: Benign vs Malignant
    - Segmentation: Tumor region mask
    """
    
    def __init__(
        self,
        classifier_checkpoint,
        segmentation_checkpoint,
        device='cuda'
    ):
        """
        Initialize multi-task inference.
        
        Args:
            classifier_checkpoint: Path to classification model checkpoint
            segmentation_checkpoint: Path to segmentation model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("=" * 60)
        print("Multi-Task Mammography Inference")
        print("=" * 60)
        print(f"Device: {self.device}")
        
        # Load models
        self.classifier = self._load_classifier(classifier_checkpoint)
        self.segmentation = self._load_segmentation(segmentation_checkpoint)
        
        print("=" * 60)
        print("Ready for inference!")
        print("=" * 60)
    
    def _load_classifier(self, checkpoint_path):
        """Load classification model."""
        print(f"\n📦 Loading classifier: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        model = ImprovedHybridV15(
            num_classes=2,
            dropout=0.3,
            pretrained=False
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_auc' in checkpoint:
                print(f"   AUC: {checkpoint['best_auc']:.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        print("   ✓ Classifier loaded")
        
        return model
    
    def _load_segmentation(self, checkpoint_path):
        """Load segmentation model."""
        print(f"\n📦 Loading segmentation: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        config = checkpoint.get('config', {})
        model_type = config.get('model_type', 'bcdu_net')
        
        if model_type == 'bcdu_net_lite':
            model = BCDUNetLite(in_channels=1, out_channels=1)
        else:
            model = BCDUNet(
                in_channels=1,
                out_channels=1,
                base_features=config.get('base_features', 64),
                use_bconvlstm=config.get('use_bconvlstm', True)
            )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_dice' in checkpoint:
                print(f"   Dice: {checkpoint['best_dice']:.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        print("   ✓ Segmentation loaded")
        
        return model
    
    def preprocess_image(self, image_path, apply_clahe=True):
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to image file
            apply_clahe: Whether to apply CLAHE enhancement
        
        Returns:
            tuple: (original_image, classifier_input, segmentation_input)
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original = image.copy()
        original_size = image.shape[:2]
        
        # Apply CLAHE
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        
        # Prepare for classifier (256x256, 3-channel)
        cls_img = cv2.resize(image, (256, 256))
        cls_img = cls_img.astype(np.float32) / 255.0
        cls_tensor = torch.from_numpy(cls_img).unsqueeze(0).repeat(3, 1, 1)  # (3, 256, 256)
        cls_tensor = cls_tensor.unsqueeze(0).to(self.device)  # (1, 3, 256, 256)
        
        # Prepare for segmentation (512x512, 1-channel)
        seg_img = cv2.resize(image, (512, 512))
        seg_img = seg_img.astype(np.float32) / 255.0
        seg_tensor = torch.from_numpy(seg_img).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
        seg_tensor = seg_tensor.to(self.device)
        
        return original, original_size, cls_tensor, seg_tensor
    
    @torch.no_grad()
    def predict(self, image_path, seg_threshold=0.5):
        """
        Run multi-task inference on a single image.
        
        Args:
            image_path: Path to mammogram image
            seg_threshold: Threshold for segmentation mask
        
        Returns:
            dict: Prediction results
        """
        # Preprocess
        original, original_size, cls_input, seg_input = self.preprocess_image(image_path)
        
        # Classification
        cls_output = self.classifier(cls_input)
        cls_probs = F.softmax(cls_output, dim=1)[0]
        benign_prob = cls_probs[0].item()
        malignant_prob = cls_probs[1].item()
        prediction = 'MALIGNANT' if malignant_prob > benign_prob else 'BENIGN'
        confidence = max(benign_prob, malignant_prob)
        
        # Segmentation
        seg_output = self.segmentation(seg_input)
        seg_probs = torch.sigmoid(seg_output)[0, 0].cpu().numpy()
        seg_mask = (seg_probs > seg_threshold).astype(np.uint8) * 255
        
        # Resize mask to original size
        seg_mask_original = cv2.resize(seg_mask, (original_size[1], original_size[0]), 
                                        interpolation=cv2.INTER_NEAREST)
        seg_probs_original = cv2.resize(seg_probs, (original_size[1], original_size[0]))
        
        # Compute tumor statistics
        tumor_area_pixels = (seg_mask_original > 0).sum()
        total_pixels = seg_mask_original.size
        tumor_area_ratio = tumor_area_pixels / total_pixels
        
        # Find tumor bounding box
        contours, _ = cv2.findContours(seg_mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest_contour)
        
        return {
            'image_path': str(image_path),
            'classification': {
                'prediction': prediction,
                'benign_probability': benign_prob,
                'malignant_probability': malignant_prob,
                'confidence': confidence
            },
            'segmentation': {
                'mask': seg_mask_original,
                'probabilities': seg_probs_original,
                'tumor_area_pixels': int(tumor_area_pixels),
                'tumor_area_ratio': float(tumor_area_ratio),
                'bounding_box': bbox  # (x, y, w, h)
            },
            'original_image': original
        }
    
    def visualize_result(self, result, save_path=None, show=False):
        """
        Visualize prediction result.
        
        Args:
            result: Result from predict()
            save_path: Path to save visualization
            show: Whether to display the plot
        """
        original = result['original_image']
        mask = result['segmentation']['mask']
        probs = result['segmentation']['probabilities']
        cls = result['classification']
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Mammogram')
        axes[0].axis('off')
        
        # Probability map
        axes[1].imshow(probs, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Segmentation Probability')
        axes[1].axis('off')
        
        # Binary mask
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f'Tumor Mask\nArea: {result["segmentation"]["tumor_area_ratio"]*100:.2f}%')
        axes[2].axis('off')
        
        # Overlay
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 0] = mask  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Draw bounding box
        if result['segmentation']['bounding_box']:
            x, y, w, h = result['segmentation']['bounding_box']
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        axes[3].imshow(overlay)
        axes[3].set_title(f'Classification: {cls["prediction"]}\n'
                         f'Confidence: {cls["confidence"]*100:.1f}%\n'
                         f'Malignant: {cls["malignant_probability"]*100:.1f}%')
        axes[3].axis('off')
        
        plt.suptitle(f'Multi-Task Analysis: {Path(result["image_path"]).name}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def batch_inference(self, image_dir, output_dir, extensions=('.png', '.jpg', '.jpeg')):
        """
        Run inference on all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            extensions: Valid image extensions
        
        Returns:
            list: List of results
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
        
        print(f"\nProcessing {len(image_files)} images...")
        
        results = []
        
        for img_path in image_files:
            try:
                print(f"  Processing: {img_path.name}")
                result = self.predict(img_path)
                
                # Save visualization
                viz_path = output_dir / f'{img_path.stem}_result.png'
                self.visualize_result(result, save_path=str(viz_path))
                
                # Save mask
                mask_path = output_dir / f'{img_path.stem}_mask.png'
                cv2.imwrite(str(mask_path), result['segmentation']['mask'])
                
                # Collect result (without large arrays)
                results.append({
                    'image': img_path.name,
                    'classification': result['classification'],
                    'tumor_area_ratio': result['segmentation']['tumor_area_ratio'],
                    'bounding_box': result['segmentation']['bounding_box']
                })
                
            except Exception as e:
                print(f"    Error: {e}")
        
        # Save summary
        summary_path = output_dir / 'results_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Processed {len(results)} images")
        print(f"  Results saved to: {output_dir}")
        
        # Print summary
        malignant_count = sum(1 for r in results if r['classification']['prediction'] == 'MALIGNANT')
        print(f"\n📊 Summary:")
        print(f"   Benign: {len(results) - malignant_count}")
        print(f"   Malignant: {malignant_count}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Multi-task mammography inference')
    parser.add_argument('--classifier', type=str, 
                        default='../checkpoints/best_resnet50_swinv2.ptrom',
                        help='Path to classifier checkpoint')
    parser.add_argument('--segmentation', type=str,
                        default='../checkpoints/segmentation/best_bcdu_segmentation.pth',
                        help='Path to segmentation checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory of images for batch inference')
    parser.add_argument('--output-dir', type=str, default='../outputs/multitask',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MultiTaskInference(
        classifier_checkpoint=args.classifier,
        segmentation_checkpoint=args.segmentation
    )
    
    if args.image:
        # Single image inference
        result = pipeline.predict(args.image)
        
        print(f"\n📊 Results for: {args.image}")
        print(f"   Classification: {result['classification']['prediction']}")
        print(f"   Confidence: {result['classification']['confidence']*100:.1f}%")
        print(f"   Malignant probability: {result['classification']['malignant_probability']*100:.1f}%")
        print(f"   Tumor area: {result['segmentation']['tumor_area_ratio']*100:.2f}%")
        
        # Save visualization
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_path = output_dir / f'{Path(args.image).stem}_result.png'
        pipeline.visualize_result(result, save_path=str(viz_path))
        
    elif args.image_dir:
        # Batch inference
        pipeline.batch_inference(args.image_dir, args.output_dir)
    
    else:
        print("Please specify --image or --image-dir")


if __name__ == '__main__':
    main()

