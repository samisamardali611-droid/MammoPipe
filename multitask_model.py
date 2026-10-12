"""
Multi-Task Breast Cancer Model: Classification + Segmentation
=============================================================

This module implements a multi-task deep learning model that performs both:
1. Classification: Benign vs Malignant probability
2. Segmentation: Pixel-perfect tumor mask

The architecture merges the existing ResNet-SwinV2 classifier with BCDU-Net
segmentation to create a unified model for comprehensive mammography analysis.

Architecture Overview:
----------------------
                    Input Image (1, 512, 512)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  Classification │       │  Segmentation   │
    │     Branch      │       │     Branch      │
    │  (ResNet+Swin)  │       │   (BCDU-Net)    │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             ▼                         ▼
     Benign/Malignant             Tumor Mask
      Probability                 (512, 512)

Usage:
------
    from multitask_model import MultiTaskBreastCancerModel
    
    model = MultiTaskBreastCancerModel(
        classifier_checkpoint='checkpoints/best_classifier.pth',
        segmentation_checkpoint='checkpoints/best_bcdu_segmentation.pth'
    )
    
    cls_output, seg_output = model(image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import local modules
from model import ImprovedHybridV15, load_model as load_classifier
from bcdu_net import BCDUNet, BCDUNetLite


class MultiTaskBreastCancerModel(nn.Module):
    """
    Multi-Task Model for simultaneous classification and segmentation.
    
    This model combines:
    - Classification branch: ResNet-101 + SwinV2-Tiny hybrid for benign/malignant
    - Segmentation branch: BCDU-Net for tumor segmentation
    
    The branches can share a common encoder or operate independently.
    
    Args:
        num_classes (int): Number of classification classes (default: 2)
        shared_encoder (bool): Whether to share encoder between tasks (default: False)
        classifier_checkpoint (str): Path to pretrained classifier weights
        segmentation_checkpoint (str): Path to pretrained segmentation weights
        freeze_classifier (bool): Freeze classifier weights during training (default: False)
        freeze_segmentation (bool): Freeze segmentation weights during training (default: False)
    
    Example:
        >>> model = MultiTaskBreastCancerModel(num_classes=2)
        >>> image = torch.randn(4, 1, 512, 512)
        >>> cls_out, seg_out = model(image)
        >>> print(cls_out.shape)  # (4, 2)
        >>> print(seg_out.shape)  # (4, 1, 512, 512)
    """
    
    def __init__(
        self,
        num_classes=2,
        shared_encoder=False,
        classifier_checkpoint=None,
        segmentation_checkpoint=None,
        freeze_classifier=False,
        freeze_segmentation=False,
        seg_model_type='bcdu_net'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.shared_encoder = shared_encoder
        
        # =====================================================================
        # CLASSIFICATION BRANCH
        # =====================================================================
        # ResNet-101 + SwinV2-Tiny hybrid model
        print("🔧 Initializing Classification Branch...")
        self.classifier = ImprovedHybridV15(
            num_classes=num_classes,
            dropout=0.3,
            pretrained=classifier_checkpoint is None
        )
        
        # Load pretrained classifier weights
        if classifier_checkpoint:
            self._load_classifier_weights(classifier_checkpoint)
        
        # Freeze classifier if requested
        if freeze_classifier:
            self._freeze_module(self.classifier)
            print("   ❄️ Classifier weights frozen")
        
        # =====================================================================
        # SEGMENTATION BRANCH
        # =====================================================================
        # BCDU-Net for tumor segmentation
        print("🔧 Initializing Segmentation Branch...")
        
        if seg_model_type == 'bcdu_net_lite':
            self.segmentation = BCDUNetLite(in_channels=1, out_channels=1)
        else:
            self.segmentation = BCDUNet(
                in_channels=1,
                out_channels=1,
                base_features=64,
                use_bconvlstm=True
            )
        
        # Load pretrained segmentation weights
        if segmentation_checkpoint:
            self._load_segmentation_weights(segmentation_checkpoint)
        
        # Freeze segmentation if requested
        if freeze_segmentation:
            self._freeze_module(self.segmentation)
            print("   ❄️ Segmentation weights frozen")
        
        # =====================================================================
        # MODEL SUMMARY
        # =====================================================================
        cls_params = sum(p.numel() for p in self.classifier.parameters())
        seg_params = sum(p.numel() for p in self.segmentation.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n✓ MultiTaskBreastCancerModel initialized:")
        print(f"  Classification params: {cls_params:,}")
        print(f"  Segmentation params: {seg_params:,}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
    
    def _load_classifier_weights(self, checkpoint_path):
        """Load pretrained classifier weights."""
        print(f"   📦 Loading classifier from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            if 'best_auc' in checkpoint:
                print(f"   ✓ Loaded (AUC: {checkpoint['best_auc']:.4f})")
        else:
            self.classifier.load_state_dict(checkpoint)
            print(f"   ✓ Loaded weights")
    
    def _load_segmentation_weights(self, checkpoint_path):
        """Load pretrained segmentation weights."""
        print(f"   📦 Loading segmentation from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.segmentation.load_state_dict(checkpoint['model_state_dict'])
            if 'best_dice' in checkpoint:
                print(f"   ✓ Loaded (Dice: {checkpoint['best_dice']:.4f})")
        else:
            self.segmentation.load_state_dict(checkpoint)
            print(f"   ✓ Loaded weights")
    
    def _freeze_module(self, module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False
    
    def _unfreeze_module(self, module):
        """Unfreeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass through multi-task model.
        
        Args:
            x (Tensor): Input images (B, 1, H, W) for grayscale
                       Will be converted to (B, 3, H, W) for classifier
        
        Returns:
            tuple: (classification_logits, segmentation_logits)
                - classification_logits: (B, num_classes)
                - segmentation_logits: (B, 1, H, W)
        """
        # Handle input dimensions
        # Segmentation expects 1-channel, classifier expects 3-channel
        if x.shape[1] == 1:
            # Expand 1-channel to 3-channel for classifier
            x_cls = x.repeat(1, 3, 1, 1)
            x_seg = x
        elif x.shape[1] == 3:
            # Use 3-channel for classifier, convert to 1-channel for segmentation
            x_cls = x
            x_seg = x[:, 0:1, :, :]  # Take first channel
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")
        
        # Resize for classifier if needed (expects 256x256)
        if x_cls.shape[2] != 256 or x_cls.shape[3] != 256:
            x_cls = F.interpolate(x_cls, size=(256, 256), mode='bilinear', align_corners=True)
        
        # Classification branch
        cls_output = self.classifier(x_cls)
        
        # Segmentation branch
        seg_output = self.segmentation(x_seg)
        
        return cls_output, seg_output
    
    def predict(self, x, seg_threshold=0.5):
        """
        Convenience method for inference.
        
        Args:
            x (Tensor): Input images
            seg_threshold (float): Threshold for binary segmentation mask
        
        Returns:
            dict: Dictionary containing:
                - 'classification': Predicted class label
                - 'class_probabilities': Probability for each class
                - 'segmentation_mask': Binary segmentation mask
                - 'segmentation_probs': Segmentation probability map
        """
        self.eval()
        with torch.no_grad():
            cls_logits, seg_logits = self.forward(x)
            
            # Classification
            cls_probs = F.softmax(cls_logits, dim=1)
            cls_pred = torch.argmax(cls_probs, dim=1)
            
            # Segmentation
            seg_probs = torch.sigmoid(seg_logits)
            seg_mask = (seg_probs > seg_threshold).float()
            
            return {
                'classification': cls_pred,
                'class_probabilities': cls_probs,
                'segmentation_mask': seg_mask,
                'segmentation_probs': seg_probs
            }


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    
    Combines classification loss (CrossEntropy) with segmentation loss (Dice + BCE).
    
    Args:
        cls_weight (float): Weight for classification loss (default: 0.5)
        seg_weight (float): Weight for segmentation loss (default: 0.5)
        dice_weight (float): Weight for Dice loss within segmentation (default: 0.5)
        bce_weight (float): Weight for BCE loss within segmentation (default: 0.5)
    """
    
    def __init__(self, cls_weight=0.5, seg_weight=0.5, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        # Classification loss
        self.cls_criterion = nn.CrossEntropyLoss()
        
        # Segmentation losses
        self.bce_criterion = nn.BCEWithLogitsLoss()
    
    def _dice_loss(self, pred, target, smooth=1.0):
        """Compute Dice loss."""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def forward(self, cls_pred, cls_target, seg_pred, seg_target):
        """
        Compute combined multi-task loss.
        
        Args:
            cls_pred (Tensor): Classification logits (B, num_classes)
            cls_target (Tensor): Classification labels (B,)
            seg_pred (Tensor): Segmentation logits (B, 1, H, W)
            seg_target (Tensor): Segmentation masks (B, 1, H, W)
        
        Returns:
            dict: Dictionary with total_loss and individual losses
        """
        # Classification loss
        cls_loss = self.cls_criterion(cls_pred, cls_target)
        
        # Segmentation loss (Dice + BCE)
        dice_loss = self._dice_loss(seg_pred, seg_target)
        bce_loss = self.bce_criterion(seg_pred, seg_target)
        seg_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        # Combined loss
        total_loss = self.cls_weight * cls_loss + self.seg_weight * seg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'seg_loss': seg_loss,
            'dice_loss': dice_loss,
            'bce_loss': bce_loss
        }


def load_multitask_model(checkpoint_path, device='cuda'):
    """
    Load a trained multi-task model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (model, config)
    """
    print(f"📦 Loading multi-task model from: {checkpoint_path}")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint.get('config', {})
    
    # Create model
    model = MultiTaskBreastCancerModel(
        num_classes=config.get('num_classes', 2),
        seg_model_type=config.get('seg_model_type', 'bcdu_net')
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model, config


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

def predict_mammogram(model, image_path, device='cuda'):
    """
    Run multi-task inference on a single mammogram.
    
    Args:
        model: MultiTaskBreastCancerModel instance
        image_path (str): Path to mammogram image
        device (str): Device for inference
    
    Returns:
        dict: Prediction results with classification and segmentation
    """
    import cv2
    import numpy as np
    
    # Load and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    # Resize to model input size
    original_size = image.shape[:2]
    image = cv2.resize(image, (512, 512))
    
    # Normalize and convert to tensor
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    results = model.predict(image_tensor)
    
    # Process results
    cls_pred = results['classification'][0].item()
    cls_probs = results['class_probabilities'][0].cpu().numpy()
    seg_mask = results['segmentation_mask'][0, 0].cpu().numpy()
    
    # Resize segmentation back to original size
    seg_mask_original = cv2.resize(
        seg_mask,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    return {
        'classification': {
            'prediction': 'malignant' if cls_pred == 1 else 'benign',
            'benign_probability': float(cls_probs[0]),
            'malignant_probability': float(cls_probs[1]),
            'confidence': float(max(cls_probs))
        },
        'segmentation': {
            'mask': (seg_mask_original * 255).astype(np.uint8),
            'tumor_area_ratio': float(seg_mask_original.mean())
        }
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    """Test multi-task model implementation."""
    print("Testing MultiTaskBreastCancerModel...")
    
    # Create model without pretrained weights for testing
    model = MultiTaskBreastCancerModel(
        num_classes=2,
        classifier_checkpoint=None,
        segmentation_checkpoint=None
    )
    
    # Test forward pass
    x = torch.randn(2, 1, 512, 512)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        cls_out, seg_out = model(x)
    
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Segmentation output shape: {seg_out.shape}")
    
    # Test prediction
    results = model.predict(x)
    print(f"\nPrediction results:")
    print(f"  Classification: {results['classification']}")
    print(f"  Class probabilities: {results['class_probabilities']}")
    print(f"  Segmentation mask shape: {results['segmentation_mask'].shape}")
    
    # Test loss function
    print("\n--- Testing MultiTaskLoss ---")
    criterion = MultiTaskLoss()
    
    cls_target = torch.randint(0, 2, (2,))
    seg_target = torch.randint(0, 2, (2, 1, 512, 512)).float()
    
    losses = criterion(cls_out, cls_target, seg_out, seg_target)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Classification loss: {losses['cls_loss'].item():.4f}")
    print(f"Segmentation loss: {losses['seg_loss'].item():.4f}")
    
    print("\n✓ All tests passed!")

