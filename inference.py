#!/usr/bin/env python3
"""
Inference Script
==========================

This module provides the complete inference pipeline for the mammography
classification system. It handles model loading, image preprocessing,
batch inference, metrics computation, and result visualization.

Supported Operations:
---------------------
1. Single Image Prediction: Classify one mammogram at a time
2. Batch Inference: Process entire test dataset with metrics
3. Visualization: Generate confusion matrix and ROC curve

Performance Metrics:
--------------------
Best model (ResNet-50 + SwinV2-Small + SE) achieves:
    - AUC: 0.8103
    - Accuracy: ~0.78
    - Sensitivity: ~0.82
    - Specificity: ~0.75

Key Functions:
--------------
- preprocess_image(): Prepare single image for inference
- predict_single_image(): Classify one mammogram
- run_batch_inference(): Process entire dataset
- compute_metrics(): Calculate AUC, accuracy, sensitivity, etc.
- main(): Full inference pipeline

Usage:
------
    # Command line (full test set):
    $ python inference.py
    
    # Programmatic (single image):
    from inference import predict_single_image, load_model
    model, device = load_model('checkpoints/best_resnet50_swinv2.ptrom')
    result = predict_single_image(model, 'mammogram.png', device)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import sys

# Local imports
from src.config import config
from src.model import load_model
from src.dataset import CBISDDSMDataset


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_path, image_size=256):
    """
    Preprocess a single mammogram image for inference.
    
    This function applies the same preprocessing pipeline used during training
    to ensure consistent model behavior. The pipeline includes CLAHE contrast
    enhancement and normalization.
    
    Preprocessing Steps:
        1. Load as grayscale
        2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        3. Resize to target size
        4. Normalize to [0, 1] range
        5. Convert to 3-channel tensor
    
    Args:
        image_path (str): Path to mammogram image file (.jpg, .png, etc.)
        image_size (int): Target size for resizing (default: 256)
    
    Returns:
        Tensor: Preprocessed image of shape (3, image_size, image_size)
                Ready to be batched and passed to model
    
    Raises:
        ValueError: If image cannot be loaded from path
    
    Example:
        >>> image = preprocess_image('mammogram.jpg', image_size=256)
        >>> print(image.shape)  # torch.Size([3, 256, 256])
        >>> 
        >>> # Add batch dimension for model
        >>> batch = image.unsqueeze(0)  # Shape: (1, 3, 256, 256)
    """
    # Load image as grayscale (mammograms are grayscale)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply CLAHE for contrast enhancement
    # This is critical for revealing subtle features in mammograms
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    # Resize to model input size
    image = cv2.resize(image, (image_size, image_size))
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to 3-channel tensor (required for pretrained models)
    # Shape: (H, W) -> (1, H, W) -> (3, H, W)
    image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
    
    return image


# =============================================================================
# GRADCAM VISUALIZATION
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) for model interpretability.
    
    GradCAM generates visual explanations for CNN predictions by computing the
    gradient of the target class score with respect to feature maps, then
    weighting each feature map by its gradient importance.
    
    For the hybrid model, we use the ResNet backbone's final convolutional layer
    since it provides spatial feature maps (unlike the Transformer which uses tokens).
    
    How it works:
        1. Forward pass: Store activations from target layer
        2. Backward pass: Compute gradients of target class w.r.t. activations
        3. Weight feature maps by average gradient (importance)
        4. Apply ReLU to focus on positive contributions
        5. Resize heatmap to original image size
    
    Attributes:
        model: The hybrid model to explain
        target_layer: Layer to compute GradCAM on (default: ResNet layer4)
        activations: Stored feature map activations
        gradients: Stored gradients
    
    Example:
        >>> gradcam = GradCAM(model)
        >>> heatmap = gradcam.generate(image_tensor, target_class=1)
        >>> overlay = create_heatmap_overlay(original_image, heatmap)
    
    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM.
        
        Args:
            model: The hybrid model (ImprovedHybridV15)
            target_layer: Specific layer to compute GradCAM on.
                         If None, uses the last ResNet layer (layer4).
        """
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = []  # Store hook handles for cleanup
        
        # Use ResNet's layer4 as default (best spatial resolution for GradCAM)
        if target_layer is None:
            self.target_layer = model.resnet.layer4
        else:
            self.target_layer = target_layer
        
        # Register hooks to capture activations and gradients
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            # Don't detach - we need gradient flow through activations
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Store handles so we can remove hooks later
        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles = [handle_fwd, handle_bwd]
    
    def remove_hooks(self):
        """Remove registered hooks to prevent accumulation."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap for the input image.
        
        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W)
            target_class: Class to generate explanation for.
                         If None, uses the predicted class.
        
        Returns:
            numpy.ndarray: Heatmap of shape (H, W) with values in [0, 1]
                          Higher values indicate more important regions.
        """
        self.model.eval()
        
        # Create a copy of input tensor that requires gradients
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Zero all existing gradients
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        # Forward pass - get model output
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass for target class
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Compute GradCAM
        # Global average pooling of gradients (importance weights)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted sum of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU (focus on positive contributions)
        cam = F.relu(cam)
        
        # Convert to numpy and normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        
        # Handle edge case where cam is all zeros
        if cam.max() == cam.min():
            return np.zeros_like(cam)
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def __del__(self):
        """Cleanup hooks when object is deleted."""
        self.remove_hooks()


def generate_gradcam(model, input_tensor, target_class=None):
    """
    Convenience function to generate GradCAM without manually managing the class.
    
    This function creates a GradCAM instance, generates the heatmap, and cleans up.
    
    Args:
        model: The model to explain
        input_tensor: Preprocessed image tensor (1, 3, H, W)
        target_class: Target class for explanation (None = predicted class)
    
    Returns:
        numpy.ndarray: GradCAM heatmap with values in [0, 1]
    """
    gradcam = GradCAM(model)
    try:
        heatmap = gradcam.generate(input_tensor, target_class)
    finally:
        gradcam.remove_hooks()
    return heatmap


def create_heatmap_overlay(original_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Create a visualization overlay of the GradCAM heatmap on the original image.
    
    Args:
        original_image: Original image as numpy array (H, W) or (H, W, 3)
        heatmap: GradCAM heatmap as numpy array (H', W') with values in [0, 1]
        alpha: Transparency of heatmap overlay (0=invisible, 1=opaque)
        colormap: OpenCV colormap to use (default: JET for red-blue gradient)
    
    Returns:
        numpy.ndarray: Overlay image of shape (H, W, 3) with heatmap
    
    Example:
        >>> overlay = create_heatmap_overlay(image, heatmap, alpha=0.5)
        >>> cv2.imwrite('gradcam_result.jpg', overlay)
    """
    # Ensure original image is proper format
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    h, w = original_image.shape[:2]
    
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def predict_with_gradcam(model, image_path, device='cuda', save_path=None):
    """
    Predict mammogram class with GradCAM visualization.
    
    This function combines prediction with interpretability by generating
    a GradCAM heatmap that highlights the regions the model focuses on.
    This is useful for:
        - Understanding model decisions
        - Identifying potential cancer regions
        - Quality assurance and validation
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to mammogram image
        device: Device for inference ('cuda' or 'cpu')
        save_path: If provided, save the visualization to this path
    
    Returns:
        dict: Prediction results containing:
            - 'prediction': Human-readable label
            - 'class': Numeric class
            - 'confidence': Confidence score
            - 'probabilities': Class probabilities
            - 'heatmap': Raw GradCAM heatmap (numpy array)
            - 'overlay': Overlay visualization (numpy array)
            - 'save_path': Path where visualization was saved (if save_path provided)
    
    Example:
        >>> result = predict_with_gradcam(model, 'mammogram.jpg', device, 'output.jpg')
        >>> print(f"Prediction: {result['prediction']}")
        >>> print(f"Saved visualization to: {result['save_path']}")
    """
    # Load and preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Load original image for visualization
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get prediction first
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    # Generate GradCAM heatmap for the PREDICTED class (not always malignant)
    # This shows what regions the model used for its actual prediction
    heatmap = generate_gradcam(model, image_tensor, target_class=pred)
    
    # Create overlay visualization
    overlay = create_heatmap_overlay(original_image, heatmap, alpha=0.5)
    
    # Prepare result
    result = {
        'prediction': 'Malignant' if pred == 1 else 'Benign',
        'class': pred,
        'confidence': confidence,
        'probabilities': {
            'benign': probs[0][0].item(),
            'malignant': probs[0][1].item()
        },
        'heatmap': heatmap,
        'overlay': overlay
    }
    
    # Save visualization if path provided
    if save_path:
        # Create a comprehensive visualization
        save_gradcam_visualization(
            original_image, overlay, heatmap, result, save_path
        )
        result['save_path'] = save_path
    
    return result


def extract_segmentation_contours(heatmap, threshold=0.5, min_area=100):
    """
    Extract contours from GradCAM heatmap to create segmentation boundaries.
    
    This function converts the continuous GradCAM heatmap into discrete
    segmentation regions by thresholding and extracting contours.
    
    Args:
        heatmap: GradCAM heatmap as numpy array (H, W) with values in [0, 1]
        threshold: Activation threshold for region detection (default: 0.5)
                  Lower values = more regions, higher = only strong activations
        min_area: Minimum contour area in pixels (filters noise)
    
    Returns:
        list: List of contours, each contour is numpy array of points
        numpy.ndarray: Binary mask of segmented regions
    
    Example:
        >>> contours, mask = extract_segmentation_contours(heatmap, threshold=0.5)
        >>> cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    """
    # Convert heatmap to uint8
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply threshold to get binary mask
    threshold_value = int(255 * threshold)
    _, binary_mask = cv2.threshold(heatmap_uint8, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    return filtered_contours, binary_mask


def create_segmentation_overlay(original_image, heatmap, threshold=0.5, 
                                  contour_color=(0, 255, 0), contour_thickness=2,
                                  fill_alpha=0.4, show_intensity=True):
    """
    Create a pixel-level segmentation visualization on the original image.
    
    Instead of showing the full colorful heatmap, this only highlights the
    high-activation regions (the "red/hot" parts) as a clean segmentation mask.
    This mimics true pixel-level annotation.
    
    Args:
        original_image: Original image as numpy array (H, W, 3)
        heatmap: GradCAM heatmap as numpy array (H', W') with values in [0, 1]
        threshold: Activation threshold for segmentation (default: 0.5)
                  Higher = only strongest activations, lower = more regions
        contour_color: BGR color for contour lines (default: green)
        contour_thickness: Thickness of contour lines
        fill_alpha: Transparency for filled regions (0=invisible, 1=opaque)
        show_intensity: If True, vary fill opacity based on activation strength
    
    Returns:
        numpy.ndarray: Image with pixel-level segmentation mask
        list: List of detected contours
        dict: Region statistics (count, areas, bounding boxes)
    """
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    h, w = original_image.shape[:2]
    
    # Resize heatmap to match original image (bilinear interpolation for smooth edges)
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create output image (start with original)
    result = original_image.copy()
    
    if show_intensity:
        # Create smooth pixel-level mask based on activation intensity
        # Only show pixels above threshold, with intensity proportional to activation
        mask = np.zeros_like(heatmap_resized)
        above_threshold = heatmap_resized >= threshold
        
        # Scale activations above threshold to [0, 1] for display
        if above_threshold.any():
            mask[above_threshold] = (heatmap_resized[above_threshold] - threshold) / (1.0 - threshold + 1e-8)
        
        # Create colored overlay based on activation intensity
        # Higher activation = more color opacity
        overlay = original_image.copy()
        
        # Apply color to the mask regions with intensity-based alpha
        for c in range(3):
            overlay[:, :, c] = np.where(
                above_threshold,
                original_image[:, :, c] * (1 - mask * fill_alpha) + contour_color[c] * mask * fill_alpha,
                original_image[:, :, c]
            ).astype(np.uint8)
        
        result = overlay
    
    # Extract contours for the segmentation boundary
    contours, binary_mask = extract_segmentation_contours(
        heatmap_resized, threshold=threshold, min_area=int(w * h * 0.002)
    )
    
    # Draw clean contour boundaries
    cv2.drawContours(result, contours, -1, contour_color, contour_thickness)
    
    # Calculate region statistics
    region_stats = {
        'count': len(contours),
        'areas': [cv2.contourArea(c) for c in contours],
        'bounding_boxes': [cv2.boundingRect(c) for c in contours],
        'centroids': [],
        'total_area_percent': 0.0
    }
    
    # Calculate total segmented area as percentage
    total_area = sum(region_stats['areas'])
    region_stats['total_area_percent'] = (total_area / (w * h)) * 100
    
    # Calculate centroids
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            region_stats['centroids'].append((cx, cy))
    
    return result, contours, region_stats


def create_pixel_mask(original_image, heatmap, threshold=0.5, color=(0, 0, 255)):
    """
    Create a clean pixel-level segmentation mask showing only the detected regions.
    
    This produces a visualization similar to ground truth segmentation masks,
    with filled regions and clear boundaries.
    
    Args:
        original_image: Original image as numpy array
        heatmap: GradCAM heatmap with values in [0, 1]
        threshold: Activation threshold
        color: BGR color for the mask (default: red)
    
    Returns:
        numpy.ndarray: Image with pixel mask overlay
        numpy.ndarray: Binary mask (0 or 255)
    """
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    h, w = original_image.shape[:2]
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create binary mask from threshold
    binary_mask = (heatmap_resized >= threshold).astype(np.uint8) * 255
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Create colored mask
    colored_mask = np.zeros_like(original_image)
    colored_mask[binary_mask > 0] = color
    
    # Blend with original
    alpha = 0.4
    result = cv2.addWeighted(original_image, 1.0, colored_mask, alpha, 0)
    
    # Draw contours for sharp edges
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    
    return result, binary_mask


def predict_with_segmentation(model, image_path, device='cuda', save_path=None,
                               threshold=0.4):
    """
    Predict mammogram class with segmentation-style visualization.
    
    Combines classification with pseudo-segmentation by using GradCAM
    to identify and outline regions of interest with contours.
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to mammogram image
        device: Device for inference ('cuda' or 'cpu')
        save_path: If provided, save the visualization to this path
        threshold: Activation threshold for segmentation (0.0-1.0)
    
    Returns:
        dict: Results containing:
            - 'prediction': Human-readable label
            - 'class': Numeric class
            - 'confidence': Confidence score
            - 'probabilities': Class probabilities
            - 'heatmap': Raw GradCAM heatmap
            - 'segmentation': Image with contours
            - 'contours': List of detected contours
            - 'region_stats': Statistics about detected regions
    
    Example:
        >>> result = predict_with_segmentation(model, 'mammogram.jpg', device)
        >>> print(f"Detected {result['region_stats']['count']} regions")
    """
    # Load and preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Load original image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get prediction first
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    # Generate GradCAM heatmap for the predicted class
    heatmap = generate_gradcam(model, image_tensor, target_class=pred)
    
    # Choose contour color based on prediction
    # Green for benign, Red for malignant
    contour_color = (0, 0, 255) if pred == 1 else (0, 255, 0)
    
    # Create segmentation overlay
    segmentation, contours, region_stats = create_segmentation_overlay(
        original_image, heatmap, 
        threshold=threshold,
        contour_color=contour_color,
        contour_thickness=3,
        fill_alpha=0.25
    )
    
    # Also create heatmap overlay for comparison
    overlay = create_heatmap_overlay(original_image, heatmap, alpha=0.5)
    
    result = {
        'prediction': 'Malignant' if pred == 1 else 'Benign',
        'class': pred,
        'confidence': confidence,
        'probabilities': {
            'benign': probs[0][0].item(),
            'malignant': probs[0][1].item()
        },
        'heatmap': heatmap,
        'segmentation': segmentation,
        'overlay': overlay,
        'contours': contours,
        'region_stats': region_stats
    }
    
    if save_path:
        save_segmentation_visualization(
            original_image, overlay, segmentation, heatmap, result, save_path
        )
        result['save_path'] = save_path
    
    return result


def save_segmentation_visualization(original, overlay, segmentation, heatmap, 
                                     prediction, save_path):
    """
    Save a comprehensive visualization with pixel-level segmentation.
    
    Creates a 3-panel visualization showing:
        - Original image
        - GradCAM heatmap overlay (colorful)
        - Pixel-level segmentation (only red/hot regions)
    """
    h, w = original.shape[:2]
    
    # Resize images to same height
    target_h = 512
    scale = target_h / h
    target_w = int(w * scale)
    
    original_resized = cv2.resize(original, (target_w, target_h))
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    segmentation_resized = cv2.resize(segmentation, (target_w, target_h))
    
    # Create 3-panel image (Original | Heatmap | Segmentation)
    combined = np.hstack([original_resized, overlay_resized, segmentation_resized])
    
    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Panel labels
    cv2.putText(combined, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(combined, "GradCAM Heatmap", (target_w + 10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(combined, "Pixel Segmentation", (2 * target_w + 10, 30), font, font_scale, (255, 255, 255), thickness)
    
    # Prediction result on segmentation panel
    pred_text = f"{prediction['prediction']} ({prediction['confidence']:.1%})"
    color = (0, 0, 255) if prediction['class'] == 1 else (0, 255, 0)
    cv2.putText(combined, pred_text, (2 * target_w + 10, 60), font, font_scale, color, thickness)
    
    # Region info
    region_count = prediction['region_stats']['count']
    area_pct = prediction['region_stats'].get('total_area_percent', 0)
    cv2.putText(combined, f"Regions: {region_count} | Area: {area_pct:.1f}%", 
               (2 * target_w + 10, 90), font, 0.5, (255, 255, 255), 1)
    
    # Add probability bar at bottom of segmentation panel
    prob_malignant = prediction['probabilities']['malignant']
    bar_width = target_w - 30
    bar_height = 20
    bar_x = 2 * target_w + 15
    bar_y = target_h - 40
    
    # Benign portion (green)
    benign_width = int(bar_width * (1 - prob_malignant))
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + benign_width, bar_y + bar_height), (0, 180, 0), -1)
    
    # Malignant portion (red)
    cv2.rectangle(combined, (bar_x + benign_width, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 220), -1)
    
    # Border
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
    
    # Labels
    cv2.putText(combined, "B", (bar_x - 12, bar_y + 15), font, 0.4, (0, 255, 0), 1)
    cv2.putText(combined, "M", (bar_x + bar_width + 3, bar_y + 15), font, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite(str(save_path), combined)
    print(f"✓ Saved visualization: {save_path}")


def save_gradcam_visualization(original, overlay, heatmap, prediction, save_path):
    """
    Save a comprehensive GradCAM visualization with annotations.
    
    Creates a side-by-side visualization showing:
        - Original image
        - GradCAM heatmap overlay
        - Prediction and confidence
    
    Args:
        original: Original image (numpy array)
        overlay: Heatmap overlay image (numpy array)
        heatmap: Raw heatmap (numpy array)
        prediction: Prediction result dict
        save_path: Path to save the visualization
    """
    h, w = original.shape[:2]
    
    # Resize both images to same height for side-by-side
    target_h = 512
    scale = target_h / h
    target_w = int(w * scale)
    
    original_resized = cv2.resize(original, (target_w, target_h))
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    
    # Create side-by-side image
    combined = np.hstack([original_resized, overlay_resized])
    
    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Prediction text
    pred_text = f"{prediction['prediction']} ({prediction['confidence']:.1%})"
    color = (0, 0, 255) if prediction['class'] == 1 else (0, 255, 0)  # Red for malignant, green for benign
    
    # Add labels
    cv2.putText(combined, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(combined, "GradCAM", (target_w + 10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(combined, pred_text, (target_w + 10, 60), font, font_scale, color, thickness)
    
    # Add probability bar
    prob_malignant = prediction['probabilities']['malignant']
    bar_width = int(target_w * 0.8)
    bar_height = 20
    bar_x = target_w + 10
    bar_y = target_h - 40
    
    # Background bar
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    # Malignant probability bar
    mal_bar_width = int(bar_width * prob_malignant)
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + mal_bar_width, bar_y + bar_height), (0, 0, 255), -1)
    # Border
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
    
    # Labels for bar
    cv2.putText(combined, "B", (bar_x - 15, bar_y + 15), font, 0.5, (0, 255, 0), 1)
    cv2.putText(combined, "M", (bar_x + bar_width + 5, bar_y + 15), font, 0.5, (0, 0, 255), 1)
    
    # Save
    cv2.imwrite(str(save_path), combined)
    print(f"✓ Saved GradCAM visualization: {save_path}")


# =============================================================================
# SINGLE IMAGE PREDICTION
# =============================================================================


def predict_single_image(model, image_path, device='cuda'):
    """
    Predict the class of a single mammogram image.
    
    This function handles the complete prediction pipeline for one image:
    preprocessing, inference with mixed precision, and result formatting.
    
    Args:
        model: Loaded PyTorch model (should be in eval mode)
        image_path (str): Path to mammogram image file
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        dict: Prediction results containing:
            - 'prediction': Human-readable label ('Benign' or 'Malignant')
            - 'class': Numeric class (0 or 1)
            - 'confidence': Confidence score for predicted class (0.0 to 1.0)
            - 'probabilities': Dict with class probabilities
                - 'benign': Probability of benign class
                - 'malignant': Probability of malignant class
    
    Example:
        >>> from inference import predict_single_image, load_model
        >>> 
        >>> # Load model
        >>> model, device = load_model('checkpoints/best_resnet50_swinv2.ptrom')
        >>> 
        >>> # Predict
        >>> result = predict_single_image(model, 'patient_mammogram.jpg', device)
        >>> 
        >>> print(f"Diagnosis: {result['prediction']}")
        >>> print(f"Confidence: {result['confidence']:.1%}")
        >>> print(f"Benign probability: {result['probabilities']['benign']:.3f}")
        >>> print(f"Malignant probability: {result['probabilities']['malignant']:.3f}")
    """
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Add batch dimension and move to device
    # Shape: (3, H, W) -> (1, 3, H, W)
    image = image.unsqueeze(0).to(device)
    
    # Set model to evaluation mode (disables dropout)
    model.eval()
    
    # Run inference with no gradient computation and mixed precision
    with torch.no_grad():
        with torch.amp.autocast('cuda'):  # FP16 for speed
            logits = model(image)
        
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)
        
        # Get predicted class and confidence
        pred = logits.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    # Format results
    return {
        'prediction': 'Malignant' if pred == 1 else 'Benign',
        'class': pred,
        'confidence': confidence,
        'probabilities': {
            'benign': probs[0][0].item(),
            'malignant': probs[0][1].item()
        }
    }


# =============================================================================
# BATCH INFERENCE
# =============================================================================

def run_batch_inference(model, dataloader, device='cuda'):
    """
    Run inference on an entire dataset.
    
    This function processes all samples in the dataloader, collecting
    predictions, labels, and probabilities for subsequent metric computation.
    
    Args:
        model: Loaded PyTorch model (should be in eval mode)
        dataloader: PyTorch DataLoader containing test samples
        device (str): Device for inference ('cuda' or 'cpu')
    
    Returns:
        dict: Inference results containing:
            - 'predictions': numpy array of predicted classes (0 or 1)
            - 'labels': numpy array of ground truth labels
            - 'probabilities': numpy array of shape (N, 2) with class probs
    
    Example:
        >>> results = run_batch_inference(model, test_loader, device)
        >>> print(f"Processed {len(results['predictions'])} samples")
        >>> 
        >>> # Compute metrics
        >>> metrics = compute_metrics(results)
    """
    model.eval()
    
    # Containers for collecting results
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔮 Running batch inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Move data to device
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                logits = model(images)
            
            # Convert to probabilities and predictions
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    results = {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }
    
    print(f"✓ Processed {len(all_preds)} samples")
    return results


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(results, threshold=0.5):
    """
    Compute comprehensive classification metrics.
    
    Calculates all standard binary classification metrics from the
    inference results. Uses the specified threshold for converting
    probabilities to binary predictions.
    
    Metrics Computed:
        - AUC: Area Under ROC Curve (threshold-independent)
        - Accuracy: Overall classification accuracy
        - Sensitivity (Recall): True positive rate
        - Specificity: True negative rate
        - Precision: Positive predictive value
        - F1-Score: Harmonic mean of precision and recall
        - Confusion Matrix: TP, TN, FP, FN counts
    
    Args:
        results (dict): Output from run_batch_inference()
        threshold (float): Classification threshold (default: 0.5)
                          Samples with malignant_prob >= threshold are
                          classified as malignant
    
    Returns:
        dict: Computed metrics including:
            - 'threshold': Classification threshold used
            - 'accuracy': Overall accuracy
            - 'auc': Area under ROC curve
            - 'sensitivity': True positive rate (recall)
            - 'specificity': True negative rate
            - 'precision': Positive predictive value
            - 'f1_score': F1 score
            - 'confusion_matrix': Dict with 'tn', 'fp', 'fn', 'tp'
    
    Example:
        >>> metrics = compute_metrics(results, threshold=0.5)
        >>> print(f"AUC: {metrics['auc']:.4f}")
        >>> print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    """
    labels = results['labels']
    probs_malignant = results['probabilities'][:, 1]  # P(malignant)
    
    # Apply threshold to get binary predictions
    preds = (probs_malignant >= threshold).astype(int)
    
    # Compute confusion matrix components
    # tn = true negatives, fp = false positives
    # fn = false negatives, tp = true positives
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs_malignant)
    
    # Sensitivity (Recall): What fraction of actual positives were detected?
    # Important in medical imaging - we don't want to miss malignant cases
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity: What fraction of actual negatives were correctly identified?
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision: What fraction of predicted positives are actually positive?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1-Score: Harmonic mean of precision and sensitivity
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_confusion_matrix(metrics, save_path):
    """
    Generate and save a confusion matrix visualization.
    
    Creates a heatmap showing the distribution of true vs predicted labels,
    annotated with counts and the overall AUC score.
    
    Args:
        metrics (dict): Output from compute_metrics()
        save_path (str): Path to save the PNG image
    
    Visualization:
        - Rows: True labels (Benign, Malignant)
        - Columns: Predicted labels (Benign, Malignant)
        - Color: Intensity indicates count
        - Title: Includes AUC score
    """
    cm = metrics['confusion_matrix']
    
    # Reshape confusion matrix for visualization
    # [[TN, FP], [FN, TP]]
    cm_array = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f"Confusion Matrix (AUC: {metrics['auc']:.4f})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion matrix: {save_path}")


def plot_roc_curve(results, save_path):
    """
    Generate and save an ROC curve visualization.
    
    The ROC (Receiver Operating Characteristic) curve plots the trade-off
    between sensitivity and specificity at various thresholds.
    
    Args:
        results (dict): Output from run_batch_inference()
        save_path (str): Path to save the PNG image
    
    Interpretation:
        - Curve closer to top-left corner = better performance
        - AUC of 1.0 = perfect classifier
        - AUC of 0.5 = random classifier (diagonal line)
        - Our model achieves AUC ~0.81 (good performance)
    """
    # Compute ROC curve points
    fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'][:, 1])
    auc = roc_auc_score(results['labels'], results['probabilities'][:, 1])
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Mammography Classification')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved ROC curve: {save_path}")


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def save_results(metrics, results, output_dir):
    """
    Save all inference results to disk.
    
    Creates timestamped output files including:
        1. metrics_TIMESTAMP.json: All computed metrics
        2. predictions_TIMESTAMP.csv: Per-sample predictions
        3. confusion_matrix_TIMESTAMP.png: Confusion matrix plot
        4. roc_curve_TIMESTAMP.png: ROC curve plot
    
    Args:
        metrics (dict): Output from compute_metrics()
        results (dict): Output from run_batch_inference()
        output_dir (str): Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics as JSON
    metrics_file = output_dir / f'metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_file}")
    
    # Save predictions as CSV
    predictions_df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'prob_benign': results['probabilities'][:, 0],
        'prob_malignant': results['probabilities'][:, 1]
    })
    predictions_file = output_dir / f'predictions_{timestamp}.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Saved predictions: {predictions_file}")
    
    # Generate visualizations
    cm_file = output_dir / f'confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(metrics, cm_file)
    
    roc_file = output_dir / f'roc_curve_{timestamp}.png'
    plot_roc_curve(results, roc_file)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Main inference pipeline - runs complete evaluation on test set.
    
    Pipeline Steps:
        1. Load trained model from checkpoint
        2. Load test dataset
        3. Run batch inference on all test samples
        4. Compute comprehensive metrics
        5. Print results summary
        6. Save results and visualizations
    
    Output:
        - Console: Metrics summary
        - outputs/: JSON metrics, CSV predictions, PNG visualizations
    """
    print("="*70)
    print("Inference - Best Model (AUC 0.8103)")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Model
    # -------------------------------------------------------------------------
    print("\n Loading model...")
    checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / 'best_resnet50_swinv2.ptrom'
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    model, device = load_model(str(checkpoint_path))
    print(f"✓ Model loaded on {device}")
    
    # -------------------------------------------------------------------------
    # Step 2: Load Dataset
    # -------------------------------------------------------------------------
    print("\n Loading dataset...")
    dataset = CBISDDSMDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        transform=None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,  # Keep order for reproducibility
        num_workers=config['data']['num_workers'],
        pin_memory=True  # Faster CPU to GPU transfer
    )
    
    print(f"✓ Loaded {len(dataset)} test samples")
    
    # -------------------------------------------------------------------------
    # Step 3: Run Inference
    # -------------------------------------------------------------------------
    results = run_batch_inference(model, dataloader, device)
    
    # -------------------------------------------------------------------------
    # Step 4: Compute Metrics
    # -------------------------------------------------------------------------
    print("\n Computing metrics...")
    metrics = compute_metrics(results)
    
    # -------------------------------------------------------------------------
    # Step 5: Print Results
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['tn']:4d}  FP: {cm['fp']:4d}")
    print(f"  FN: {cm['fn']:4d}  TP: {cm['tp']:4d}")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 6: Save Results
    # -------------------------------------------------------------------------
    output_dir = Path(__file__).parent.parent / 'outputs'
    save_results(metrics, results, output_dir)
    
    print(f"\n✅ Inference complete! Results saved to: {output_dir}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run the complete inference pipeline.
    
    Usage: python inference.py
    """
    main()