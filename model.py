"""
Model Architecture - Mammography Classification System
=========================================================================

This module implements a state-of-the-art hybrid deep learning architecture for
mammography image classification, combining the strengths of CNNs and Vision Transformers.

Architecture Overview:
----------------------
The model uses a dual-backbone approach:
    1. ResNet-101: Captures local features and edge patterns in mammograms
    2. SwinV2-Tiny: Captures global context and long-range dependencies

These features are enhanced with Squeeze-and-Excitation (SE) attention blocks
and fused through a multi-layer fusion network for final classification.

Key Components:
---------------
- SEBlock: Channel attention mechanism for adaptive feature recalibration
- ImprovedHybridV15: Main hybrid architecture (~75M parameters)
- load_model: Utility function for loading trained checkpoints

Performance:
------------
Best performing model achieved AUC of 0.813 on the CBIS-DDSM test set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for Channel Attention.
    
    SE blocks adaptively recalibrate channel-wise feature responses by explicitly
    modeling interdependencies between channels. This helps the model focus on
    the most informative features for mammography classification.
    
    The mechanism works in two steps:
        1. Squeeze: Global average pooling compresses spatial dimensions
        2. Excitation: Two fully-connected layers learn channel-wise importance weights
    
    Architecture:
    -------------
        Input (B, C) → Squeeze (B, C, 1) → FC1 (B, C/r) → ReLU → FC2 (B, C) → Sigmoid → Scale
    
    Attributes:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for the bottleneck layer (default: 16)
                        Higher reduction = fewer parameters but less capacity
    
    Example:
        >>> se_block = SEBlock(channels=2048, reduction=16)
        >>> features = torch.randn(4, 2048)  # Batch of 4, 2048 channels
        >>> output = se_block(features)      # Same shape, recalibrated weights
    
    Reference:
        Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    
    def __init__(self, channels, reduction=16):
        """
        Initialize the SE Block.
        
        Args:
            channels (int): Number of input feature channels
            reduction (int): Reduction ratio for bottleneck. A value of 16 means
                            the hidden layer has channels/16 neurons.
        """
        super().__init__()
        
        # Squeeze: Adaptive pooling to reduce spatial dimensions to 1x1
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: Two FC layers with ReLU and Sigmoid activation
        # This learns the channel-wise importance weights
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # Bottleneck layer
            nn.ReLU(inplace=True),                                    # Non-linearity
            nn.Linear(channels // reduction, channels, bias=False),  # Restore dimensions
            nn.Sigmoid()                                              # Normalize to [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass through the SE block.
        
        Args:
            x (Tensor): Input features of shape (batch_size, channels)
        
        Returns:
            Tensor: Recalibrated features with same shape as input,
                   where each channel is scaled by its learned importance weight
        """
        b, c = x.shape
        
        # Reshape for pooling: (B, C) -> (B, C, 1)
        y = x.unsqueeze(2)
        
        # Squeeze: Global average pooling
        y = self.squeeze(y).squeeze(2)
        
        # Excitation: Learn channel importance weights
        y = self.excitation(y)
        
        # Scale: Multiply input by learned weights (channel-wise attention)
        return x * y


class ImprovedHybridV15(nn.Module):
    """
    V15 Hybrid Architecture: ResNet-101 + SwinV2-Tiny + SE Attention.
    
    This is the flagship model of the MammoPipe system, combining:
        - ResNet-101 CNN backbone for local feature extraction
        - SwinV2-Tiny Vision Transformer for global context
        - SE attention blocks for adaptive feature recalibration
        - Multi-layer fusion network for combining dual-backbone features
    
    Architecture Diagram:
    ---------------------
                        Input Image (3, 256, 256)
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
         ResNet-101                         SwinV2-Tiny
         (45M params)                       (21M params)
              │                                  │
              ▼                                  ▼
         SE Attention                       SE Attention
         (2048-dim)                         (768-dim)
              │                                  │
              └────────────────┬────────────────┘
                               ▼
                         Concatenate
                          (2816-dim)
                               │
                               ▼
                        Fusion Network
                    (LayerNorm + GELU + Dropout)
                          (512-dim)
                               │
                               ▼
                          Classifier
                    (Benign vs Malignant)
    
    Why Hybrid Architecture?
    ------------------------
    - ResNet-101 excels at detecting local patterns (microcalcifications, edges)
    - SwinV2 captures global relationships (mass shape, tissue distribution)
    - SE blocks help both backbones focus on diagnostically relevant features
    - Fusion network learns optimal combination of both feature types
    
    Attributes:
        resnet: ResNet-50 backbone (pretrained on ImageNet)
        swin: SwinV2-Tiny transformer backbone (pretrained on ImageNet)
        resnet_se: SE attention block for ResNet features
        swin_se: SE attention block for SwinV2 features
        fusion: Multi-layer fusion network
        classifier: Final classification head
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary)
        dropout (float): Dropout rate for regularization (default: 0.3)
        pretrained (bool): Whether to use ImageNet pretrained weights (default: True)
    
    Example:
        >>> model = ImprovedHybridV15(num_classes=2, dropout=0.3)
        >>> x = torch.randn(4, 3, 256, 256)  # Batch of 4 mammograms
        >>> logits = model(x)                 # Shape: (4, 2)
        >>> probs = F.softmax(logits, dim=1)  # Class probabilities
    """
    
    def __init__(self, num_classes=2, dropout=0.3, pretrained=True):
        """
        Initialize the V15 Hybrid Model.
        
        Args:
            num_classes (int): Number of output classes. For binary classification
                              (benign vs malignant), use 2.
            dropout (float): Dropout probability for regularization. Higher values
                            reduce overfitting but may underfit. Recommended: 0.3
            pretrained (bool): If True, initialize backbones with ImageNet weights.
                              Set to False only for inference with loaded checkpoints.
        """
        super().__init__()
        
        # =====================================================================
        # BACKBONE 1: ResNet-101 CNN
        # =====================================================================
        # ResNet-101 is excellent for detecting local features like:
        # - Microcalcifications (small calcium deposits)
        # - Edge patterns and boundaries
        # - Texture variations in breast tissue
        self.resnet = timm.create_model(
            'resnet101',
            pretrained=pretrained,
            num_classes=0,              # Remove classification head (features only)
            drop_rate=dropout * 0.5     # Light dropout in backbone
        )
        
        resnet_dim = self.resnet.num_features  # Output: 2048 dimensions
        
        # =====================================================================
        # BACKBONE 2: SwinV2-Small Vision Transformer
        # =====================================================================
        # SwinV2 captures global context and long-range dependencies:
        # - Overall mass shape and structure
        # - Relationship between lesion and surrounding tissue
        # - Bilateral symmetry patterns
        self.swin = timm.create_model(
            'swinv2_tiny_window8_256',   # Window size 8, input 256x256
            pretrained=pretrained,
            num_classes=0,               # Remove classification head
            drop_rate=dropout * 0.3      # Light dropout
        )
        
        swin_dim = self.swin.num_features  # Output: 768 dimensions
        
        # =====================================================================
        # SE ATTENTION BLOCKS
        # =====================================================================
        # Apply channel attention to both backbone outputs
        # This helps each backbone focus on its most relevant features
        self.resnet_se = SEBlock(resnet_dim, reduction=16)
        self.swin_se = SEBlock(swin_dim, reduction=16)
        
        # =====================================================================
        # FUSION NETWORK
        # =====================================================================
        # Combines ResNet (2048-dim) and SwinV2 (768-dim) features
        # Two-layer MLP with LayerNorm for stable training
        self.fusion = nn.Sequential(
            nn.Linear(resnet_dim + swin_dim, 512),  # 2816 -> 512
            nn.LayerNorm(512),                       # Normalize for stability
            nn.GELU(),                               # Smooth activation
            nn.Dropout(dropout),                     # Regularization
            nn.Linear(512, 512),                     # Second fusion layer
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)                # Lighter dropout
        )
        
        # =====================================================================
        # CLASSIFICATION HEAD
        # =====================================================================
        # Final classifier: 512 -> 128 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Print model summary for verification
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ V15 Model:")
        print(f"  Total params: {total_params:,}")
        print(f"  ResNet-101: ~45M")
        print(f"  SwinV2-Tiny: ~21M")
        print(f"  Dropout: {dropout}")
    
    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Processing Pipeline:
            1. Extract features from both backbones in parallel
            2. Apply SE attention to each feature set
            3. Concatenate attended features
            4. Pass through fusion network
            5. Generate class logits
        
        Args:
            x (Tensor): Input mammogram images
                       Shape: (batch_size, 3, 256, 256)
                       Note: Grayscale images should be converted to 3-channel
        
        Returns:
            Tensor: Raw logits for each class
                   Shape: (batch_size, num_classes)
                   Apply softmax for probabilities: F.softmax(logits, dim=1)
        """
        # x shape: (B, 3, 256, 256)
        
        # ----- Step 1: Extract features from both backbones -----
        # ResNet-101: Local feature extraction (edges, textures, calcifications)
        resnet_features = self.resnet(x)  # (B, 2048)
        
        # SwinV2: Global feature extraction (shape, context, relationships)
        swin_features = self.swin(x)  # (B, 768)
        
        # ----- Step 2: Apply SE attention for adaptive feature weighting -----
        # Recalibrate channels based on learned importance
        resnet_features = self.resnet_se(resnet_features)  # (B, 2048)
        swin_features = self.swin_se(swin_features)  # (B, 768)
        
        # ----- Step 3: Combine features from both backbones -----
        # Concatenate along channel dimension
        combined = torch.cat([resnet_features, swin_features], dim=1)  # (B, 2816)
        
        # ----- Step 4: Fusion network -----
        # Learn optimal combination of CNN and Transformer features
        fused = self.fusion(combined)  # (B, 512)
        
        # ----- Step 5: Classification -----
        # Generate final class logits
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits


def load_model(checkpoint_path, num_classes=2):
    """
    Load a trained V15 model from a checkpoint file.
    
    This function handles both full checkpoints (with training state) and
    bare state dictionaries. It automatically selects the best available
    device (CUDA GPU or CPU).
    
    Args:
        checkpoint_path (str): Path to the .ptrom checkpoint file
        num_classes (int): Number of output classes (must match training config)
    
    Returns:
        tuple: (model, device)
            - model: Loaded model in evaluation mode (model.eval())
            - device: torch.device the model is loaded on
    
    Checkpoint Format:
        Full checkpoint dict containing:
            - 'model_state_dict': Model weights
            - 'epoch': Training epoch when saved
            - 'best_auc': Best validation AUC achieved
            - 'optimizer_state_dict': Optimizer state (not loaded for inference)
    
    Example:
        >>> model, device = load_model('checkpoints/best_resnet50_swinv2.ptrom')
        >>> model.eval()  # Already in eval mode
        >>> 
        >>> # Single image prediction
        >>> image = preprocess_image('mammogram.png')
        >>> with torch.no_grad():
        ...     logits = model(image.unsqueeze(0).to(device))
        ...     probs = F.softmax(logits, dim=1)
    
    Raises:
        FileNotFoundError: If checkpoint_path does not exist
        RuntimeError: If checkpoint architecture doesn't match model
    """
    # Determine best available device (GPU preferred for speed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📦 Loading model from: {checkpoint_path}")
    print(f"🔧 Device: {device}")
    
    # Load checkpoint (weights_only=False needed for full checkpoint dicts)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model architecture (pretrained=False since we're loading our weights)
    model = ImprovedHybridV15(
        num_classes=num_classes,
        dropout=0.3,
        pretrained=False  # Don't download ImageNet weights for inference
    ).to(device)
    
    # Load the trained weights
    if 'model_state_dict' in checkpoint:
        # Full checkpoint with training metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
        
        # Print training info if available
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_auc' in checkpoint:
            print(f"  Best AUC: {checkpoint['best_auc']:.4f}")
    else:
        # Bare state dict (just the weights)
        model.load_state_dict(checkpoint)
        print(f"✓ Model loaded successfully")
    
    # Set to evaluation mode (disables dropout, batch norm in eval mode)
    model.eval()
    
    return model, device


# =============================================================================
# TESTING / DEBUGGING
# =============================================================================
if __name__ == '__main__':
    """
    Test script to verify model creation and forward pass.
    
    Run with: python model.py
    
    Expected output:
        - Model architecture summary
        - Successful forward pass with correct output shape
    """
    print("Testing model creation...")
    
    # Create model without pretrained weights (faster for testing)
    model = ImprovedHybridV15(num_classes=2, dropout=0.3, pretrained=False)
    
    # Create dummy input batch (2 images, 3 channels, 256x256)
    x = torch.randn(2, 3, 256, 256)
    
    # Test forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
