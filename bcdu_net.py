"""
BCDU-Net: Bidirectional Convolutional LSTM U-Net with Densely Connected Convolutions
====================================================================================

Implementation of BCDU-Net for medical image segmentation, particularly optimized
for mammography tumor segmentation.

Architecture Overview:
----------------------
BCDU-Net extends U-Net with three key improvements:
    1. Bidirectional ConvLSTM (BConvLSTM) - Captures spatial dependencies in both directions
    2. Dense Connections - Improves feature propagation and gradient flow
    3. Deep Supervision - Optional auxiliary outputs for better training

Network Structure:
------------------
    Input (1, H, W)
         │
    ┌────▼────┐
    │ Encoder │  (4 blocks with max pooling)
    │ Block 1 │ → Skip 1
    │ Block 2 │ → Skip 2  
    │ Block 3 │ → Skip 3
    │ Block 4 │ → Skip 4
    └────┬────┘
         │
    ┌────▼────┐
    │ Bridge  │  (BConvLSTM + Dense Block)
    └────┬────┘
         │
    ┌────▼────┐
    │ Decoder │  (4 blocks with upsampling + skip connections)
    │ Block 1 │ ← Skip 4 + BConvLSTM
    │ Block 2 │ ← Skip 3 + BConvLSTM
    │ Block 3 │ ← Skip 2 + BConvLSTM
    │ Block 4 │ ← Skip 1 + BConvLSTM
    └────┬────┘
         │
    ┌────▼────┐
    │  Output │  (1, H, W) - Segmentation mask
    └─────────┘

Reference:
----------
    Azad et al., "Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions"
    https://arxiv.org/abs/1909.00166
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BConvLSTM(nn.Module):
    """
    Simplified Bidirectional Convolutional LSTM for BCDU-Net.
    
    This implementation uses a simplified approach where instead of iterating
    over channels, we apply convolutions to capture spatial dependencies in
    both forward and backward directions, then combine the results.
    
    This is more memory efficient and avoids the complexity of true sequential
    ConvLSTM while still capturing bidirectional spatial dependencies.
    
    Args:
        input_channels (int): Number of input channels
        hidden_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size (default: 3)
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Forward path - processes features in one direction
        self.forward_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        
        # Backward path - processes features in reverse direction
        self.backward_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        
        # Combine forward and backward features
        self.combine = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Forward pass through simplified bidirectional ConvLSTM.
        
        The bidirectional aspect is achieved by:
        1. Processing features normally (forward)
        2. Flipping features spatially, processing, then flipping back (backward)
        3. Combining both representations
        
        Args:
            x (Tensor): Input tensor (B, C, H, W)
        
        Returns:
            Tensor: Output tensor (B, hidden_channels, H, W)
        """
        # Forward pass
        forward_out = self.forward_conv(x)
        
        # Backward pass - flip horizontally, process, flip back
        x_flipped = torch.flip(x, dims=[3])  # Flip along width
        backward_out = self.backward_conv(x_flipped)
        backward_out = torch.flip(backward_out, dims=[3])  # Flip back
        
        # Combine forward and backward
        combined = torch.cat([forward_out, backward_out], dim=1)
        output = self.combine(combined)
        
        return output


class DenseBlock(nn.Module):
    """
    Dense Block with densely connected convolutions.
    
    Each layer in the block receives feature maps from all preceding layers
    through concatenation, improving feature reuse and gradient flow.
    
    Structure:
    ----------
        x → Conv1 → [x, f1] → Conv2 → [x, f1, f2] → Conv3 → [x, f1, f2, f3] → ...
    
    Args:
        in_channels (int): Number of input channels
        growth_rate (int): Number of output channels per conv layer (default: 32)
        num_layers (int): Number of convolutional layers (default: 4)
    """
    
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(layer_in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(layer_in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
                )
            )
        
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x):
        """
        Forward pass through dense block.
        
        Args:
            x (Tensor): Input tensor (B, C, H, W)
        
        Returns:
            Tensor: Output tensor with densely connected features
        """
        features = [x]
        
        for layer in self.layers:
            # Concatenate all previous features
            combined = torch.cat(features, dim=1)
            out = layer(combined)
            features.append(out)
        
        # Final concatenation
        return torch.cat(features, dim=1)


class EncoderBlock(nn.Module):
    """
    Encoder block with two convolutions and optional pooling.
    
    Structure:
    ----------
        Conv → BN → ReLU → Conv → BN → ReLU → [MaxPool]
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        pool (bool): Whether to apply max pooling (default: True)
    """
    
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else None
    
    def forward(self, x):
        """
        Forward pass through encoder block.
        
        Args:
            x (Tensor): Input tensor
        
        Returns:
            tuple: (features before pooling, features after pooling or same if no pool)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        skip = x
        if self.pool:
            x = self.pool(x)
        
        return skip, x


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling, skip connection, BConvLSTM, and convolutions.
    
    Structure:
    ----------
        Upsample → Concat(skip) → BConvLSTM → Conv → BN → ReLU → Conv → BN → ReLU
    
    Args:
        in_channels (int): Number of input channels (from previous decoder block)
        skip_channels (int): Number of channels from skip connection
        out_channels (int): Number of output channels
        use_bconvlstm (bool): Whether to use BConvLSTM (default: True)
    """
    
    def __init__(self, in_channels, skip_channels, out_channels, use_bconvlstm=True):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        combined_channels = in_channels + skip_channels
        
        self.use_bconvlstm = use_bconvlstm
        if use_bconvlstm:
            self.bconvlstm = BConvLSTM(combined_channels, combined_channels, kernel_size=3)
            conv_in_channels = combined_channels
        else:
            conv_in_channels = combined_channels
        
        self.conv1 = nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        """
        Forward pass through decoder block.
        
        Args:
            x (Tensor): Input from previous decoder block
            skip (Tensor): Skip connection from encoder
        
        Returns:
            Tensor: Decoded features
        """
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply BConvLSTM if enabled
        if self.use_bconvlstm:
            x = self.bconvlstm(x)
        
        # Convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x


class BCDUNet(nn.Module):
    """
    BCDU-Net: Bidirectional ConvLSTM U-Net with Densely Connected Convolutions.
    
    A powerful medical image segmentation architecture that combines:
    - U-Net encoder-decoder structure for multi-scale feature extraction
    - Bidirectional ConvLSTM for capturing spatial dependencies
    - Dense connections for improved feature reuse
    
    This implementation is optimized for mammography tumor segmentation.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels/classes (default: 1 for binary)
        base_features (int): Base number of features (default: 64)
        use_bconvlstm (bool): Whether to use BConvLSTM in decoder (default: True)
    
    Example:
        >>> model = BCDUNet(in_channels=1, out_channels=1)
        >>> x = torch.randn(4, 1, 512, 512)  # Batch of 4 mammograms
        >>> output = model(x)  # (4, 1, 512, 512) - Segmentation masks
        >>> probs = torch.sigmoid(output)  # Apply sigmoid for probabilities
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_features=64, use_bconvlstm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Feature dimensions at each level
        f1, f2, f3, f4, f5 = base_features, base_features*2, base_features*4, base_features*8, base_features*16
        # f1=64, f2=128, f3=256, f4=512, f5=1024
        
        # ========== ENCODER ==========
        self.enc1 = EncoderBlock(in_channels, f1, pool=True)   # 512 -> 256
        self.enc2 = EncoderBlock(f1, f2, pool=True)            # 256 -> 128
        self.enc3 = EncoderBlock(f2, f3, pool=True)            # 128 -> 64
        self.enc4 = EncoderBlock(f3, f4, pool=True)            # 64 -> 32
        
        # ========== BRIDGE (Bottom) ==========
        self.bridge = nn.Sequential(
            nn.Conv2d(f4, f5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f5),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5, f5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f5),
            nn.ReLU(inplace=True),
        )
        
        # Dense block at bridge
        self.bridge_dense = DenseBlock(f5, growth_rate=32, num_layers=3)
        bridge_out = self.bridge_dense.out_channels  # 1024 + 3*32 = 1120
        
        # Reduce channels after dense block
        self.bridge_reduce = nn.Conv2d(bridge_out, f5, kernel_size=1)
        
        # ========== DECODER ==========
        self.dec4 = DecoderBlock(f5, f4, f4, use_bconvlstm=use_bconvlstm)  # 32 -> 64
        self.dec3 = DecoderBlock(f4, f3, f3, use_bconvlstm=use_bconvlstm)  # 64 -> 128
        self.dec2 = DecoderBlock(f3, f2, f2, use_bconvlstm=use_bconvlstm)  # 128 -> 256
        self.dec1 = DecoderBlock(f2, f1, f1, use_bconvlstm=use_bconvlstm)  # 256 -> 512
        
        # ========== OUTPUT ==========
        self.output_conv = nn.Conv2d(f1, out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ BCDU-Net initialized:")
        print(f"  Input channels: {in_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Base features: {base_features}")
        print(f"  Use BConvLSTM: {use_bconvlstm}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def _init_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through BCDU-Net.
        
        Args:
            x (Tensor): Input images (B, C, H, W)
                       For mammograms: (B, 1, 512, 512)
        
        Returns:
            Tensor: Segmentation logits (B, out_channels, H, W)
                   Apply sigmoid for binary segmentation probabilities
        """
        # ========== ENCODER ==========
        skip1, x = self.enc1(x)  # skip1: (B, 64, 512, 512), x: (B, 64, 256, 256)
        skip2, x = self.enc2(x)  # skip2: (B, 128, 256, 256), x: (B, 128, 128, 128)
        skip3, x = self.enc3(x)  # skip3: (B, 256, 128, 128), x: (B, 256, 64, 64)
        skip4, x = self.enc4(x)  # skip4: (B, 512, 64, 64), x: (B, 512, 32, 32)
        
        # ========== BRIDGE ==========
        x = self.bridge(x)       # (B, 1024, 32, 32)
        x = self.bridge_dense(x) # (B, 1120, 32, 32)
        x = self.bridge_reduce(x)# (B, 1024, 32, 32)
        
        # ========== DECODER ==========
        x = self.dec4(x, skip4)  # (B, 512, 64, 64)
        x = self.dec3(x, skip3)  # (B, 256, 128, 128)
        x = self.dec2(x, skip2)  # (B, 128, 256, 256)
        x = self.dec1(x, skip1)  # (B, 64, 512, 512)
        
        # ========== OUTPUT ==========
        output = self.output_conv(x)  # (B, 1, 512, 512)
        
        return output
    
    def predict(self, x, threshold=0.5):
        """
        Convenience method for inference.
        
        Args:
            x (Tensor): Input images
            threshold (float): Probability threshold for binary mask
        
        Returns:
            Tensor: Binary segmentation mask (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()


class BCDUNetLite(nn.Module):
    """
    Lightweight version of BCDU-Net for faster training and inference.
    
    Reduces parameters by:
    - Halving base features (32 instead of 64)
    - Simpler BConvLSTM implementation
    - Fewer dense connections
    
    Args:
        in_channels (int): Number of input channels (default: 1)
        out_channels (int): Number of output channels (default: 1)
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Smaller feature dimensions
        f1, f2, f3, f4, f5 = 32, 64, 128, 256, 512
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, f1, pool=True)
        self.enc2 = EncoderBlock(f1, f2, pool=True)
        self.enc3 = EncoderBlock(f2, f3, pool=True)
        self.enc4 = EncoderBlock(f3, f4, pool=True)
        
        # Bridge (simplified)
        self.bridge = nn.Sequential(
            nn.Conv2d(f4, f5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f5),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5, f5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f5),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (without BConvLSTM for speed)
        self.dec4 = DecoderBlock(f5, f4, f4, use_bconvlstm=False)
        self.dec3 = DecoderBlock(f4, f3, f3, use_bconvlstm=False)
        self.dec2 = DecoderBlock(f3, f2, f2, use_bconvlstm=False)
        self.dec1 = DecoderBlock(f2, f1, f1, use_bconvlstm=False)
        
        # Output
        self.output_conv = nn.Conv2d(f1, out_channels, kernel_size=1)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ BCDU-Net Lite initialized: {total_params:,} parameters")
    
    def forward(self, x):
        """Forward pass through lightweight BCDU-Net."""
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        return self.output_conv(x)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice coefficient measures overlap between prediction and ground truth.
    Dice Loss = 1 - Dice Coefficient
    
    Formula:
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        Loss = 1 - Dice
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1.0)
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Compute Dice loss.
        
        Args:
            pred (Tensor): Predicted logits (B, 1, H, W)
            target (Tensor): Ground truth masks (B, 1, H, W)
        
        Returns:
            Tensor: Scalar Dice loss
        """
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross Entropy Loss.
    
    Combines the benefits of both losses:
    - BCE: Good for pixel-wise accuracy
    - Dice: Good for handling class imbalance
    
    Args:
        dice_weight (float): Weight for Dice loss (default: 0.5)
        bce_weight (float): Weight for BCE loss (default: 0.5)
    """
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Compute combined loss.
        
        Args:
            pred (Tensor): Predicted logits
            target (Tensor): Ground truth masks
        
        Returns:
            Tensor: Combined loss value
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.
    
    Tversky index is a generalization of Dice that allows controlling
    the trade-off between false positives and false negatives.
    
    - alpha > beta: Penalize false negatives more (find all tumors)
    - alpha < beta: Penalize false positives more (precise boundaries)
    - alpha = beta = 0.5: Equivalent to Dice loss
    
    For tumor segmentation, alpha=0.7, beta=0.3 works well
    (we want to find all tumors, even if some false positives).
    
    Args:
        alpha (float): Weight for false negatives (default: 0.7)
        beta (float): Weight for false positives (default: 0.3)
        smooth (float): Smoothing factor (default: 1.0)
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false negatives, false positives
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Best for medical image segmentation.
    
    Combines Tversky loss with focal weighting to focus on hard examples.
    The gamma parameter controls how much to focus on hard examples.
    
    This loss is specifically designed for:
    - Small object segmentation (tumors)
    - Highly imbalanced datasets
    - Cases where Dice loss plateaus
    
    Args:
        alpha (float): Weight for false negatives (default: 0.7)
        beta (float): Weight for false positives (default: 0.3)
        gamma (float): Focal parameter (default: 0.75, range 1-3)
        smooth (float): Smoothing factor (default: 1.0)
    """
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false negatives, false positives
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal weighting
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class CombinedLoss(nn.Module):
    """
    Combined Focal Tversky + BCE Loss - Recommended for tumor segmentation.
    
    This combination provides:
    - Focal Tversky: Handles class imbalance and small objects
    - BCE: Provides stable gradients and pixel-wise accuracy
    
    Args:
        tversky_weight (float): Weight for Focal Tversky (default: 0.7)
        bce_weight (float): Weight for BCE (default: 0.3)
        alpha (float): Tversky alpha (default: 0.7)
        beta (float): Tversky beta (default: 0.3)
        gamma (float): Focal gamma (default: 0.75)
    """
    
    def __init__(self, tversky_weight=0.7, bce_weight=0.3, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        ft_loss = self.focal_tversky(pred, target)
        bce_loss = self.bce(pred, target)
        return self.tversky_weight * ft_loss + self.bce_weight * bce_loss


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    """Test BCDU-Net implementation."""
    print("Testing BCDU-Net...")
    
    # Test standard BCDU-Net
    model = BCDUNet(in_channels=1, out_channels=1, use_bconvlstm=True)
    x = torch.randn(2, 1, 512, 512)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test inference
    mask = model.predict(x)
    print(f"Prediction shape: {mask.shape}")
    print(f"Unique values: {torch.unique(mask)}")
    
    # Test loss functions
    print("\n--- Testing Loss Functions ---")
    target = torch.randint(0, 2, (2, 1, 512, 512)).float()
    
    dice_loss = DiceLoss()
    dice_bce_loss = DiceBCELoss()
    
    d_loss = dice_loss(output, target)
    db_loss = dice_bce_loss(output, target)
    
    print(f"Dice Loss: {d_loss.item():.4f}")
    print(f"Dice+BCE Loss: {db_loss.item():.4f}")
    
    print("\n✓ All tests passed!")

