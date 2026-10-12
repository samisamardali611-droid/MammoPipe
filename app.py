#!/usr/bin/env python3
"""
MammoPipe Web Application
=========================

Streamlit web interface for mammography analysis with classification and segmentation.

Usage:
------
    streamlit run app.py
"""

import sys
from pathlib import Path
import tempfile

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model import ImprovedHybridV15
from bcdu_net import BCDUNet

# Page configuration
st.set_page_config(
    page_title="MammoPipe - Mammography Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>

    /* ===== PAGE BACKGROUND ===== */
    .stApp {
        background-color: #ffc2e0;
    }

    /* ===== MAIN TEXT ===== */
    [data-testid="stMarkdown"],
    [data-testid="stText"],
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] li {
        color: #3a0f23 !important;
    }

    /* ===== HEADERS / TITLES ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #7a294a !important;
    }

    /* ===== ICONS (SVG + EMOJI) ===== */
    svg {
        fill: #7a294a !important;
        color: #7a294a !important;
    }

    /* ===== BUTTONS ===== */
    button {
        background-color: #7a294a !important;
        color: white !important;
        border-radius: 10px;
        border: none;
    }

    button:hover {
        background-color: #5c1f38 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


class MammoPipeApp:
    """Streamlit application for mammography analysis."""

    def __init__(self):
        """Initialize the app and load models."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model paths
        self.classifier_path = 'checkpoints/best_resnet101_swinv2.ptrom'
        self.segmentation_path = 'checkpoints/best_bcdu_segmentation.pth'

        # Load models (cached)
        self.classifier = self._load_classifier()
        self.segmentation = self._load_segmentation()

    @st.cache_resource
    def _load_classifier(_self):
        """Load classification model (cached)."""
        checkpoint = torch.load(
            _self.classifier_path,
            map_location=_self.device,
            weights_only=False
        )

        model = ImprovedHybridV15(num_classes=2, dropout=0.3, pretrained=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(_self.device)
        model.eval()

        return model

    @st.cache_resource
    def _load_segmentation(_self):
        """Load segmentation model (cached)."""
        checkpoint = torch.load(
            _self.segmentation_path,
            map_location=_self.device,
            weights_only=False
        )

        config = checkpoint.get('config', {})

        model = BCDUNet(
            in_channels=1,
            out_channels=1,
            base_features=config.get('base_features', 64),
            use_bconvlstm=config.get('use_bconvlstm', True)
        )

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(_self.device)
        model.eval()

        return model

    def preprocess(self, image):
        """
        Preprocess PIL image for both models.

        Args:
            image: PIL Image (RGB or grayscale)

        Returns:
            tuple: (original_array, cls_tensor, seg_tensor)
        """
        # Convert to grayscale numpy array
        if image.mode == 'RGB':
            image_array = np.array(image.convert('L'))
        else:
            image_array = np.array(image)

        original = image_array.copy()

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_array)

        # Prepare for classifier (256x256, 3-channel)
        cls_img = cv2.resize(enhanced, (256, 256)).astype(np.float32) / 255.0
        cls_tensor = torch.from_numpy(cls_img).unsqueeze(0).repeat(3, 1, 1)
        cls_tensor = cls_tensor.unsqueeze(0).to(self.device)

        # Prepare for segmentation (512x512, 1-channel)
        seg_img = cv2.resize(enhanced, (512, 512)).astype(np.float32) / 255.0
        seg_tensor = torch.from_numpy(seg_img).unsqueeze(0).unsqueeze(0)
        seg_tensor = seg_tensor.to(self.device)

        return original, cls_tensor, seg_tensor

    @torch.no_grad()
    def predict(self, image, seg_threshold=0.5):
        """
        Run inference on uploaded image.

        Args:
            image: PIL Image
            seg_threshold: Threshold for segmentation mask

        Returns:
            dict: Prediction results
        """
        # Preprocess
        original, cls_input, seg_input = self.preprocess(image)
        original_size = original.shape[:2]

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

        # Resize to original size
        seg_probs_orig = cv2.resize(seg_probs, (original_size[1], original_size[0]))
        seg_mask_orig = cv2.resize(seg_mask, (original_size[1], original_size[0]),
                                    interpolation=cv2.INTER_NEAREST)

        # Compute tumor statistics
        tumor_area = (seg_mask_orig > 0).sum()
        total_area = seg_mask_orig.size
        tumor_ratio = tumor_area / total_area

        # Find bounding box
        contours, _ = cv2.findContours(seg_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest)

        return {
            'original': original,
            'classification': {
                'prediction': prediction,
                'confidence': confidence,
                'benign_probability': benign_prob,
                'malignant_probability': malignant_prob
            },
            'segmentation': {
                'probabilities': seg_probs_orig,
                'mask': seg_mask_orig,
                'tumor_detected': tumor_ratio > 0.001,
                'tumor_area_pixels': int(tumor_area),
                'tumor_area_ratio': float(tumor_ratio),
                'bounding_box': bbox
            }
        }

    def create_visualization(self, result, display_mode):
        """
        Create visualization based on display mode.

        Args:
            result: Prediction results
            display_mode: 'both', 'classification', or 'segmentation'

        Returns:
            matplotlib figure
        """
        original = result['original']
        probs = result['segmentation']['probabilities']
        mask = result['segmentation']['mask']
        cls = result['classification']
        seg = result['segmentation']

        if display_mode == 'classification':
            # Classification only - show original with classification result
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            ax.imshow(original, cmap='gray')
            ax.axis('off')

            # Add classification text
            color = 'red' if cls['prediction'] == 'MALIGNANT' else 'green'
            title = f"{cls['prediction']}\nConfidence: {cls['confidence']*100:.1f}%"
            ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)

            plt.tight_layout()
            return fig

        elif display_mode == 'segmentation':
            # Segmentation only - show original, heatmap, mask, and overlay
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Original
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # Probability heatmap
            axes[1].imshow(original, cmap='gray', alpha=0.7)
            heatmap = axes[1].imshow(probs, cmap='hot', alpha=0.5, vmin=0, vmax=1)
            axes[1].set_title('Tumor Probability', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

            # Binary mask
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title(f'Tumor Mask\nArea: {seg["tumor_area_ratio"]*100:.2f}%',
                            fontsize=12, fontweight='bold')
            axes[2].axis('off')

            # Overlay with bounding box
            overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = mask
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

            axes[3].imshow(overlay)

            if seg['bounding_box']:
                x, y, w, h = seg['bounding_box']
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                axes[3].add_patch(rect)
                axes[3].text(x, y - 10, 'Tumor', color='lime', fontsize=10, fontweight='bold')

            axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
            axes[3].axis('off')

            plt.tight_layout()
            return fig

        else:  # 'both'
            # Complete visualization with both classification and segmentation
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(2, 4, height_ratios=[3, 1], hspace=0.3, wspace=0.2)

            # Row 1: Images
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[0, 3])

            # Row 2: Metrics
            ax5 = fig.add_subplot(gs[1, :2])
            ax6 = fig.add_subplot(gs[1, 2:])

            # Original
            ax1.imshow(original, cmap='gray')
            ax1.set_title('Original Mammogram', fontsize=12, fontweight='bold')
            ax1.axis('off')

            # Probability heatmap
            ax2.imshow(original, cmap='gray', alpha=0.7)
            heatmap = ax2.imshow(probs, cmap='hot', alpha=0.5, vmin=0, vmax=1)
            ax2.set_title('Tumor Probability', fontsize=12, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)

            # Binary mask
            ax3.imshow(mask, cmap='gray')
            ax3.set_title(f'Tumor Mask\nArea: {seg["tumor_area_ratio"]*100:.2f}%',
                         fontsize=12, fontweight='bold')
            ax3.axis('off')

            # Overlay with bounding box
            overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = mask
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

            ax4.imshow(overlay)

            if seg['bounding_box']:
                x, y, w, h = seg['bounding_box']
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                ax4.add_patch(rect)
                ax4.text(x, y - 10, 'Tumor', color='lime', fontsize=10, fontweight='bold')

            color = 'red' if cls['prediction'] == 'MALIGNANT' else 'green'
            ax4.set_title(f'{cls["prediction"]}\nConfidence: {cls["confidence"]*100:.1f}%',
                         fontsize=12, fontweight='bold', color=color)
            ax4.axis('off')

            # Classification metrics
            ax5.axis('off')
            cls_text = f"""
╔══════════════════════════════════════════╗
║           CLASSIFICATION RESULTS         ║
╠══════════════════════════════════════════╣
║  Prediction:    {cls['prediction']:<24} ║
║  Confidence:    {cls['confidence']*100:>6.1f}%                  ║
║                                          ║
║  Benign:        {cls['benign_probability']*100:>6.1f}%                  ║
║  Malignant:     {cls['malignant_probability']*100:>6.1f}%                  ║
╚══════════════════════════════════════════╝
"""
            ax5.text(0.1, 0.5, cls_text, transform=ax5.transAxes, fontsize=11,
                    fontfamily='monospace', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax5.set_title('Classification Metrics', fontsize=12, fontweight='bold')

            # Segmentation metrics
            ax6.axis('off')
            tumor_status = "YES" if seg['tumor_detected'] else "NO"
            bbox_str = f"({seg['bounding_box'][0]}, {seg['bounding_box'][1]}, {seg['bounding_box'][2]}, {seg['bounding_box'][3]})" if seg['bounding_box'] else "N/A"

            seg_text = f"""
╔══════════════════════════════════════════╗
║           SEGMENTATION RESULTS           ║
╠══════════════════════════════════════════╣
║  Tumor Detected:   {tumor_status:<21} ║
║  Tumor Area:       {seg['tumor_area_ratio']*100:>6.2f}%               ║
║  Tumor Pixels:     {seg['tumor_area_pixels']:>6,}               ║
║                                          ║
║  Bounding Box:     {bbox_str:<21}║
╚══════════════════════════════════════════╝
"""
            ax6.text(0.1, 0.5, seg_text, transform=ax6.transAxes, fontsize=11,
                    fontfamily='monospace', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax6.set_title('Segmentation Metrics', fontsize=12, fontweight='bold')

            fig.suptitle('MammoPipe Analysis', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            return fig


def main():
    """Main Streamlit application."""

    # Header
    st.title("🏥 MammoPipe - Mammography Analysis")
    st.markdown("""
    Upload a mammogram image to get AI-powered classification and segmentation analysis.
    - **Classification**: Detects if the mammogram is benign or malignant
    - **Segmentation**: Identifies and highlights tumor regions
    """)

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    # Display mode selector
    display_mode = st.sidebar.radio(
        "Display Mode",
        options=['both', 'classification', 'segmentation'],
        format_func=lambda x: {
            'both': '🔬 Both (Classification + Segmentation)',
            'classification': '📊 Classification Only',
            'segmentation': '🎯 Segmentation Only'
        }[x],
        index=0
    )

    # Segmentation threshold
    seg_threshold = st.sidebar.slider(
        "Segmentation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for tumor detection in segmentation"
    )

    st.sidebar.markdown("---")

    # Device info
    device = 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'
    st.sidebar.info(f"**Device**: {device}")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a mammogram image...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff' , 'dicom', 'dcm'],
        help="Upload a mammogram image in common formats (PNG, JPG, etc.)"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📤 Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("ℹ️ Image Information")
            st.write(f"**Filename**: {uploaded_file.name}")
            st.write(f"**Size**: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Mode**: {image.mode}")
            st.write(f"**Format**: {image.format}")

        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):

            with st.spinner("Loading models..."):
                try:
                    app = MammoPipeApp()
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
                    st.info("Make sure the model checkpoints exist in the 'checkpoints/' directory")
                    return

            with st.spinner("Running inference..."):
                try:
                    result = app.predict(image, seg_threshold=seg_threshold)
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
                    return

            # Display results
            st.success("✅ Analysis complete!")

            # Show metrics based on display mode
            if display_mode in ['both', 'classification']:
                cls = result['classification']

                st.subheader("📊 Classification Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    prediction_color = 'red' if cls['prediction'] == 'MALIGNANT' else 'green'
                    st.markdown(f"### :{prediction_color}[{cls['prediction']}]")
                    st.metric("Confidence", f"{cls['confidence']*100:.1f}%")

                with col2:
                    st.metric("Benign Probability", f"{cls['benign_probability']*100:.1f}%")

                with col3:
                    st.metric("Malignant Probability", f"{cls['malignant_probability']*100:.1f}%")

            if display_mode in ['both', 'segmentation']:
                seg = result['segmentation']

                st.subheader("🎯 Segmentation Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    tumor_status = "✅ YES" if seg['tumor_detected'] else "❌ NO"
                    st.metric("Tumor Detected", tumor_status)

                with col2:
                    st.metric("Tumor Area", f"{seg['tumor_area_ratio']*100:.2f}%")

                with col3:
                    st.metric("Tumor Pixels", f"{seg['tumor_area_pixels']:,}")

                with col4:
                    if seg['bounding_box']:
                        x, y, w, h = seg['bounding_box']
                        st.metric("Bounding Box", f"{w}×{h}")
                    else:
                        st.metric("Bounding Box", "N/A")

            # Visualization
            st.subheader("📈 Visualization")

            fig = app.create_visualization(result, display_mode)
            st.pyplot(fig)
            plt.close(fig)

    else:
        # Show example instructions
        st.info("👆 Upload a mammogram image to begin analysis")

        st.markdown("""
        ### How to use:
        1. Upload a mammogram image using the file uploader above
        2. Choose your preferred display mode in the sidebar:
           - **Both**: See complete analysis with classification and segmentation
           - **Classification Only**: Focus on benign/malignant prediction
           - **Segmentation Only**: Focus on tumor detection and localization
        3. Adjust the segmentation threshold if needed (default: 0.5)
        4. Click "Analyze Image" to run the AI models
        5. View the results and visualizations

        ### About the Models:
        - **Classification Model**: ResNet101 + SwinV2 hybrid model for benign/malignant classification
        - **Segmentation Model**: BCDU-Net for tumor region detection and localization
        """)


if __name__ == '__main__':
    main()
