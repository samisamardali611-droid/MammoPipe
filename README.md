# Local trained weights

The Streamlit application expects both files in this directory:

- `best_resnet101_swinv2.ptrom` — classification
- `best_bcdu_segmentation.pth` — segmentation

These names were visible in the supplied directory screenshot. Their contents and compatibility were not checked during packaging. Weights are excluded from this source release. Do not rename the Core ResNet-50 model to the ResNet-101 filename: the architectures differ. The existing code uses `torch.load(..., weights_only=False)`, so only use checkpoints from a trusted source.
