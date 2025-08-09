# Sentinel-1 SAR Super-Resolution for Korean Disaster Monitoring

A comprehensive deep learning solution for 4× super-resolution of dual-polarimetric Sentinel-1 SAR imagery, specifically designed for Korean disaster monitoring applications including flood mapping and landslide detection.

## 🎯 Project Overview

This project implements state-of-the-art deep learning models to enhance the spatial resolution of Sentinel-1 SAR images from approximately 20m to 5m pixel spacing, preserving both amplitude and phase information critical for disaster monitoring applications.

### Key Features

- **Dual-Polarimetric Processing**: Full support for VV+VH polarization channels
- **Phase-Preserving Architecture**: Complex-valued neural networks maintaining SAR coherence
- **Advanced Model Architecture**: AC-Swin-UNet++ with shifted-window Swin Transformers
- **GPU-Accelerated Pipeline**: CUDA-optimized training and inference
- **Comprehensive Evaluation**: SAR-specific metrics including CPIF, cross-pol coherence, and phase RMSE

## 🏗️ Architecture

### Primary Model: AC-Swin-UNet++
Our main super-resolution model combines several cutting-edge techniques:

- **Complex-Valued Convolutions**: Preserves both amplitude and phase information
- **Swin Transformer Blocks**: Captures long-range dependencies with shifted windows (8×4 configuration)
- **Dense Skip Connections**: U-Net++ architecture for multi-scale feature fusion
- **Attention Mechanisms**: Complex SE attention + spatial attention for enhanced feature selection
- **Complex PixelShuffle**: Learnable upsampling that mitigates checkerboard artifacts

```python
Input:  (Batch, 4, H, W)     # [VV-Real, VV-Imag, VH-Real, VH-Imag]
Output: (Batch, 4, 4H, 4W)   # 4× super-resolved SAR imagery
```

### Data Processing Pipeline

1. **SNAP Preprocessing** (`data/final.xml`): Complete processing workflow including:
   - Apply Orbit File with Sentinel Precise orbits
   - Radiometric Calibration (complex output preserving VV+VH polarizations)
   - TOPSAR Split for all subswaths (IW1, IW2, IW3)
   - TOPSAR Deburst and Merge for seamless subswath combination

2. **Interactive Patch Extraction** (`workflows/patch_extract_v2.ipynb`):
   - Quality-controlled extraction from SNAP-processed complex data
   - Cross-polarization coherence validation and zero-value filtering
   - Statistical analysis with amplitude/phase distribution plots
   - Output to quality-filtered dataset: `data/patches/zero_filtered/`

3. **Training Data Preparation**:
   - 512×256 HR patches from filtered dataset
   - Synthetic 128×64 LR patches via realistic degradation
   - GPU-accelerated caching and augmentation during training

## 📊 Current Performance

### Model Performance Metrics
- **PSNR**: 40.59 dB (exceeds 30 dB target for disaster monitoring)
- **Cross-pol Coherence**: 0.7052 (good polarimetric preservation)
- **CPIF**: 36.34 dB (excellent complex intensity preservation)

### Regional Performance Analysis
- **Rural Areas**: 41.12 dB PSNR, 0.6904 SSIM (excellent performance)
- **Urban Areas**: 26.70 dB PSNR, 0.6202 SSIM (adequate for most applications)

## 🚧 Current Status & Recent Progress

### ✅ Completed Milestones
- [x] Complete data preprocessing pipeline with SNAP integration
- [x] AC-Swin-UNet++ model implementation and training infrastructure
- [x] TensorBoard integration with SAR-specific visualization
- [x] Comprehensive evaluation framework with disaster monitoring metrics
- [x] GPU-accelerated patch extraction and caching system

### 🔄 Active Development: Checkerboard Artifact Mitigation

**Issue**: Recent training iterations have shown checkerboard artifacts in super-resolution outputs, particularly affecting high-frequency details.

**Mitigation Efforts** (Versions 4-5):
- Modified upsampling strategy using Complex PixelShuffle
- Adjusted loss function weighting between amplitude and phase components
- Experimented with different training regularization techniques
- Enhanced data augmentation to improve model robustness

**Recent Commits**:
- `1258390`: Latest checkerboard artifact mitigation attempts
- `8f6964f`: Improved TensorBoard visualization for artifact analysis
- `8d6699c`: Loss function modifications and visualization enhancements

### 🎯 Next Steps
- [ ] Resolve checkerboard artifacts through architectural refinements
- [ ] Implement progressive training strategy for artifact reduction
- [ ] Expand dataset with additional Korean Peninsula coverage
- [ ] Deploy real-time inference pipeline for operational use

## 🛠️ Quick Start

### Training a Model
```bash
# Basic training with default AC-Swin-UNet++ model
python model/train.py

# Training with TensorBoard monitoring
python model/train.py --batch-size-auto --auto-workers

# Limited sample training for testing
python model/train.py --max-samples 1000 --no-perceptual
```

### Monitoring Training Progress
```bash
# Launch TensorBoard dashboard
python model/visualize_tensorboard.py --launch

# Export training plots
python model/visualize_tensorboard.py --export-plots
```

### Applying Super-Resolution
```bash
# Process single patch
python workflows/SR_apply.py --input patch.npy --output sr_patch.npy --model model_weights/version5/acswin_unet_pp.pth

# Batch processing
python workflows/SR_apply.py --input-dir data/patches/LR --output-dir results/SR --model model_weights/version5/acswin_unet_pp.pth
```

## 📁 Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed directory organization and file conventions.

## 🔬 Technical Details

### Data Specifications
- **Preprocessing**: SNAP-processed with `final.xml` graph (orbit correction, calibration, TOPSAR processing)
- **Extraction Method**: Interactive notebook `patch_extract_v2.ipynb` with quality control
- **Input Format**: Complex64 SAR patches (2, H, W) from `data/patches/zero_filtered/`
- **Spatial Resolution**: 512×256 HR patches → 128×64 LR patches (4× super-resolution)
- **Quality Control**: Zero-value filtering and cross-polarization coherence validation
- **Dataset**: Korean Peninsula Sentinel-1 scenes (2020-2022) with statistical analysis

### Training Configuration
- **Hardware**: CUDA-enabled GPU with 8-12GB memory
- **Batch Size**: 32 (auto-adjustable based on GPU memory)
- **Learning Rate**: 1e-4 with cosine annealing schedule
- **Loss Function**: Hybrid amplitude MSE + phase L1 + optional perceptual loss
- **Early Stopping**: 10 epochs patience with best weight restoration

### Model Variants
- **AC-Swin-UNet++**: Primary production model (recommended)
- **Complex U-Net**: Baseline comparison model
- **Legacy Models**: Previous iterations preserved for comparison

## 📈 Performance Analysis

### Strengths
- Excellent amplitude reconstruction with high PSNR values
- Well-preserved cross-polarization coherence for multi-pol analysis
- Effective speckle reduction while maintaining structural details
- Strong performance in rural/agricultural areas

### Areas for Improvement
- **Phase Reconstruction**: Higher phase RMSE may impact interferometric applications
- **Urban Performance**: Lower performance in dense urban environments
- **Artifact Issues**: Ongoing work to eliminate checkerboard artifacts
- **Texture Preservation**: Some over-smoothing in high-texture regions

### Disaster Monitoring Suitability
- **Flood Mapping**: Good performance for water boundary detection
- **Landslide Detection**: Adequate for major terrain changes
- **Agricultural Monitoring**: Excellent for crop damage assessment
- **Infrastructure Assessment**: Good for major structural damage evaluation

## 🤝 Development Workflow

### Code Organization
- **model/**: Core deep learning implementation and training
- **workflows/**: Data processing and application scripts
- **data/**: SAR datasets and preprocessing configurations
- **results/**: Model outputs and performance analysis

### Quality Assurance
- Comprehensive unit testing for data processing pipeline
- Automated performance regression testing
- TensorBoard integration for training monitoring
- Version-controlled model checkpoints with performance metadata

## 📚 References & Acknowledgments

This project builds upon state-of-the-art research in SAR image processing and deep learning super-resolution, specifically adapted for Korean disaster monitoring requirements.

### Key Technologies
- **Sentinel-1 SAR**: European Space Agency radar satellite constellation
- **SNAP**: ESA's Sentinel Application Platform for SAR preprocessing
- **PyTorch**: Deep learning framework with CUDA acceleration
- **Complex-Valued Neural Networks**: Specialized architectures for SAR data
- **Swin Transformers**: Vision transformers with shifted windows

---

*Last Updated: August 2025*  
*Status: Active Development - Checkerboard Artifact Mitigation Phase*