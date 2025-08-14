# Sentinel-1 SAR Super-Resolution for Korean Disaster Monitoring

A comprehensive deep learning solution for 4× super-resolution of dual-polarimetric Sentinel-1 SAR imagery, specifically designed for Korean disaster monitoring applications including flood mapping and landslide detection.

## 🎯 Project Overview

This project implements state-of-the-art deep learning models to enhance the spatial resolution of Sentinel-1 SAR images from approximately 20m to 5m pixel spacing, preserving both amplitude and phase information critical for disaster monitoring applications.

### Key Features

- **Dual-Polarimetric Processing**: Full support for VV+VH polarization channels with phase preservation
- **Physical Degradation Models**: Research-driven HR→LR synthesis with realistic PSF and noise simulation
- **Advanced Model Architecture**: AC-Swin-UNet++ with phase-safe complex operations and artifact mitigation
- **Physical Loss Framework**: Comprehensive loss system including magnitude, phase, coherence, and spectral constraints
- **GPU-Accelerated Pipeline**: CUDA-optimized training with mixed precision and performance profiling
- **Research Integration**: Direct implementation of 연구질문.md research objectives and constraints

## 🏗️ Architecture

### Primary Model: AC-Swin-UNet++ (연구질문.md-Aligned)
Our main super-resolution model addresses key research questions with advanced techniques:

- **Phase-Safe Complex Operations**: All layers preserve complex phase equivariance (연구질문.md §B)
- **Swin Transformer Blocks**: Long-range dependencies with shifted windows (8×4) and magnitude-renormalized attention
- **Dense Skip Connections**: U-Net++ architecture for multi-scale feature fusion with residual scaling (0.3×output + 0.7×residual)
- **Attention Mechanisms**: Complex SE attention (with zero-channel fix) + spatial attention
- **Resize-Based Upsampling**: Bilinear interpolation + convolution to eliminate checkerboard artifacts

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

3. **Physical Degradation System** (`model/degradations.py`):
   - **Research-Driven LR Synthesis**: Addresses 연구질문.md question 1 (HR→LR 열화 모델)
   - **Forward Operator H**: Gaussian/Sinc PSF convolution with reflect padding
   - **Decimation D**: Integer-scale spatial subsampling after anti-aliasing
   - **Physical Noise Simulation** (연구질문.md §4.3):
     - Multiplicative speckle: Gamma(L,L) with adjustable ENL
     - Thermal noise: Complex Gaussian with configurable noise floor
   - **Metadata-Driven Caching**: Ensures parameter consistency across training runs

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

### 🔄 Active Research: Phase Processing Stability (연구질문.md §3)

**Research Question**: "Complex network 위상 정보의 학습 불안정성을 어떻게 처리할 것인지?"

**Major Refactoring Completed**:
- **Physical Loss Framework**: Magnitude, phase, coherence, spectral, and data consistency losses
- **Architecture Improvements**: Complex SE fix, resize upsampling, residual scaling optimization
- **Degradation Models**: Realistic PSF + noise simulation replacing simplistic averaging
- **Performance Optimization**: bfloat16 AMP, metadata caching, profiling integration

**Current Solutions** (Version 6):
- Phase-safe complex operations throughout the network
- Balanced residual connections (0.3×output + 0.7×residual) for stability
- Magnitude-renormalized attention for complex-valued features
- Comprehensive physical loss system addressing all research objectives

### 🎯 Research Roadmap (연구질문.md-Driven)
- [ ] **Conditional SR**: Investigate DEM/LC conditioning for disaster-specific applications
- [ ] **Alternative Activations**: Experiment with ComplexLeakyReLU vs ComplexGELU for phase preservation
- [ ] **Disaster Task Specialization**: Focus on landslide vs ground subsidence applications
- [ ] **Data Consistency**: Implement forward model constraints for self-supervised training
- [ ] **Korean Dataset Expansion**: Additional coverage for diverse terrain and disaster scenarios

## 🛠️ Quick Start

### Training a Model
```bash
# Basic training with physical degradation (연구질문.md compliant)
python model/train.py --lr-mode complex_lp --lp-kind gaussian --lp-sigma 1.2 --enl 10.0 --noise-std 0.01

# Physical loss framework training
python model/train.py --w-mag 1.0 --w-phase 1.0 --w-coh 0.1 --w-spec 0.05 --w-dc 0.1

# Performance-optimized training with auto-adjustment
python model/train.py --batch-size-auto --auto-workers --amp bf16

# Development: quick validation with synthetic data
python model/train.py --tiny --num-epochs 1 --dry-run
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
# Process single patch with latest model
python workflows/SR_apply.py --input patch.npy --output sr_patch.npy --model model_weights/version6/acswin_unet_pp.pth

# Batch processing with optimized model
python workflows/SR_apply.py --input-dir data/patches/LR --output-dir results/SR --model model_weights/version6/acswin_unet_pp.pth
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
- **Loss Function (A-constraints)**: sum of
  - Data Consistency: ||Down_H(û) − y||₁ (optional)
  - Magnitude: L1(log|S_SR|, log|S_HR|)
  - Phase: circular MAE after global phase align
  - Coherence: 1 - |γ| (local window)
  - Spectral band: ∥(1−M)⊙F{û}∥² with elliptical mask M
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