# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Sentinel-1 SAR super-resolution project that implements deep learning models for 4x upsampling of dual-polarimetric (VV+VH) SAR imagery. The project is designed specifically for Korean disaster monitoring applications and includes comprehensive data processing, model training, and evaluation capabilities.

## Common Commands

### Training Commands
```bash
# Basic model training (recommended)
python model/train.py

# Training with specific model architecture
python model/train.py --model-type swin

# Training with custom batch size and workers
python model/train.py --batch-size-auto --auto-workers

# Training with limited samples for testing
python model/train.py --max-samples 1000

# Disable perceptual loss for faster training
python model/train.py --no-perceptual

# Debug mode with profiling
python model/train.py --profile 100 --log-level DEBUG
```

### TensorBoard Monitoring
```bash
# Launch TensorBoard for latest experiment
python model/visualize_tensorboard.py --launch

# Export training plots
python model/visualize_tensorboard.py --export-plots

# Analyze training logs
python model/visualize_tensorboard.py --analyze
```

### Data Processing
```bash
# Extract patches from SAR data (interactive notebook method - recommended)
# Use workflows/patch_extract_v2.ipynb for quality-controlled extraction

# Alternative: GPU-accelerated extractor (legacy)
python workflows/patch_extractor_gpu_enhanced.py

# Apply super-resolution to patches (single file) - using latest model
python workflows/SR_apply.py --input patch.npy --output sr_patch.npy --model model_weights/version5/acswin_unet_pp.pth

# Apply super-resolution to directory - using latest model
python workflows/SR_apply.py --input-dir data/patches/LR --output-dir results/SR --model model_weights/version5/acswin_unet_pp.pth

# Generate LR patches from HR patches
python workflows/degrade_patches.py --hr-root data/patches/zero_filtered --lr-output data/patches/LR --num-patches 100

# Convert numpy arrays to PNG for visualization (VV polarization)
python workflows/npy2png.py --input patch.npy --output patch_vv.png --polarization VV

# Convert numpy arrays to PNG for visualization (VH polarization)
python workflows/npy2png.py --input patch.npy --output patch_vh.png --polarization VH
```

### Model Testing
```bash
# Test model architectures
python model/cv_unet.py
python model/ac_swin_unet_pp.py

# Dry run to test model loading
python model/train.py --dry-run
```

## Project Architecture

### Core Model Implementation
- **AC-Swin-UNet++** (`model/ac_swin_unet_pp.py`): Main super-resolution model featuring:
  - Complex-valued convolutions for phase-preserving SAR processing
  - Shifted-window Swin Transformer blocks for long-range dependencies
  - Dense skip connections (U-Net++) for multi-scale feature fusion
  - Complex SE + Spatial attention mechanisms
  - Complex PixelShuffle for learnable upsampling

- **Complex U-Net** (`model/cv_unet.py`): Baseline model with:
  - Traditional U-Net architecture adapted for complex SAR data
  - VH attention mechanism for cross-polarization guidance
  - Residual connections for training stability

### Training System (`model/train.py`)
- **Hybrid Loss Function**: Combines amplitude MSE, phase L1, and optional perceptual loss
- **Dual-Pol Metrics**: Separate PSNR/SSIM calculation for VV and VH polarizations
- **Advanced Features**:
  - Mixed precision training (AMP) for memory efficiency  
  - Early stopping with best weight restoration
  - Comprehensive TensorBoard logging with SAR-specific visualizations
  - GPU-accelerated data degradation simulation
  - Automatic batch size adjustment and worker optimization

### Data Processing Pipeline
1. **SNAP Preprocessing** (`data/final.xml`): Complete processing graph including:
   - Apply Orbit File (Sentinel Precise Auto Download)
   - Radiometric Calibration (complex output, VV+VH polarizations) 
   - TOPSAR Split (IW1, IW2, IW3 subswaths)
   - TOPSAR Deburst and Merge (seamless subswath combination)
2. **Patch Extraction** (`workflows/patch_extract_v2.ipynb`): Interactive notebook for:
   - Quality-controlled patch extraction from SNAP-processed data
   - Dual-pol complex patch generation with cross-pol coherence validation
   - Zero-value filtering and statistical analysis
   - Output to `data/patches/zero_filtered/` directory
3. **Data Loading** (`model/data_cache.py`): Efficient caching and LR degradation from filtered patches

### Key Data Formats
- **Input Patches**: (2, H, W) complex64 arrays representing [VV, VH] polarizations
- **Model I/O**: 4-channel real tensors [VV-Re, VV-Im, VH-Re, VH-Im] for PyTorch compatibility
- **Training Data**: HR patches (512x256) with synthetic LR patches (128x64) via degradation
- **SR Pipeline**: complex64 (2,128,64) → model → complex64 (2,512,256) for 4x super-resolution

## Important Implementation Details

### Model Selection
- **Current Default**: AC-Swin-UNet++ (`--model-type swin`) - recommended for production
- **Model Creation**: Always use `create_model()` factory function from `ac_swin_unet_pp.py`
- **Legacy Model**: Complex U-Net has channel mismatch issues and is not supported in current training
- **Active Model Path**: `model_weights/version5/acswin_unet_pp.pth` (latest with artifact mitigation)

### SAR-Specific Considerations
- **Complex Data Handling**: All models preserve both amplitude and phase information
- **Cross-Pol Analysis**: VH polarization provides texture guidance for VV reconstruction
- **Quality Metrics**: Uses CPIF (Complex Peak Intensity Factor) and phase RMSE for SAR evaluation

### Memory and Performance
- **GPU Acceleration**: CUDA-optimized training with mixed precision
- **Data Caching**: LR patches cached to disk to avoid repeated degradation computation
- **Batch Processing**: Configurable batch sizes with automatic GPU memory adjustment

### Directory Structure
- `model/`: Core training code and model definitions
- `workflows/`: Data processing and visualization scripts  
- `data/`: SAR datasets and processing configurations
- `runs/`: TensorBoard logs organized by timestamp
- `results/`: Model outputs and comparison visualizations
- `model_weights/`: Saved model checkpoints organized by version

## Configuration Files

### Training Configuration
Training parameters are configured via command-line arguments in `model/train.py`. Key settings:
- **Data directory**: `D:\Sentinel-1\data\patches\zero_filtered` (quality-filtered patches from SNAP+notebook workflow)
- Model save path: `D:\Sentinel-1\model_weights/version5/acswin_unet_pp.pth` (latest)
- Backup save path: `D:\Sentinel-1\model\acswin_unet_pp.pth` (legacy compatibility)
- Batch size: 32 (auto-adjustable)
- Learning rate: 1e-4 with cosine annealing
- Early stopping: 10 epochs patience
- **Artifact Monitoring**: Enhanced logging for checkerboard pattern detection

### Data Processing Configuration
**Primary Method** - SNAP + Notebook workflow:
1. `data/final.xml`: Complete SNAP processing graph (orbit, calibration, TOPSAR processing)
2. `workflows/patch_extract_v2.ipynb`: Interactive patch extraction with quality control
   - Output directory: `data/patches/zero_filtered/`
   - Patch dimensions: 256x512 (width x height)
   - Quality filtering: Zero-value removal and cross-pol coherence validation
   - Statistical analysis and visualization included

**Alternative Method** - Direct extraction:
- `workflows/patch_extractor_gpu_enhanced.py`: GPU-accelerated processing
- Stride: non-overlapping (256x512)
- Quality threshold: cross-pol coherence > 0.01

## Development Workflow

### Setting up Training
1. Ensure SAR data is processed and available in patch format
2. Adjust data paths in training configuration if needed  
3. Start training with TensorBoard monitoring
4. Monitor training progress through TensorBoard dashboard
5. Evaluate model performance using built-in metrics

### Model Development
- New models should inherit from PyTorch nn.Module
- Implement `count_parameters()` method for logging
- Support 4-channel real input/output format
- Include `create_model()` factory function

### Data Pipeline
- SAR patches should be complex64 format with shape (2, H, W)
- Quality control through cross-pol coherence calculation
- Maintain consistent file naming: `*_dual_pol_complex_{x}_{y}.npy`
- **LR Degradation**: Uses block-wise amplitude averaging + phase averaging for realistic degradation
- **Data Caching**: LR patches cached automatically in `lr_cache/` subdirectories

## Performance Targets

### Quality Metrics (SAR Super-Resolution)
- **PSNR > 30 dB**: Suitable for disaster monitoring applications  
- **Cross-pol Coherence > 0.8**: Good preservation of polarimetric information
- **Phase RMSE < 0.5 rad**: Excellent phase reconstruction quality

### Training Efficiency
- **Memory Usage**: ~8-12GB GPU memory for batch size 32
- **Training Speed**: ~0.66 patches/second processing rate
- **Convergence**: Typically converges within 50-100 epochs

## Critical Implementation Notes

### Data Handling
- **Complex Data**: All SAR data must be complex64 format - workflows now validate and convert automatically
- **TensorBoard Fix**: LR visualization now uses bilinear upsampling instead of nearest neighbor for smoother display
- **Shape Validation**: Training pipeline validates HR patches are (512,256) and LR patches are (128,64)

### Workflow Compatibility
- **SR_apply.py**: Fixed to properly handle complex64 input and use `create_model()` function
- **npy2png.py**: Enhanced to correctly identify and process complex SAR data vs real data
- **Model Loading**: Always use model factory functions, never instantiate model classes directly

### Checkerboard Artifact Mitigation (Current Priority)
- **Issue**: Models version 4-5 exhibit checkerboard artifacts in super-resolution outputs
- **Causes**: Potential issues with PixelShuffle upsampling, loss function weighting, or training dynamics
- **Current Approaches**:
  - Modified Complex PixelShuffle implementation with improved weight initialization
  - Adjusted hybrid loss function weighting (amplitude MSE vs phase L1)
  - Enhanced data augmentation strategies
  - Regularization techniques to prevent high-frequency artifacts
- **Model Versions**: 
  - Version 4: Initial checkerboard artifact identification
  - Version 5: Latest mitigation attempts (ongoing)
- **Monitoring**: TensorBoard visualization enhanced to detect and analyze artifact patterns

### Recent Updates and Fixes
- **Commit 1258390**: Latest checkerboard artifact mitigation strategies implemented
- **Commit 8f6964f**: Enhanced TensorBoard visualization for artifact analysis
- **Commit 8d6699c**: Modified loss functions and visualization improvements
- **Training Status**: Ongoing experiments with different architectural modifications

### Common Issues and Solutions
- **Shape Mismatch**: Clear LR cache if changing HR/LR dimensions: `rm -rf data/patches/*/lr_cache/`
- **Model Import**: Use `from ac_swin_unet_pp import create_model` not direct class import
- **Complex Data**: Ensure input data is complex64, not float32 - workflows now auto-detect and convert
- **Checkerboard Artifacts**: Use latest model weights from version5 directory, monitor TensorBoard for artifact patterns
- **Training Convergence**: If artifacts appear during training, consider reducing learning rate or adjusting loss weights