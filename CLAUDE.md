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
# Extract patches from SAR data
python workflows/patch_extractor_gpu_enhanced.py

# Apply super-resolution to patches
python workflows/SR_apply.py

# Convert numpy arrays to PNG for visualization
python workflows/npy2png.py
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
1. **SNAP Preprocessing** (`data/graph*.xml`): Orbit correction, calibration, TOPSAR splitting
2. **Patch Extraction** (`workflows/patch_extractor_gpu_enhanced.py`): 
   - Dual-pol complex patch extraction with quality control
   - Cross-pol coherence validation
   - GPU acceleration with CuPy support
   - Resume capability with 80% completion threshold
3. **Data Loading** (`model/data_cache.py`): Efficient caching and LR degradation

### Key Data Formats
- **Input Patches**: (2, H, W) complex64 arrays representing [VV, VH] polarizations
- **Model I/O**: 4-channel real tensors [VV-Re, VV-Im, VH-Re, VH-Im] for PyTorch compatibility
- **Training Data**: HR patches (512x256) with synthetic LR patches (128x64) via degradation

## Important Implementation Details

### Model Selection
- **Current Default**: AC-Swin-UNet++ (`--model-type swin`) - recommended for production
- **Legacy Model**: Complex U-Net has channel mismatch issues and is not supported in current training

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
- Default data directory: `D:\Sentinel-1\data\patches\zero_filtered`
- Model save path: `D:\Sentinel-1\model\acswin_unet_pp.pth`  
- Batch size: 32 (auto-adjustable)
- Learning rate: 1e-4 with cosine annealing
- Early stopping: 10 epochs patience

### Data Processing Configuration
Patch extraction parameters in `workflows/patch_extractor_gpu_enhanced.py`:
- Patch dimensions: 256x512 (width x height)
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

## Performance Targets

### Quality Metrics (SAR Super-Resolution)
- **PSNR > 30 dB**: Suitable for disaster monitoring applications  
- **Cross-pol Coherence > 0.8**: Good preservation of polarimetric information
- **Phase RMSE < 0.5 rad**: Excellent phase reconstruction quality

### Training Efficiency
- **Memory Usage**: ~8-12GB GPU memory for batch size 32
- **Training Speed**: ~0.66 patches/second processing rate
- **Convergence**: Typically converges within 50-100 epochs