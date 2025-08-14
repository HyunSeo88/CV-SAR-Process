# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Sentinel-1 SAR super-resolution project that implements deep learning models for 4x upsampling of dual-polarimetric (VV+VH) SAR imagery. The project is designed specifically for Korean disaster monitoring applications and includes comprehensive data processing, model training, and evaluation capabilities.

**Recent Major Refactoring (연구질문.md-based):**
The codebase has been significantly refactored to address key research questions:
1. **HR→LR Degradation Models**: New `degradations.py` implements physically realistic complex-valued low-pass filtering with optional speckle and thermal noise
2. **Phase Processing Stability**: Enhanced complex network architecture with improved phase consistency losses
3. **Physical Loss Functions**: Comprehensive loss system in `utils.py` supporting magnitude, phase, coherence, spectral, and data consistency losses
4. **Modular Architecture**: Separated concerns with dedicated modules for degradation, metrics, caching, and performance optimization

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

# Physical degradation parameters (연구질문.md §4.3)
python model/train.py --lr-mode complex_lp --lp-sigma 1.2 --enl 10.0 --noise-std 0.01

# Physical loss weighting (연구질문.md objectives)
python model/train.py --w-mag 1.0 --w-phase 1.0 --w-coh 0.1 --w-spec 0.05 --w-dc 0.1

# Disable perceptual loss for faster training
python model/train.py --perceptual-weight 0.0

# Debug mode with profiling
python model/train.py --profile 100
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
  - **Phase-Safe Processing**: All complex operations preserve phase equivariance (연구질문.md §B)
  - Complex-valued convolutions with ComplexGELU activation
  - Shifted-window Swin Transformer blocks (8×4 windows) for long-range dependencies
  - Dense skip connections (U-Net++) for multi-scale feature fusion
  - Complex SE + Spatial attention mechanisms
  - **Resize-based Upsampling**: Bilinear interpolation + convolution to avoid checkerboard artifacts
  - **Residual Scaling**: 0.3 × output + 0.7 × residual for training stability

- **Complex U-Net** (`model/cv_unet.py`): Legacy baseline model (currently deprecated due to channel mismatch)

### Training System (`model/train.py`)
- **Physical Loss Framework** (연구질문.md alignment):
  - **Magnitude Loss**: L1 on log-amplitude (SAR-appropriate dynamic range)
  - **Phase Loss**: Circular L1 with global phase alignment
  - **Coherence Loss**: 1 - |γ| interferometric coherence preservation
  - **Spectral Loss**: Frequency domain support constraint
  - **Data Consistency**: Forward model H + decimator D consistency check
- **Dual-Pol Metrics**: Separate PSNR/SSIM calculation for VV and VH polarizations
- **Advanced Features**:
  - Mixed precision training (AMP) with bfloat16 support
  - Early stopping with best weight restoration
  - Comprehensive TensorBoard logging with SAR-specific visualizations
  - **Physical Degradation**: GPU-accelerated complex low-pass + decimation
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
3. **Data Loading & Degradation System**:
   - **Data Cache** (`model/data_cache.py`): Efficient LR patch caching with metadata validation
   - **Physical Degradation** (`model/degradations.py`): Implements research-driven HR→LR synthesis:
     - **Complex Low-Pass Mode**: PSF convolution (Gaussian/Sinc) + decimation
     - **Optional Speckle**: Multiplicative Gamma(L,L) noise simulation
     - **Thermal Noise**: Additive complex Gaussian noise floor
     - **Baseline Mode**: Block-wise amplitude/phase averaging for ablation studies

### Key Data Formats & Physical Specifications
- **Input Patches**: (2, H, W) complex64 arrays representing [VV, VH] polarizations
- **Model I/O**: 4-channel real tensors [VV-Re, VV-Im, VH-Re, VH-Im] for PyTorch compatibility
- **Training Data**: HR patches (512×256) with synthetic LR patches (128×64) via physical degradation
- **SR Pipeline**: complex64 (2,128,64) → model → complex64 (2,512,256) for 4× isotropic super-resolution
- **Physical Parameters** (연구질문.md §4.3):
  - **PSF Options**: Gaussian (σ=1.2px) or Sinc (Kaiser windowed, β=12.0)
  - **ENL Range**: 1-20 for multiplicative speckle simulation
  - **Noise Floor**: 0.001-0.1 complex Gaussian thermal noise standard deviation

## Important Implementation Details

### Model Selection & Architecture Updates
- **Current Default**: AC-Swin-UNet++ (`--model-type swin`) - only supported architecture
- **Model Creation**: Always use `create_model()` factory function from `ac_swin_unet_pp.py`
- **Legacy Model**: Complex U-Net deprecated due to channel mismatch (expects 3 channels, receives 4)
- **Active Model Path**: `model_weights/version6/acswin_unet_pp.pth` (latest with artifact mitigation)
- **Architecture Improvements**:
  - **Complex SE Fix**: Prevents zero-channel issues by ensuring `max(1, c // r)` in squeeze ratio
  - **Resize Upsampling**: Replaces PixelShuffle to eliminate checkerboard artifacts
  - **Phase Continuity**: ComplexGELU with magnitude-based gating for phase preservation
  - **Stabilized Residuals**: 0.3×output + 0.7×residual weighting for training stability

### SAR-Specific Considerations & Research Integration
- **Complex Data Handling**: All models preserve both amplitude and phase information
- **Cross-Pol Analysis**: VH polarization provides texture guidance for VV reconstruction  
- **Physical Consistency** (연구질문.md compliance):
  - **Phase Equivariance**: All activations preserve complex phase relationships
  - **Interferometric Coherence**: Local coherence |γ| preservation between VV/VH
  - **Spectral Support**: Frequency domain constraints via elliptical masks
- **Quality Metrics**: Uses CPIF, phase RMSE, log-intensity PSNR/SSIM for comprehensive SAR evaluation

### Memory and Performance Optimization
- **GPU Acceleration**: CUDA-optimized training with bfloat16 mixed precision (AMP)
- **Data Caching**: LR patches cached with metadata validation to avoid repeated degradation computation
- **Batch Processing**: Auto-adjustable batch sizes with 25% conservative scaling
- **Performance Tools** (`model/speed_utils.py`):
  - Optimal DataLoader worker detection
  - GPU memory headroom monitoring
  - PyTorch profiler integration for bottleneck analysis
- **Efficient Complex Operations**: Memory-optimized complex convolutions with reflect padding

### Directory Structure
- `model/`: Core training code and model definitions
  - `train.py`: Main training script with comprehensive CLI options
  - `ac_swin_unet_pp.py`: Primary model architecture (AC-Swin-UNet++)
  - `utils.py`: Loss functions and evaluation metrics (연구질문.md-aligned)
  - `degradations.py`: Physical HR→LR degradation models
  - `data_cache.py`: Efficient LR patch caching system
  - `speed_utils.py`: Performance optimization utilities
- `workflows/`: Data processing and visualization scripts  
- `data/`: SAR datasets and processing configurations
- `runs/`: TensorBoard logs organized by timestamp
- `results/`: Model outputs and comparison visualizations
- `model_weights/`: Saved model checkpoints organized by version

## Configuration Files

### Training Configuration (CLI-Based)
Training parameters are configured via command-line arguments in `model/train.py`. Key settings:
- **Data directory**: `D:\Sentinel-1\data\patches\zero_filtered` (quality-filtered patches from SNAP+notebook workflow)
- **Model save path**: `D:\Sentinel-1\model_weights/version6/acswin_unet_pp.pth` (latest with artifact fixes)
- **Backup save path**: `D:\Sentinel-1\model\acswin_unet_pp.pth` (legacy compatibility)
- **Training Parameters**:
  - Batch size: 32 (auto-adjustable with `--batch-size-auto`)
  - Learning rate: 1e-4 with cosine annealing
  - Early stopping: 10 epochs patience
  - AMP: bfloat16 mixed precision (`--amp bf16`)
- **Physical Loss Weights** (연구질문.md defaults):
  - Magnitude: `--w-mag 1.0` (log-amplitude L1)
  - Phase: `--w-phase 1.0` (circular phase L1)
  - Coherence: `--w-coh 0.0` (interferometric coherence, disabled by default)
  - Spectral: `--w-spec 0.0` (frequency support constraint, disabled by default)
  - Data Consistency: `--w-dc 0.0` (forward model consistency, disabled by default)

### Data Processing Configuration
**Primary Method** - SNAP + Notebook workflow:
1. `data/final.xml`: Complete SNAP processing graph (orbit, calibration, TOPSAR processing)
2. `workflows/patch_extract_v2.ipynb`: Interactive patch extraction with quality control
   - Output directory: `data/patches/zero_filtered/`
   - Patch dimensions: 256x512 (width x height)
   - Quality filtering: Zero-value removal and cross-pol coherence validation
   - Statistical analysis and visualization included

**Alternative Method** - Direct extraction (legacy):
- `workflows/patch_extractor_gpu_enhanced.py`: GPU-accelerated processing
- Stride: non-overlapping (256×512)
- Quality threshold: cross-pol coherence > 0.01

**Degradation Configuration** (`model/degradations.py`):
- **PSF Options**: `--lp-kind gaussian` (σ=1.2px) or `sinc` (Kaiser windowed)
- **Noise Simulation**: `--enl 10.0` (speckle) + `--noise-std 0.01` (thermal)
- **Cache System**: Metadata-validated LR patches stored in `lr_cache/` subdirectories

## Development Workflow

### Setting up Training
1. Ensure SAR data is processed and available in patch format
2. Adjust data paths in training configuration if needed  
3. Start training with TensorBoard monitoring
4. Monitor training progress through TensorBoard dashboard
5. Evaluate model performance using built-in metrics

### Model Development (연구질문.md Guidelines)
- New models should inherit from PyTorch nn.Module
- **Complex Operations**: Use complex-valued layers for phase-safe processing
- **Phase Equivariance**: Ensure all activations preserve phase relationships
- Implement `count_parameters()` method for logging
- Support 4-channel real input/output format (internal complex conversion)
- Include `create_model(**kwargs)` factory function with architecture parameters
- **Attention Mechanisms**: Use magnitude re-normalization for complex attention weights

### Data Pipeline (Physical Consistency)
- SAR patches should be complex64 format with shape (2, H, W)
- Quality control through cross-pol coherence calculation
- Maintain consistent file naming: `*_dual_pol_complex_{x}_{y}.npy`
- **Physical LR Degradation** (연구질문.md §4.1):
  - **Forward Operator H**: Low-pass PSF convolution (Gaussian/Sinc)
  - **Decimation D**: Integer-scale spatial subsampling
  - **Optional Noise**: Speckle (Gamma ENL) + thermal (complex Gaussian)
- **Metadata-Driven Caching**: LR patches cached with degradation parameter validation
- **Cache Invalidation**: Automatic cache rebuild when degradation parameters change

## Performance Targets

### Quality Metrics (SAR Super-Resolution, 연구질문.md Standards)
- **PSNR > 30 dB**: Suitable for disaster monitoring applications (amplitude-based)
- **Phase RMSE < 0.5 rad**: Excellent phase reconstruction quality (circular error)
- **Local Coherence |γ| > 0.8**: Good preservation of interferometric information
- **CPIF > 25 dB**: Complex Peak Intensity Factor for overall reconstruction quality
- **Log-Intensity PSNR**: Research-standard metric for SAR image quality assessment

### Training Efficiency (Optimized)
- **Memory Usage**: ~8-12GB GPU memory for batch size 32 (bfloat16 AMP)
- **Training Speed**: ~0.66 patches/second processing rate with complex operations
- **Convergence**: Typically converges within 50-100 epochs
- **Cache Performance**: 90%+ cache hit rate after first epoch (metadata validation)
- **Profiling Support**: Integrated PyTorch profiler for performance bottleneck analysis

## Critical Implementation Notes

### Data Handling & Physical Compliance
- **Complex Data**: All SAR data must be complex64 format - workflows validate and convert automatically
- **Physical Degradation**: GPU-accelerated complex convolution with PSF + decimation
- **Metadata Validation**: Cache system ensures degradation parameter consistency across training runs
- **Shape Validation**: Training pipeline validates HR patches are (512,256) and LR patches are (128,64)
- **Noise Simulation**: Optional speckle (Gamma ENL) and thermal noise (complex Gaussian) for realistic conditions

### Workflow Compatibility & API Changes
- **SR_apply.py**: Updated to handle complex64 input and use `create_model()` function
- **npy2png.py**: Enhanced to correctly identify and process complex SAR data vs real data
- **Model Loading**: Always use model factory functions, never instantiate model classes directly
- **CLI Integration**: Comprehensive command-line interface with physical parameter controls
- **Loss Function API**: New dictionary-based loss components for detailed monitoring

### Artifact Mitigation & Research Progress (연구질문.md §3)
- **Research Question 3**: "Complex network 위상 정보의 학습 불안정성을 어떻게 처리할 것인지?"
- **Current Issue**: Models exhibit wavy/irregular pattern artifacts addressing phase processing instability
- **Root Causes Identified**:
  - ComplexGELU activation causing phase distortion through magnitude-based gating
  - SwinBlock overlap normalization creating phase discontinuities
  - Residual scaling balance affecting convergence stability
- **Implemented Solutions (Version 6)**:
  - **Complex SE Fix**: Prevents zero-channel division in squeeze ratio calculation
  - **Residual Connection**: Balanced `out * 0.3 + residual * 0.7` for training stability
  - **Resize Upsampling**: Bilinear interpolation + convolution eliminates grid artifacts
  - **Physical Loss Integration**: Magnitude, phase, coherence, and spectral losses
  - **Enhanced Caching**: Metadata-driven LR cache with parameter validation
- **Active Research Directions**:
  - Complex activation function alternatives (LeakyReLU, CReLU)
  - Phase continuity constraints in loss formulation
  - Alternative attention mechanisms for complex-valued features
- **Model Evolution**: 
  - Version 6: Current architecture with comprehensive artifact mitigation
  - Version 7+: Research-driven improvements based on 연구질문.md findings

### Recent Updates and Research Integration
- **Major Refactoring**: Complete 연구질문.md-based restructuring of codebase
- **Physical Degradation**: New `degradations.py` with realistic HR→LR synthesis
- **Loss Function Overhaul**: Physical loss components (magnitude, phase, coherence, spectral, data consistency)
- **Architecture Improvements**: Complex SE fix, resize upsampling, residual scaling optimization
- **Performance Optimization**: AMP with bfloat16, metadata-driven caching, profiling integration
- **Research Alignment**: Direct mapping from research questions to implementation features

### Common Issues and Solutions
- **Shape Mismatch**: Clear LR cache if changing HR/LR dimensions: `rm -rf data/patches/*/lr_cache/`
- **Model Import**: Use `from ac_swin_unet_pp import create_model` not direct class import
- **Complex Data**: Ensure input data is complex64 format - workflows auto-detect and convert
- **Degradation Cache**: Use `--rebuild-lr-cache` to force cache rebuild when changing degradation parameters
- **Phase Artifacts**: Monitor TensorBoard phase visualizations, adjust loss weights (`--w-phase`, `--w-coh`)
- **Memory Issues**: Use `--batch-size-auto` for automatic GPU memory optimization
- **Performance**: Enable `--auto-workers` and `--profile N` for bottleneck analysis