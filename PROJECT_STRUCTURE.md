# Project Structure

```
Sentinel-1/                           # Root project directory
├── CLAUDE.md                         # Project instructions for Claude Code
├── README.md                         # Project documentation and overview
│
├── data/                             # Dataset and preprocessing
│   ├── S1_raw/                       # Raw Sentinel-1 SAFE files
│   ├── buffer/                       # Intermediate processing outputs
│   ├── patches/                      # Extracted SAR patches
│   │   ├── S1B_*_11B5*/             # Scene-specific patch directories
│   │   ├── LR/                       # Low-resolution patches
│   │   └── zero_filtered/            # Quality-filtered patch dataset
│   ├── processed_1/                  # SNAP-processed individual swaths
│   ├── processed_2/                  # Training/validation splits
│   ├── processed_2_production_final/ # Production dataset
│   ├── graph1.xml                    # SNAP preprocessing graph (orbit correction)
│   ├── graph2.xml                    # SNAP preprocessing graph (calibration)
│   ├── final.xml                     # Complete SNAP processing workflow
│   └── validate.ipynb               # Data validation notebook
│
├── model/                            # Core deep learning implementation
│   ├── train.py                      # Main training script with TensorBoard integration
│   ├── ac_swin_unet_pp.py           # AC-Swin-UNet++ model (primary architecture)
│   ├── cv_unet.py                   # Complex U-Net baseline model
│   ├── data_cache.py                # Data loading and caching utilities
│   ├── utils.py                     # Training utilities and metrics
│   ├── speed_utils.py               # Performance optimization utilities
│   └── visualize_tensorboard.py     # TensorBoard analysis and export tools
│
├── model_weights/                    # Saved model checkpoints
│   ├── version1/                    # Initial Complex U-Net models
│   ├── version2/                    # AC-Swin-UNet++ baseline
│   ├── version3_5000samples/        # Limited dataset experiments
│   ├── version4_(checkboard_artifacts)/ # Models with checkerboard issues
│   └── version5(여전한 아티팩트)/    # Latest attempts at artifact mitigation
│
├── workflows/                        # Data processing and application scripts
│   ├── patch_extractor_gpu_enhanced.py  # GPU-accelerated patch extraction
│   ├── SR_apply.py                  # Super-resolution inference pipeline
│   ├── degrade_patches.py           # LR patch generation from HR patches
│   ├── npy2png.py                   # SAR data visualization utility
│   ├── patch_extract_v2.ipynb       # Interactive patch extraction notebook
│   ├── sar_visual_patch.ipynb       # Patch-level visualization tools
│   └── sar_visual_scene.ipynb       # Scene-level analysis tools
│
├── runs/                            # TensorBoard experiment logs
│   ├── sar_sr_20250805_*/           # Training session logs (timestamped)
│   ├── sar_sr_20250806_*/           # Checkerboard mitigation attempts
│   ├── sar_sr_20250808_*/           # Recent training experiments
│   └── sar_sr_20250809_*/           # Latest experimental runs
│
├── results/                         # Model outputs and analysis
│   ├── single_patches/              # Individual patch super-resolution results
│   ├── version1/                    # Results from baseline models
│   ├── *_comparison.png             # Visual comparisons (GT vs SR)
│   ├── gt_mosaic.mmap              # Ground truth mosaic visualization
│   └── sr_mosaic.mmap              # Super-resolution mosaic visualization
│
├── analysis_results/                # Performance analysis and metrics
│   ├── sar_analysis_report.md       # Detailed performance analysis
│   ├── sar_performance_metrics.csv  # Quantitative evaluation results
│   ├── sar_performance_summary.png  # Performance summary visualization
│   ├── statistical_analysis.png     # Statistical analysis plots
│   └── visual_comparison_*.png      # Best/median/worst case comparisons
│
├── plots/                           # Training progress visualizations
│   ├── training_loss.png            # Loss curve analysis
│   └── training_psnr.png            # PSNR progression tracking
│
├── amp_db_histogram/                # Amplitude analysis by subswath
│   ├── 148B_IW2.png                # IW2 subswath amplitude distribution
│   ├── 84AF_IW1.png                # IW1 subswath amplitude distribution
│   ├── 84AF_IW2.png                # IW2 subswath amplitude distribution
│   └── 84AF_IW3.png                # IW3 subswath amplitude distribution
│
└── auxiliary_scripts/               # Utility and analysis scripts
    ├── sar_comparative_analysis.py  # Model performance comparison
    ├── visualize_subswath.py       # Subswath-specific visualization
    └── 데이터자동다운로드스크립트.py    # Automated data download utility
```

## Key Architecture Components

### Model Architecture
- **Primary Model**: AC-Swin-UNet++ (`ac_swin_unet_pp.py`)
  - Complex-valued convolutions for SAR phase preservation
  - Swin Transformer blocks with shifted windows
  - Dense skip connections (U-Net++ architecture)
  - Complex SE attention + spatial attention mechanisms
  - Complex PixelShuffle for artifact-reduced upsampling

- **Baseline Model**: Complex U-Net (`cv_unet.py`)
  - Traditional encoder-decoder with complex operations
  - VH attention mechanism for cross-polarization guidance

### Data Pipeline
- **Input Format**: Complex64 dual-pol SAR patches (2, H, W) → VV+VH polarizations
- **Model I/O**: 4-channel real tensors [VV-Re, VV-Im, VH-Re, VH-Im]
- **Resolution**: HR patches (512×256) → LR patches (128×64) for 4× super-resolution
- **Quality Control**: Cross-polarization coherence filtering and zero-value removal

### Training Infrastructure
- **Loss Function**: Hybrid amplitude MSE + phase L1 + optional perceptual loss
- **Optimization**: Mixed precision training with early stopping
- **Monitoring**: Comprehensive TensorBoard logging with SAR-specific metrics
- **Performance**: Dual-pol PSNR/SSIM evaluation for VV and VH channels

### Current Status
- **Active Issue**: Checkerboard artifacts in super-resolution outputs
- **Mitigation Efforts**: Multiple training iterations with modified loss functions and architectures
- **Data Scale**: Working with filtered high-quality patch dataset
- **Model Versions**: Progressive improvements through 5 major model iterations

## File Naming Conventions

### SAR Patches
- Format: `{scene_id}_dual_pol_complex_{x}_{y}.npy`
- Content: Complex64 array with shape (2, 256, 512) for [VV, VH] polarizations
- Location: Organized by scene in `data/patches/{scene_id}/`

### Model Weights
- Organized by version in `model_weights/version{N}/`
- Naming: `{model_type}.pth` (e.g., `acswin_unet_pp.pth`)
- Include metadata about training configuration and performance

### Experimental Logs
- TensorBoard logs: `runs/sar_sr_{YYYYMMDD}_{HHMMSS}/`
- Training artifacts: Automatic cleanup and archival system
- Performance tracking: Integrated metrics logging and visualization