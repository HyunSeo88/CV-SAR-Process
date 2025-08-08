#!/usr/bin/env python3
"""
python visualize_subswath.py --patch-dir "D:\Sentinel-1\data\processed_2\S1A_IW_SLC__1SDV_20200714T093144_20200714T093212_033449_03E046_B97F_Orb_Cal_IW3\IW3"  
    --out-dir "results/single_patches" --patches "S1A_IW_SLC__1SDV_20200714T093144_20200714T093212_033449_03E046_B97F_Orb_Cal_IW3_dual_pol_complex_1024_8192.npy"
    
visualize_subswath.py
======================

A dedicated script to mosaic an entire SAR subswath from patches,
apply a super-resolution model, perform standard visual corrections,
and save the final ground truth vs. super-resolved images for comparison.
"""

import numpy as np
import torch
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import re
import cv2
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# Add model directory to path to import model components
sys.path.append('./model')
try:
    from ac_swin_unet_pp import create_model as create_swin_model
    # from train import SARSuperResDataset  # Re-used for its LR degradation logic - not needed for visualization
except ImportError as e:
    print(f"Error: Could not import model components. Make sure 'ac_swin_unet_pp.py' is in the './model' directory.")
    print(f"Details: {e}")
    sys.exit(1)

class SARImageAnalyzer:
    """
    A tool to visually compare super-resolved vs. ground truth SAR images.
    This class is self-contained and handles model loading, inference, and image generation.
    """
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self):
        """Load a trained Swin U-Net model."""
        model = create_swin_model()
        print(f"Loading trained model...")
        print(f"Swin U-Net initialized with {sum(p.numel() for p in model.parameters())} parameters")

        if not Path(self.model_path).exists():
            print(f"ERROR: Model checkpoint not found at {self.model_path}")
            sys.exit(1)
        
        try:
            # FIX: Set weights_only=False to load older checkpoints
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model state_dict directly.")
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            sys.exit(1)
        
        model.to(self.device)
        model.eval()
        return model
    
    def generate_super_resolved(self, lr_patch_np: np.ndarray) -> np.ndarray:
        """Generate a single super-resolved image from a low-resolution patch."""
        with torch.no_grad():
            # The Swin U-Net model expects a 4-channel input: VV-Re, VV-Im, VH-Re, VH-Im
            # The LR patch is complex (2, H, W). We need to convert it to 4-channel real.
            vv_real = torch.from_numpy(lr_patch_np[0].real).float().unsqueeze(0)
            vv_imag = torch.from_numpy(lr_patch_np[0].imag).float().unsqueeze(0)
            vh_real = torch.from_numpy(lr_patch_np[1].real).float().unsqueeze(0)
            vh_imag = torch.from_numpy(lr_patch_np[1].imag).float().unsqueeze(0)
            
            model_input = torch.cat([vv_real, vv_imag, vh_real, vh_imag], dim=0).unsqueeze(0).to(self.device)
            
            sr_patch = self.model(model_input).squeeze(0).cpu()
            
            # Convert model output (2-channel real) back to complex
            sr_complex = torch.complex(sr_patch[0], sr_patch[1]).numpy()
            return sr_complex

    def create_comparison_plot(self, sr_complex: np.ndarray, gt_complex: np.ndarray, save_path: Path):
        """Creates and saves a side-by-side plot of GT and SR VV-polarized amplitude."""
        sr_amp_log = 10 * np.log10(np.abs(sr_complex)**2 + 1e-8)
        gt_amp_log = 10 * np.log10(np.abs(gt_complex)**2 + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f'Visual Comparison: {save_path.stem}', fontsize=16)

        im1 = axes[0].imshow(gt_amp_log, cmap='gray')
        axes[0].set_title('Ground Truth (VV Amplitude - dB Scale)')
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], shrink=0.8)

        im2 = axes[1].imshow(sr_amp_log, cmap='gray')
        axes[1].set_title('Super-Resolved (VV Amplitude - dB Scale)')
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], shrink=0.8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comparison plot to: {save_path}")

    def simulate_lr_from_hr(self, hr_data: np.ndarray) -> np.ndarray:
        """Simulate LR from HR using the same logic as SARSuperResDataset."""
        hr = torch.from_numpy(hr_data).to(self.device, non_blocking=False)
        
        # Scale = HR/LR ratio
        scale = 4  # Assuming 4x downscaling
        lr_size = (hr.shape[-2] // scale, hr.shape[-1] // scale)
        
        # 1) Power sum → √: divisor_override=1 for 'sum' calculation
        power = (hr.real**2 + hr.imag**2).unsqueeze(0)       # (1,2,H,W)
        power_sum = torch.nn.functional.avg_pool2d(power, scale, stride=scale,
                             divisor_override=1)         # sum = ∑|z|²
        amp_lr = power_sum.sqrt().squeeze(0)                 # (2,H/4,W/4)
        
        # 2) Phase average
        phase = torch.angle(hr).unsqueeze(0)                    # (1,2,H,W)
        # depth-wise average using cos/sin
        cos_p = torch.nn.functional.avg_pool2d(torch.cos(phase), scale, stride=scale)
        sin_p = torch.nn.functional.avg_pool2d(torch.sin(phase), scale, stride=scale)
        phase_lr = torch.atan2(sin_p, cos_p).squeeze(0)

        # 3) Recombine (complex)
        lr_complex = amp_lr * torch.exp(1j * phase_lr)

        # 4) Return to CPU
        return lr_complex.to("cpu").numpy().astype(np.complex64)

    def process_and_save_patch(self, patch_path: Path):
        """Process a single patch: load GT, generate LR, create SR, and save comparison."""
        # 1. Load Ground Truth (HR) patch directly
        gt_patch = np.load(patch_path)  # (2, H, W) complex

        # Ensure correct shape and data type
        if gt_patch.shape != (2, 512, 256):
            # Transpose if dimensions are swapped
            if gt_patch.shape == (2, 256, 512):
                gt_patch = np.transpose(gt_patch, (0, 2, 1))  # Swap height/width
        
        if gt_patch.dtype != np.complex64:
            gt_patch = gt_patch.astype(np.complex64)

        # 2. Generate Low-Resolution version
        lr_patch = self.simulate_lr_from_hr(gt_patch)  # (2, H/4, W/4) complex

        # 3. Generate Super-Resolved version
        sr_patch = self.generate_super_resolved(lr_patch)  # (H, W) complex from VV channel

        # 4. Create and save the comparison plot
        save_path = self.output_dir / f"{patch_path.stem}_comparison.png"
        # We compare the VV channel (index 0)
        self.create_comparison_plot(sr_patch, gt_patch[0], save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and Compare Full SAR Subswaths.")
    parser.add_argument("--ckpt", default="model/acswin_unet_pp.pth", help="Path to the trained model checkpoint.")
    parser.add_argument("--patch-dir", required=True, help="Directory containing the HR .npy patches for a single subswath.")
    parser.add_argument("--out-dir", default="results/subswath_comparison", help="Directory to save the final mosaic images.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for model inference.")
    parser.add_argument("--patches", type=str, default=None, help="Comma-separated list of specific patch filenames to process (e.g., 'patch1.npy,patch2.npy'). Overrides patch-dir scan.")
    parser.add_argument("--use-cache", action='store_true', default=False,
                        help="Reuse cached HR patch file list (valid_hr_files_*.pkl) if present. Default is off to force fresh directory scan.")
    return parser.parse_args()

def parse_xy_from_filename(filename: str) -> (int, int):
    """Parses x and y pixel coordinates from the patch filename."""
    match = re.search(r'_complex_(\d+)_(\d+)', filename)
    if not match:
        raise ValueError(f"Could not parse coordinates from filename: {filename}")
    return int(match.group(1)), int(match.group(2))

def apply_sar_visual_correction(complex_data: np.ndarray) -> np.ndarray:
    """Applies standard SAR corrections for visualization."""
    # 1. Amplitude Calculation
    amplitude = np.abs(complex_data)
    
    # 2. Speckle Filtering (using a simple median filter)
    filtered_amplitude = median_filter(amplitude, size=3)
    
    # 3. dB Scaling
    # Add a small epsilon to avoid log(0)
    power = filtered_amplitude**2
    db_scaled = 10 * np.log10(power + 1e-9)
    
    # 4. 8-bit Normalization
    min_val, max_val = np.percentile(db_scaled, (5, 95)) # Clip outliers for better contrast
    db_scaled = np.clip(db_scaled, min_val, max_val)
    normalized = (db_scaled - min_val) / (max_val - min_val) * 255
    
    return normalized.astype(np.uint8)

def run_single_patch_analysis(args):
    """Function to handle single patch analysis."""
    print(f"DEBUG: args.patches = '{args.patches}'")
    print(f"DEBUG: type(args.patches) = {type(args.patches)}")
    
    patch_path = Path(args.patches)
    print(f"DEBUG: patch_path = '{patch_path}'")
    print(f"DEBUG: patch_path.exists() = {patch_path.exists()}")
    
    if not patch_path.exists():
        print(f"ERROR: Patch file does not exist: {patch_path}")
        return
    
    print("Running single patch analysis. Using device: cuda")

    # Load model
    analyzer = SARImageAnalyzer(model_path=args.ckpt, output_dir=args.out_dir)
    
    # Process and save the single patch (no dataset needed)
    analyzer.process_and_save_patch(patch_path)

def main():
    args = parse_args()
    
    if args.patches:
        run_single_patch_analysis(args)
    else:
        run_mosaic_analysis(args)

def run_mosaic_analysis(args):
    """Runs the full subswath mosaic analysis."""
    patch_dir = Path(args.patch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running full mosaic analysis. Using device: {device}")
    
    # Create an analyzer instance which will load the model
    analyzer = SARImageAnalyzer(model_path=args.ckpt, output_dir=out_dir)

    # --- 1. Scan patches and determine mosaic size ---
    print("Scanning patches to determine subswath dimensions...")
    patch_files = list(patch_dir.glob("*_complex_*.npy"))
    if not patch_files:
        print(f"Error: No complex patch files found in {patch_dir}")
        sys.exit(1)

    max_x, max_y = 0, 0
    patch_shape = np.load(patch_files[0]).shape
    patch_h, patch_w = patch_shape[1], patch_shape[2]

    for f in patch_files:
        x, y = parse_xy_from_filename(f.name)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        
    total_width = max_x + patch_w
    total_height = max_y + patch_h
    print(f"Subswath dimensions calculated: {total_width}x{total_height} pixels")
    
    # --- 2. Create empty mosaics (using memmap for large images) ---
    gt_mosaic_path = out_dir / 'gt_mosaic.mmap'
    sr_mosaic_path = out_dir / 'sr_mosaic.mmap'
    
    gt_mosaic = np.memmap(gt_mosaic_path, dtype=np.complex64, mode='w+', shape=(total_height, total_width))
    sr_mosaic = np.memmap(sr_mosaic_path, dtype=np.complex64, mode='w+', shape=(total_height, total_width))
    
    # --- 3. Process patches in batches and fill mosaics ---
    print("Processing patches, running inference, and creating mosaics...")

    for i in tqdm(range(0, len(patch_files), args.batch_size), desc="Inferring"):
        batch_files = patch_files[i:i+args.batch_size]
        
        hr_patches_np = [np.load(f) for f in batch_files]
        
        # Create LR batch using the analyzer's LR simulation logic
        lr_tensors = [analyzer.simulate_lr_from_hr(hr) for hr in hr_patches_np]
        lr_batch = torch.stack([torch.from_numpy(np.stack([lr[0].real, lr[0].imag, lr[1].real, lr[1].imag])).float() for lr in lr_tensors]).to(device)

        # Run SR inference
        with torch.no_grad():
            sr_batch_torch = analyzer.model(lr_batch).cpu()

        # Place patches into mosaics
        for j, file_path in enumerate(batch_files):
            x, y = parse_xy_from_filename(file_path.name)
            
            # GT Mosaic (VV polarization, channel 0)
            gt_mosaic[y:y+patch_h, x:x+patch_w] = hr_patches_np[j][0, :, :]
            
            # SR Mosaic (VV polarization, channel 0)
            sr_complex = torch.complex(sr_batch_torch[j, 0], sr_batch_torch[j, 1]).numpy()
            sr_mosaic[y:y+patch_h, x:x+patch_w] = sr_complex

    # --- 4. Apply visual correction and save final images ---
    print("Applying final visual corrections...")
    gt_corrected = apply_sar_visual_correction(gt_mosaic)
    sr_corrected = apply_sar_visual_correction(sr_mosaic)
    
    gt_save_path = out_dir / f"{patch_dir.parent.name}_GT_corrected.png"
    sr_save_path = out_dir / f"{patch_dir.parent.name}_SR_corrected.png"
    
    print(f"Saving Ground Truth image to {gt_save_path}")
    cv2.imwrite(str(gt_save_path), gt_corrected)
    
    print(f"Saving Super-Resolved image to {sr_save_path}")
    cv2.imwrite(str(sr_save_path), sr_corrected)
    
    print("\nAnalysis complete. You can find the final images in the output directory.")


if __name__ == "__main__":
    main()
