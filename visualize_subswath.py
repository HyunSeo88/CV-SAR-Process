#!/usr/bin/env python3
"""
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
    from cv_unet import ComplexUNet
    from train import SARSuperResDataset  # Re-used for its LR degradation logic
except ImportError as e:
    print(f"Error: Could not import model components. Make sure 'cv_unet.py' and 'train.py' are in the './model' directory.")
    print(f"Details: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and Compare Full SAR Subswaths.")
    parser.add_argument("--ckpt", default="model/cv_unet.pth", help="Path to the trained model checkpoint.")
    parser.add_argument("--patch-dir", required=True, help="Directory containing the HR .npy patches for a single subswath.")
    parser.add_argument("--out-dir", default="results/subswath_comparison", help="Directory to save the final mosaic images.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for model inference.")
    parser.add_argument("--patches", type=str, default=None, help="Comma-separated list of specific patch filenames to process (e.g., 'patch1.npy,patch2.npy'). Overrides patch-dir scan.")
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

def main():
    args = parse_args()
    
    if args.patches:
        run_single_patch_analysis(args)
    else:
        run_mosaic_analysis(args)

def run_single_patch_analysis(args):
    """Runs the analysis for a few specific patches."""
    patch_dir = Path(args.patch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running single patch analysis. Using device: {device}")

    # Load Model
    print("Loading trained model...")
    model = ComplexUNet()
    try:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        sys.exit(1)
    model.to(device).eval()
    
    patch_filenames = [p.strip() for p in args.patches.split(',')]
    dummy_dataset = SARSuperResDataset(data_dir=str(patch_dir), split='test', use_cache=False)

    for filename in tqdm(patch_filenames, desc="Processing individual patches"):
        patch_path = patch_dir / filename
        if not patch_path.exists():
            print(f"Warning: Patch file not found, skipping: {patch_path}")
            continue

        hr_patch_np = np.load(patch_path)
        
        lr_complex = dummy_dataset._simulate_lr_from_hr(hr_patch_np)
        lr_tensor = torch.from_numpy(np.stack([lr_complex[0].real, lr_complex[0].imag, np.abs(lr_complex[1])])).float().unsqueeze(0).to(device)

        with torch.no_grad():
            sr_torch = model(lr_tensor).cpu().squeeze(0)
        
        sr_complex = torch.complex(sr_torch[0], sr_torch[1]).numpy()

        # Apply corrections and save side-by-side comparison
        gt_corrected = apply_sar_visual_correction(hr_patch_np[0])
        sr_corrected = apply_sar_visual_correction(sr_complex)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Comparison for {filename}', fontsize=16)
        axes[0].imshow(gt_corrected, cmap='gray')
        axes[0].set_title('Ground Truth (Corrected)')
        axes[0].axis('off')
        axes[1].imshow(sr_corrected, cmap='gray')
        axes[1].set_title('Super-Resolved (Corrected)')
        axes[1].axis('off')
        
        save_path = out_dir / f"{Path(filename).stem}_comparison.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved comparison for {filename} to {save_path}")

def run_mosaic_analysis(args):
    """Runs the full subswath mosaic analysis."""
    patch_dir = Path(args.patch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running full mosaic analysis. Using device: {device}")
    
    # Load Model
    print("Loading trained model...")
    model = ComplexUNet()
    try:
        # FIX: Set weights_only=False to load older checkpoints
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state_dict directly.")

    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()

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
    # Use a dummy dataset instance to access the LR degradation logic
    dummy_dataset = SARSuperResDataset(data_dir=str(patch_dir), split='test', use_cache=False)

    for i in tqdm(range(0, len(patch_files), args.batch_size), desc="Inferring"):
        batch_files = patch_files[i:i+args.batch_size]
        
        hr_patches_np = [np.load(f) for f in batch_files]
        
        # Create LR batch using the logic from SARSuperResDataset
        lr_tensors = [dummy_dataset._simulate_lr_from_hr(hr) for hr in hr_patches_np]
        lr_batch = torch.stack([torch.from_numpy(np.stack([lr[0].real, lr[0].imag, np.abs(lr[1])])).float() for lr in lr_tensors]).to(device)

        # Run SR inference
        with torch.no_grad():
            sr_batch_torch = model(lr_batch).cpu()

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
