#!/usr/bin/env python3
"""
SAR LR Degradation Script
=========================
Applies the same LR degradation used in training to generate LR patches from HR patches.

Usage:
  # Apply to first 100 patches found recursively
  python degrade_patches.py --hr-root D:\Sentinel-1\data\patches\zero_filtered --lr-output D:\Sentinel-1\data\patches\LR --num-patches 100

  # Apply to all patches found recursively, preserving folder structure
  python degrade_patches.py --hr-root D:\Sentinel-1\data\patches\zero_filtered --lr-output D:\Sentinel-1\data\patches\LR --all

  # Apply to specific files (paths relative to --hr-root or absolute paths)
  python degrade_patches.py --hr-root D:\Sentinel-1\data\patches\zero_filtered --lr-output D:\Sentinel-1\data\patches\LR --files patch1.npy patch2.npy

Designed based on the training pipeline code provided.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- LR Degradation Function (Copied/Adapted from training script) ---

def simulate_lr_from_hr(hr_data: np.ndarray, lr_size: tuple = (128, 64), hr_size: tuple = (512, 256), device_str: str = 'cpu') -> np.ndarray:
    """
    Resolution-scaled SAR 열화 시뮬레이션(RSS) - Identical logic to training script.
    Args:
        hr_data: High resolution complex patch (2, H, W) np.ndarray complex64.
        lr_size: Target low resolution size (height, width).
        hr_size: Expected high resolution size (height, width).
        device_str: Device to run degradation on ('cpu' or 'cuda').
    Returns:
        lr_data: Low resolution complex patch (2, h, w) np.ndarray complex64.
    """
    device = torch.device(device_str)
    
    # --- 크기 검증 및 전처리 (train.py SARSuperResDataset.__getitem__ 로직 반영) ---
    # 1. 채널 수 확인 (첫 번째 차원이 2여야 함)
    if hr_data.shape[0] != 2:
        raise ValueError(f"HR data must have 2 channels in the first dimension. Got shape: {hr_data.shape}")

    # 2. 공간 차원 확인 및 필요시 전치
    if hr_data.shape[1:] == hr_size:
        # 이미 올바른 형태 (2, 512, 256)
        pass
    elif hr_data.shape[1:] == (hr_size[1], hr_size[0]):
        # 차원이 바뀐 경우 (2, 256, 512) -> (2, 512, 256)
        print(f"Info: Transposing HR data from {hr_data.shape} to (2, {hr_size[0]}, {hr_size[1]})")
        hr_data = np.transpose(hr_data, (0, 2, 1))
    else:
        # 예상 크기와 맞지 않음 -> 오류
        raise ValueError(f"HR spatial dimensions {hr_data.shape[1:]} do not match expected {hr_size} or {hr_size[1], hr_size[0]}.")

    # --- 열화 로직 (train.py SARSuperResDataset._simulate_lr_from_hr 와 동일) ---
    
    # HR 복소 텐서 로드 → (2, H, W)
    hr = torch.from_numpy(hr_data).to(device, non_blocking=False)
    
    # 스케일 = HR/LR 비
    scale_h = hr_size[0] // lr_size[0] # e.g., 512 // 128 = 4
    scale_w = hr_size[1] // lr_size[1] # e.g., 256 // 64 = 4

    # Assert integer scale factors (train.py와 동일)
    assert hr_size[0] == lr_size[0] * scale_h and hr_size[1] == lr_size[1] * scale_w, \
        f"HR size {hr_size} must be an integer multiple of LR size {lr_size}"

    # 1) 전력합 → √: divisor_override=1 로 ‘합’을 바로 계산
    power = (hr.real**2 + hr.imag**2).unsqueeze(0)       # (1,2,H,W)
    power_sum = F.avg_pool2d(power, (scale_h, scale_w), stride=(scale_h, scale_w),
                             divisor_override=1)         # 합 = ∑|z|²
    amp_lr = power_sum.sqrt().squeeze(0)                 # (2,H/4,W/4)

    # 2) 위상 평균
    phase = torch.angle(hr).unsqueeze(0)                    # (1,2,H,W)
    # depth-wise 평균을 위해 cos/sin → 같은 커널 사용
    cos_p = F.avg_pool2d(torch.cos(phase), (scale_h, scale_w), stride=(scale_h, scale_w))
    sin_p = F.avg_pool2d(torch.sin(phase), (scale_h, scale_w), stride=(scale_h, scale_w))
    phase_lr = torch.atan2(sin_p, cos_p).squeeze(0)

    # 3) 재조합 (complex)
    lr_complex = amp_lr * torch.exp(1j * phase_lr)

    # 4) CPU 반환
    return lr_complex.to("cpu").numpy().astype(np.complex64)


# --- Main Script Logic ---

def find_hr_files(hr_root: Path, pattern: str = "*_dual_pol_complex_*.npy") -> list[Path]:
    """Find all HR patch files recursively."""
    return list(hr_root.rglob(pattern))

def find_file_by_name(hr_root: Path, target_filename: str) -> Path:
    """
    Find a file by name recursively in the hr_root directory.
    Returns the first match found.
    """
    # Remove .npy extension if present for comparison
    base_name = target_filename.replace('.npy', '')
    
    for file_path in hr_root.rglob("*.npy"):
        file_base_name = file_path.stem  # filename without extension
        if file_base_name == base_name:
            return file_path
    
    raise FileNotFoundError(f"File '{target_filename}' not found in {hr_root} or its subdirectories")

def process_patches(hr_files: list[Path], hr_root: Path, lr_output_root: Path, lr_size: tuple, hr_size: tuple, device_str: str):
    """
    Process a list of HR files, applying degradation and saving LR versions.
    Preserves relative folder structure.
    """
    lr_output_root.mkdir(parents=True, exist_ok=True)

    for hr_file in tqdm(hr_files, desc="Degradation"):
        try:
            # 1. Load HR data
            hr_data = np.load(hr_file)
            
            # 2. Apply degradation
            lr_data = simulate_lr_from_hr(hr_data, lr_size, hr_size, device_str)
            
            # 3. Determine LR output path, preserving relative structure
            # e.g., D:\Sentinel-1\data\patches\zero_filtered\subfolder\patch.npy
            #   -> D:\Sentinel-1\data\patches\LR\subfolder\patch.npy
            try:
                rel_path = hr_file.relative_to(hr_root)
                lr_file_path = lr_output_root / rel_path
            except ValueError:
                # If hr_file is an absolute path not under hr_root, save directly in lr_output_root
                print(f"Warning: {hr_file} is not under {hr_root}. Saving to root of {lr_output_root}.")
                lr_file_path = lr_output_root / hr_file.name
            
            # 4. Create output directory if needed
            lr_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 5. Save LR data
            np.save(lr_file_path, lr_data)
        except Exception as e:
            print(f"\nError processing {hr_file}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Apply SAR LR degradation to HR patches.")
    parser.add_argument('--hr-root', type=Path, required=True, help='Root directory containing HR patches.')
    parser.add_argument('--lr-output', type=Path, required=True, help='Root directory to save generated LR patches.')
    parser.add_argument('--num-patches', type=int, default=None, help='Number of patches to process (from the start of the list).')
    parser.add_argument('--all', action='store_true', help='Process all patches found recursively.')
    parser.add_argument('--files', nargs='+', type=str, default=None, help='List of specific HR file names (relative to --hr-root) or absolute paths.')
    parser.add_argument('--use-gpu', action='store_true', help='Run degradation on GPU if available.')
    parser.add_argument('--hr-size', nargs=2, type=int, default=[512, 256], help='HR patch size (H, W). Default: 512 256')
    parser.add_argument('--lr-size', nargs=2, type=int, default=[128, 64], help='Target LR patch size (H, W). Default: 128 64')

    args = parser.parse_args()

    # --- Configuration ---
    HR_ROOT = args.hr_root.resolve()
    LR_OUTPUT_ROOT = args.lr_output.resolve()
    LR_SIZE = tuple(args.lr_size)
    HR_SIZE = tuple(args.hr_size)
    DEVICE_STR = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    print(f"HR Root: {HR_ROOT}")
    print(f"LR Output Root: {LR_OUTPUT_ROOT}")
    print(f"Target LR Size: {LR_SIZE}")
    print(f"Expected HR Size: {HR_SIZE}")
    print(f"Using device for degradation: {DEVICE_STR}")
    print("-" * 30)

    # --- Determine files to process ---
    hr_files_to_process = []
    if args.files:
        print("Processing specified files...")
        for f_str in args.files:
            try:
                f_path = Path(f_str)
                
                # Case 1: Absolute path provided
                if f_path.is_absolute():
                    if f_path.exists():
                        if f_path.suffix == '.npy':
                            hr_files_to_process.append(f_path.resolve())
                        else:
                            # If no .npy extension, try adding it
                            f_with_ext = f_path.with_suffix('.npy')
                            if f_with_ext.exists():
                                hr_files_to_process.append(f_with_ext.resolve())
                            else:
                                print(f"Warning: File {f_path} or {f_with_ext} does not exist. Skipping.", file=sys.stderr)
                    else:
                         # If no .npy extension, try adding it
                        f_with_ext = f_path.with_suffix('.npy')
                        if f_with_ext.exists():
                            hr_files_to_process.append(f_with_ext.resolve())
                        else:
                            print(f"Warning: File {f_path} or {f_with_ext} does not exist. Skipping.", file=sys.stderr)
                else:
                    # Case 2: Relative path or filename provided, search recursively
                    try:
                        found_file = find_file_by_name(HR_ROOT, f_str)
                        hr_files_to_process.append(found_file)
                        print(f"Found: {found_file}")
                    except FileNotFoundError:
                        print(f"Warning: File '{f_str}' not found in {HR_ROOT} or its subdirectories. Skipping.", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error processing file '{f_str}': {e}. Skipping.", file=sys.stderr)
        
        if not hr_files_to_process:
            print("No valid files provided via --files.", file=sys.stderr)
            sys.exit(1)
        else:
             print(f"Found {len(hr_files_to_process)} files to process via --files.")
    elif args.all:
        print("Finding all HR patches...")
        hr_files_to_process = find_hr_files(HR_ROOT)
        if not hr_files_to_process:
            print(f"No HR files found in {HR_ROOT} matching the pattern.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(hr_files_to_process)} HR patches.")
    elif args.num_patches is not None and args.num_patches > 0:
        print(f"Finding HR patches and selecting first {args.num_patches}...")
        all_hr_files = find_hr_files(HR_ROOT)
        if not all_hr_files:
             print(f"No HR files found in {HR_ROOT} matching the pattern.", file=sys.stderr)
             sys.exit(1)
        hr_files_to_process = all_hr_files[:args.num_patches]
        print(f"Selected {len(hr_files_to_process)} HR patches.")
    else:
        parser.error("One of --num-patches, --all, or --files must be specified.")

    # --- Confirm before processing large number of files ---
    if len(hr_files_to_process) > 1000:
        confirm = input(f"You are about to process {len(hr_files_to_process)} files. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    # --- Process files ---
    print("Starting degradation process...")
    try:
        process_patches(hr_files_to_process, HR_ROOT, LR_OUTPUT_ROOT, LR_SIZE, HR_SIZE, DEVICE_STR)
        print("\nDegradation process completed successfully!")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nDegradation process failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()