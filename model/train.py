#!/usr/bin/env python3
"""
Training Script for Complex U-Net SAR Super-Resolution
======================================================

Complete training pipeline for dual-pol SAR super-resolution:
- Custom SAR Dataset with LR/HR patch loading
- Training loop with early stopping
- Validation and metrics tracking
- Model checkpointing

Designed for Korean disaster monitoring applications.
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Tuple, Optional
import datetime
from contextlib import nullcontext  # PERF: for optional context
# PERF: Import argparse for CLI flags
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# PERF: Import AMP modules for mixed precision
# PERF: Use new AMP API to avoid deprecation warnings
from torch import amp
autocast = amp.autocast
GradScaler = amp.GradScaler

# Import utils always
from utils import sr_loss, MetricsCalculator, PerceptualLoss

# Provide a default model factory to satisfy static references; CLI may override at runtime
try:
    from ac_swin_unet_pp import create_model as _default_create_model
    create_model = _default_create_model  # will be overridden in __main__ if needed
except Exception:
    create_model = None  # Will be set in __main__
from data_cache import load_or_compute_lr
# PERF: Import speed utils
from speed_utils import get_optimal_workers, auto_adjust_batch_size, setup_profiler


class SARSuperResDataset(Dataset):
    """
    SAR Super-Resolution Dataset
    
    Loads dual-pol complex SAR patches for super-resolution training.
    """
    
    def __init__(self, file_list: list, data_dir: str, lr_size: Tuple[int, int] = (128, 64),
                 hr_size: Tuple[int, int] = (512, 256), *, use_cache: bool = True, gpu_degrade: bool = False):

        """
        Initialize SAR dataset
        
        Args:
            file_list: A list of file paths for this dataset split.
            data_dir: Base data directory (for synthetic data generation).
            lr_size: Low resolution patch size (height, width)
            hr_size: High resolution patch size (height, width)
        """
        self.hr_files = file_list
        self.data_dir = Path(data_dir)
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.use_cache = use_cache
        self.gpu_degrade = gpu_degrade
        
        print(f"Initialized dataset with {len(self.hr_files)} SAR patches.")
        
        if len(self.hr_files) == 0:
            print(f"Warning: No SAR patches provided for this split.")
            print("Creating synthetic dataset for testing...")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic data for testing when no real data is available"""
        n_samples = 1000 if self.split == 'train' else 100
        
        self.synthetic_data = []
        for i in range(n_samples):
            # Create synthetic HR patch
            hr_real = np.random.randn(2, *self.hr_size).astype(np.float32)
            hr_imag = np.random.randn(2, *self.hr_size).astype(np.float32)
            hr_data = hr_real + 1j * hr_imag
            
            self.synthetic_data.append(hr_data)
        
        self.hr_files = [f"synthetic_{i}.npy" for i in range(n_samples)]
    
    def _simulate_lr_from_hr(self, hr_data: np.ndarray) -> np.ndarray:
        """Resolution-scaled SAR 열화 시뮬레이션(RSS)"""
        # 0) 디바이스 선택
        use_gpu = self.gpu_degrade and torch.cuda.is_available()
        device  = torch.device("cuda") if use_gpu else torch.device("cpu")

        # HR 복소 텐서 로드 → (2, H, W)
        hr = torch.from_numpy(hr_data).to(device, non_blocking=False)

        # 스케일 = HR/LR 비
        scale = self.hr_size[0] // self.lr_size[0]      # ex. 4
        assert hr.shape[-2:] == self.hr_size, f"HR shape mismatch: got {hr.shape[-2:]}, expected {self.hr_size}"

        # 1) 평균 전력 → √: 물리적 스케일 정합을 위해 평균으로 다운샘플
        power = (hr.real**2 + hr.imag**2).unsqueeze(0)       # (1,2,H,W)
        power_avg = F.avg_pool2d(power, scale, stride=scale) # 평균 = (1/|K|)∑|z|²
        amp_lr = power_avg.sqrt().squeeze(0)                 # (2,H/4,W/4)
        
        # 2) 위상 평균
        phase = torch.angle(hr).unsqueeze(0)                    # (1,2,H,W)
        # depth-wise 평균을 위해 cos/sin → 같은 커널 사용
        cos_p = F.avg_pool2d(torch.cos(phase), scale, stride=scale)
        sin_p = F.avg_pool2d(torch.sin(phase), scale, stride=scale)
        phase_lr = torch.atan2(sin_p, cos_p).squeeze(0)

        # 3) 재조합 (complex)
        lr_complex = amp_lr * torch.exp(1j * phase_lr)

        # 4) CPU 반환 - DataLoader 워커 ≥1 이어도 CUDA 컨텍스트 해제
        return lr_complex.to("cpu").numpy().astype(np.complex64)
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        """
        Get LR/HR patch pair
        
        Returns:
            lr_patch: Low resolution complex patch (2, 64, 128)
            hr_patch: High resolution complex patch (2, 256, 512)
        """
        if hasattr(self, 'synthetic_data'):
            # Use synthetic data
            hr_data = self.synthetic_data[idx]
        else:
            # Load real data
            hr_file = self.hr_files[idx]
            hr_data = np.load(hr_file)
        
        # HR size is now consistent - no cropping/padding needed
        # Ensure correct shape and data type
        if hr_data.shape != (2, *self.hr_size):
            # Transpose if dimensions are swapped (512,256) vs (256,512)
            if hr_data.shape == (2, self.hr_size[1], self.hr_size[0]):
                hr_data = np.transpose(hr_data, (0, 2, 1))  # Swap height/width
        
        # Convert to complex64 if not already
        if hr_data.dtype != np.complex64:
            hr_data = hr_data.astype(np.complex64)
        
        # Generate or load corresponding LR patch
        if not hasattr(self, 'synthetic_data') and self.use_cache:
            lr_data = load_or_compute_lr(Path(hr_file), self._simulate_lr_from_hr, hr_data)
        else:
            lr_data = self._simulate_lr_from_hr(hr_data)
        
        # Convert complex to 4 real-channel tensor [VV-Re, VV-Im, VH-Re, VH-Im]
        lr_tensor = torch.from_numpy(np.stack([
            lr_data[0].real, lr_data[0].imag,
            lr_data[1].real, lr_data[1].imag
        ], axis=0)).float()
        hr_tensor = torch.from_numpy(np.stack([
            hr_data[0].real, hr_data[0].imag,
            hr_data[1].real, hr_data[1].imag
        ], axis=0)).float()
        return lr_tensor, hr_tensor



class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=7, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def create_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 0, *, use_cache: bool = True, gpu_degrade: bool = False, max_samples: int = None):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Directory containing SAR data patches
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        use_cache: Whether to use cached file splits
        gpu_degrade: Whether to run HR→LR degradation on GPU
        max_samples: Maximum number of samples to use (None for all data)
    """
    # --- Centralized File Scanning and Splitting ---
    pattern = "*_dual_pol_complex_*.npy"
    cache_path = Path(data_dir) / 'file_split_cache.pkl'
    
    # Generate cache key based on max_samples to avoid conflicts
    if max_samples is not None:
        cache_path = Path(data_dir) / f'file_split_cache_{max_samples}.pkl'
    
    if use_cache and cache_path.exists():
        with open(cache_path, 'rb') as f:
            file_splits = pickle.load(f)
        print(f"Loaded file splits from cache: {cache_path}")
        if max_samples is not None:
            print(f"Using subset of {max_samples} samples")
    else:
        print("Scanning for all .npy files and creating new train/val/test splits...")
        all_files = list(Path(data_dir).rglob(pattern))
        # 1) Filter out cached/low-res or generated directories that should not be treated as HR
        def is_valid_hr_path(p: Path) -> bool:
            blocklist = {"lr_cache", "LR", "SR"}
            parts = {str(x) for x in p.parts}
            return not bool(parts & blocklist)
        before = len(all_files)
        all_files = [p for p in all_files if is_valid_hr_path(p)]
        after_dir = len(all_files)
        if after_dir < before:
            print(f"Filtered {before - after_dir} by directory name (excluded lr_cache/LR/SR)")

        # 2) Keep only files whose stored array shape matches HR size (or transposed)
        hr_h, hr_w = 512, 256
        valid_files = []
        bad_shape = 0
        for p in all_files:
            try:
                arr = np.load(p, mmap_mode='r')
                if arr.shape == (2, hr_h, hr_w) or arr.shape == (2, hr_w, hr_h):
                    valid_files.append(p)
                else:
                    bad_shape += 1
            except Exception:
                bad_shape += 1
                continue
        if bad_shape:
            print(f"Filtered {bad_shape} by shape (kept HR-sized only)")
        all_files = valid_files
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(all_files):
            print(f"Limiting dataset to {max_samples} samples out of {len(all_files)} available")
            np.random.seed(42)  # For reproducible subset selection
            np.random.shuffle(all_files)
            all_files = all_files[:max_samples]
        else:
            np.random.shuffle(all_files)
        
        n_total = len(all_files)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        file_splits = {
            'train': all_files[:n_train],
            'val': all_files[n_train : n_train + n_val],
            'test': all_files[n_train + n_val:]
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(file_splits, f)
        print(f"Saved new file splits to cache: {cache_path}")

    # Create datasets with pre-split file lists
    train_dataset = SARSuperResDataset(file_splits['train'], data_dir, use_cache=use_cache, gpu_degrade=gpu_degrade)
    val_dataset = SARSuperResDataset(file_splits['val'], data_dir, use_cache=use_cache, gpu_degrade=gpu_degrade)
    
    # PERF: Optimized DataLoader settings - pin_memory=False for better stability
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # Changed to False for better stability
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,  # Changed to False for better stability
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, metrics_calc, perceptual, scaler, profiler=None, *, perceptual_weight: float = 0.0, fft_weight: float = 0.0):
    """
    Train for one epoch
    
    Args:
        model: ComplexUNet model
        train_loader: Training data loader
        optimizer: Optimizer
        device: torch device
        metrics_calc: MetricsCalculator instance
        perceptual: PerceptualLoss instance
        
    Returns:
        Average training loss
    """
    model.train()
    metrics_calc.reset()
    
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)
        
        # PERF: Wrap forward and backward in AMP autocast and scaler
        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=device.type=='cuda', dtype=torch.float32):
            pred_hr = model(lr_batch)
            
            # FIX: Shape guard to ensure pred_hr and hr_batch dimensions match
            if pred_hr.shape[-2:] != hr_batch.shape[-2:]:
                h = min(pred_hr.shape[-2], hr_batch.shape[-2])
                w = min(pred_hr.shape[-1], hr_batch.shape[-1])
                pred_hr = pred_hr[..., :h, :w]
                hr_batch = hr_batch[..., :h, :w]
            
            # Calculate loss
            loss_components = sr_loss(pred_hr.float(), hr_batch, perceptual, perceptual_weight=perceptual_weight, fft_weight=fft_weight)
            
            # Handle both dict (new) and tuple (old) formats
            if isinstance(loss_components, dict):
                total_loss_val = loss_components['total_loss']
            else:
                total_loss_val, amp_loss, phase_loss = loss_components
        
        # PERF: Scale loss and backward
        scaler.scale(total_loss_val).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # PERF: Step profiler if enabled
        if profiler:
            profiler.step()
        
        # Update metrics
        metrics_calc.update(pred_hr, hr_batch, loss_components)
        
        # Accumulate loss weighted by batch size for proper averaging
        batch_size = lr_batch.shape[0]
        total_loss += total_loss_val.item() * batch_size
        total_samples += batch_size
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss_val.item():.6f}')
    
    # Return average loss weighted by actual number of samples processed
    return total_loss / max(total_samples, 1)


def validate_epoch(model, val_loader, device, metrics_calc, perceptual, *, perceptual_weight: float = 0.0, fft_weight: float = 0.0):
    """
    Validate for one epoch
    
    Args:
        model: ComplexUNet model
        val_loader: Validation data loader
        device: torch device
        metrics_calc: MetricsCalculator instance
        perceptual: PerceptualLoss instance
        
    Returns:
        Average validation loss
    """
    model.eval()
    metrics_calc.reset()
    
    total_loss = 0.0
    total_samples = 0
    
    # PERF: Use autocast for validation forward pass
    with torch.no_grad():
        with autocast(device_type='cuda', enabled=device.type=='cuda', dtype=torch.float32):
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device, non_blocking=True)
                hr_batch = hr_batch.to(device, non_blocking=True)
                
                # Forward pass
                pred_hr = model(lr_batch)
                
                # FIX: Shape guard to ensure pred_hr and hr_batch dimensions match
                if pred_hr.shape[-2:] != hr_batch.shape[-2:]:
                    h = min(pred_hr.shape[-2], hr_batch.shape[-2])
                    w = min(pred_hr.shape[-1], hr_batch.shape[-1])
                    pred_hr = pred_hr[..., :h, :w]
                    hr_batch = hr_batch[..., :h, :w]
                
                # Calculate loss
                loss_components = sr_loss(pred_hr.float(), hr_batch, perceptual, perceptual_weight=perceptual_weight, fft_weight=fft_weight)
                
                # Handle both dict (new) and tuple (old) formats
                if isinstance(loss_components, dict):
                    total_loss_val = loss_components['total_loss']
                else:
                    total_loss_val, amp_loss, phase_loss = loss_components
                
                # Update metrics
                metrics_calc.update(pred_hr, hr_batch, loss_components)
                
                # Accumulate loss weighted by batch size
                batch_size = lr_batch.shape[0]
                total_loss += total_loss_val.item() * batch_size
                total_samples += batch_size
    
    return total_loss / max(total_samples, 1)


def log_images_to_tensorboard(writer, lr_batch, hr_batch, pred_batch, epoch, num_images=1, fixed_range=(0.001, 10.0)):
    """
    Logs SAR image visualizations to TensorBoard with consistent normalization.
    - Uses adaptive normalization for clear visualization.
    - Calculates VV and VH amplitude correctly.
    - Upscales LR image for consistent visualization size.
    - All images (Raw, LR, HR, SR) come from the same source patch for meaningful comparison.
    
    Args:
        lr_batch: LR 4-channel real data (batch_size, 4, h, w)
        hr_batch: HR 4-channel real data (batch_size, 4, H, W) 
        pred_batch: SR 4-channel real data (batch_size, 4, H, W)
        epoch: Current training epoch
        num_images: Number of images to display (typically 1 for clarity)
        fixed_range: Tuple of (min_db, max_db) for reference (not used with adaptive normalization)
    """
    num_images = min(num_images, lr_batch.shape[0])
    hr_size = (hr_batch.shape[2], hr_batch.shape[3]) # e.g., (512, 256)
    
    # --- 1. Correctly calculate amplitude for each polarization (VV and VH) ---
    # Note: torch.hypot(real, imag) is a robust way to compute sqrt(real**2 + imag**2)
    
    # Raw HR (reconstruct complex from HR target for consistency)
    raw_amp_vv = torch.hypot(hr_batch[:num_images, 0], hr_batch[:num_images, 1])  # Same as HR target
    raw_amp_vh = torch.hypot(hr_batch[:num_images, 2], hr_batch[:num_images, 3])  # Same as HR target
    
    # LR (4-channel real)
    lr_amp_vv = torch.hypot(lr_batch[:num_images, 0], lr_batch[:num_images, 1])
    lr_amp_vh = torch.hypot(lr_batch[:num_images, 2], lr_batch[:num_images, 3])
    
    # HR Target (4-channel real)
    hr_amp_vv = torch.hypot(hr_batch[:num_images, 0], hr_batch[:num_images, 1])
    hr_amp_vh = torch.hypot(hr_batch[:num_images, 2], hr_batch[:num_images, 3])
    
    # SR Prediction (4-channel real)
    pred_amp_vv = torch.hypot(pred_batch[:num_images, 0], pred_batch[:num_images, 1])
    pred_amp_vh = torch.hypot(pred_batch[:num_images, 2], pred_batch[:num_images, 3])

    # --- 2. Upscale LR images for visualization purposes ---
    # This does not affect training, only what's shown on TensorBoard.
    lr_amp_vv_up = F.interpolate(lr_amp_vv.unsqueeze(1), size=hr_size, mode='bilinear', align_corners=False).squeeze(1)
    lr_amp_vh_up = F.interpolate(lr_amp_vh.unsqueeze(1), size=hr_size, mode='bilinear', align_corners=False).squeeze(1)

    # --- 3. Normalize images with fixed dB range for consistency ---
    def normalize_for_display_fixed_range(amp_tensor, min_db, max_db):
        # Log scale for better contrast in SAR images
        log_amp = 20 * torch.log10(amp_tensor + 1e-8)  # Convert to dB
        # Normalize to [0, 1] using fixed dB range
        norm_amp = (log_amp - min_db) / (max_db - min_db)
        # Clamp to [0, 1] and keep as float32 for TensorBoard
        norm_amp = norm_amp.clamp(0, 1)
        # Check for NaN/Inf and replace with 0
        norm_amp = torch.where(torch.isfinite(norm_amp), norm_amp, torch.tensor(0.0, device=norm_amp.device))
        # Add channel dimension for grayscale images
        return norm_amp.unsqueeze(1)
    
    def normalize_for_display_adaptive(amp_tensor):
        # Adaptive normalization for individual tensor
        log_amp = 20 * torch.log10(amp_tensor + 1e-8)
        min_db = log_amp.min()
        max_db = log_amp.max()
        norm_amp = (log_amp - min_db) / (max_db - min_db + 1e-8)
        norm_amp = norm_amp.clamp(0, 1)
        # Check for NaN/Inf and replace with 0
        norm_amp = torch.where(torch.isfinite(norm_amp), norm_amp, torch.tensor(0.0, device=norm_amp.device))
        return norm_amp.unsqueeze(1)

    min_db, max_db = fixed_range
    
    # Debug: Print actual data ranges to understand normalization issues
    print(f"Debug - Amplitude ranges:")
    print(f"  LR VV: {lr_amp_vv.min():.6f} - {lr_amp_vv.max():.6f}")
    print(f"  HR VV: {hr_amp_vv.min():.6f} - {hr_amp_vv.max():.6f}")
    print(f"  Pred VV: {pred_amp_vv.min():.6f} - {pred_amp_vv.max():.6f}")
    print(f"  Fixed dB range: {min_db} - {max_db}")

    # --- VV Polarization Visualization ---
    # Note: LR_Input_Upscaled shows degraded version of HR (4x4 block averaging + bilinear upscale)
    # It should appear blurred compared to HR_Target due to information loss in LR generation
    try:
        # Use adaptive normalization for clear visualization
        vv_hr_norm = normalize_for_display_adaptive(hr_amp_vv)  # HR Target (ground truth)
        vv_lr_norm = normalize_for_display_adaptive(lr_amp_vv_up)
        vv_pred_norm = normalize_for_display_adaptive(pred_amp_vv)
        
        writer.add_images('SAR_Images_VV/1_HR_Target', vv_hr_norm, epoch)
        writer.add_images('SAR_Images_VV/2_LR_Input_Upscaled', vv_lr_norm, epoch)
        writer.add_images('SAR_Images_VV/3_SR_Prediction', vv_pred_norm, epoch)
    except Exception as e:
        print(f"Warning: Failed to log VV images to TensorBoard: {e}")

    # --- VH Polarization Visualization ---
    try:
        vh_hr_norm = normalize_for_display_adaptive(hr_amp_vh)  # HR Target (ground truth)
        vh_lr_norm = normalize_for_display_adaptive(lr_amp_vh_up)
        vh_pred_norm = normalize_for_display_adaptive(pred_amp_vh)
        
        writer.add_images('SAR_Images_VH/1_HR_Target', vh_hr_norm, epoch)
        writer.add_images('SAR_Images_VH/2_LR_Input_Upscaled', vh_lr_norm, epoch)
        writer.add_images('SAR_Images_VH/3_SR_Prediction', vh_pred_norm, epoch)
    except Exception as e:
        print(f"Warning: Failed to log VH images to TensorBoard: {e}")
def log_complex_statistics(writer, tensor, tag, epoch):
    if not torch.is_complex(tensor):
        if tensor.shape[1] < 2:
            # Skip phase stats for single-channel tensors
            amplitude = tensor[:, 0]
            writer.add_scalar(f'{tag}/Amplitude_Mean', amplitude.mean().item(), epoch)
            writer.add_scalar(f'{tag}/Amplitude_Std', amplitude.std().item(), epoch)
            writer.add_scalar(f'{tag}/Amplitude_Max', amplitude.max().item(), epoch)
            writer.add_histogram(f'{tag}/Amplitude_Distribution', amplitude, epoch)
            return
        tensor = torch.complex(tensor[:,0], tensor[:,1])
    amplitude = torch.abs(tensor)
    phase = torch.angle(tensor)
    
    # Log amplitude stats
    writer.add_scalar(f'{tag}/VV_Amplitude_Mean', amplitude.mean().item(), epoch)
    writer.add_scalar(f'{tag}/VV_Amplitude_Std', amplitude.std().item(), epoch)
    writer.add_scalar(f'{tag}/VV_Amplitude_Max', amplitude.max().item(), epoch)
    
    # Log phase stats
    writer.add_scalar(f'{tag}/VV_Phase_Mean', phase.mean().item(), epoch)
    writer.add_scalar(f'{tag}/VV_Phase_Std', phase.std().item(), epoch)
    
    # Skip histogram if tensor contains NaN/Inf to avoid TensorBoard crash # FIX
    if torch.isfinite(amplitude).all() and torch.isfinite(phase).all():
        writer.add_histogram(f'{tag}/VV_Amplitude_Distribution', amplitude, epoch)
        writer.add_histogram(f'{tag}/VV_Phase_Distribution', phase, epoch)
    else:
        print(f"Warning: Skipping histogram for {tag} at epoch {epoch} due to NaN/Inf values")


def setup_tensorboard_logging(log_dir: str = None) -> SummaryWriter:
    """
    Setup TensorBoard logging
    
    Args:
        log_dir: Directory for TensorBoard logs
        
    Returns:
        SummaryWriter instance
    """
    if log_dir is None:
        # Create timestamped log directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/sar_sr_{timestamp}"
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    print(f"Run: tensorboard --logdir={log_dir}")
    
    return writer


def train_model(
    data_dir: str = r"D:\Sentinel-1\data\patches\zero_filtered",
    model_save_path: str = r"D:\Sentinel-1\model\acswin_unet_pp.pth",
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    early_stop_patience: int = 7,
    early_stop_threshold: float = 1e-4,
    enable_tensorboard: bool = True,
    enable_perceptual: bool = True,
    perceptual_weight: float = 0.0,
    fft_weight: float = 0.02,
    tensorboard_log_dir: str = None,
    profile_steps: int = 0,  # PERF: Added for profiling control
    *,
    use_cache: bool = True,
    gpu_degrade: bool = False,
    num_workers: int = 0,
    max_samples: int = None,
    resume_checkpoint_path: str = None,
):
    """
    Main training function with TensorBoard logging
    
    Args:
        data_dir: Data directory containing SAR patches
        model_save_path: Path to save trained model
        num_epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Adam optimizer learning rate
        early_stop_patience: Early stopping patience
        early_stop_threshold: Early stopping threshold
        enable_tensorboard: Enable TensorBoard logging
        tensorboard_log_dir: Custom TensorBoard log directory
    """
    print("Starting SAR Super-Resolution Training")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup TensorBoard logging
    writer = None
    if enable_tensorboard:
        writer = setup_tensorboard_logging(tensorboard_log_dir)
    
    # Create model
    print("\nCreating ComplexUNet model...")
    model = create_model()
    model.to(device)
    
    # Instantiate PerceptualLoss only when enabled and weight > 0
    perceptual = None
    if enable_perceptual and perceptual_weight > 0.0:
        perceptual = PerceptualLoss().to(device)
    
    # Log model architecture to TensorBoard (commented out due to trace errors)
    if writer:
        dummy_input = torch.randn(1, 4, 64, 128).to(device) # Adjusted dummy input to match real-valued shape
        
        # try:
        #     writer.add_graph(model, dummy_input)
        # except Exception as e:
        #     print(f"Warning: Could not log model graph: {e}")
        print("Note: Model graph logging disabled to avoid trace errors")
    
    # DataLoader workers already passed via parameter; keeping as is
    
    # Create dataloaders
    print(f"\nLoading data from {data_dir}...")
    if max_samples is not None:
        print(f"Limiting dataset to {max_samples} samples")
    train_loader, val_loader = create_dataloaders(data_dir, batch_size, num_workers, use_cache=use_cache, gpu_degrade=gpu_degrade, max_samples=max_samples)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup optimizer and scheduler for better convergence
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = sched.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_psnr': [], 'val_psnr': []}
    
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume from saved epoch
        start_epoch = checkpoint.get('epoch', 0)
        history = checkpoint.get('history', history)
        
        # Restore hyperparameters if available
        saved_hyperparams = checkpoint.get('hyperparameters', {})
        if saved_hyperparams:
            print("Restoring saved hyperparameters:")
            for key, value in saved_hyperparams.items():
                if key in locals():
                    locals()[key] = value
                    print(f"  {key}: {value}")
        
        # Restore TensorBoard log directory if available
        saved_tb_dir = checkpoint.get('tensorboard_log_dir')
        if saved_tb_dir and tensorboard_log_dir is None:
            tensorboard_log_dir = saved_tb_dir
            print(f"Continuing with TensorBoard directory: {tensorboard_log_dir}")
        
        print(f"Resumed from epoch {start_epoch}")
        print(f"Previous validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # PERF: Setup AMP scaler
    scaler = GradScaler()
    
    # PERF: Setup profiler if requested
    profiler = setup_profiler(steps=profile_steps) if profile_steps > 0 else None
    outer_ctx = profiler if profiler is not None else nullcontext()  # PERF: nullcontext when no profiler
    
    # Log hyperparameters to TensorBoard
    if writer:
        hparams = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'early_stop_patience': early_stop_patience,
            'early_stop_threshold': early_stop_threshold,
            'model_params': model.count_parameters(),
            'device': str(device)
        }
        writer.add_hparams(hparams, {})
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=early_stop_patience, 
        min_delta=early_stop_threshold,
        restore_best_weights=True
    )
    
    # Metrics calculators
    train_metrics = MetricsCalculator()
    val_metrics = MetricsCalculator()
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    # PERF: Wrap training loop with profiler (or do nothing)
    with outer_ctx:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Training
            print("Training...")
            train_loss = train_epoch(model, train_loader, optimizer, device, train_metrics, perceptual, scaler, profiler, perceptual_weight=perceptual_weight, fft_weight=fft_weight)
            train_avg_metrics = train_metrics.get_average_metrics()
            
            # Validation
            print("Validating...")
            val_loss = validate_epoch(model, val_loader, device, val_metrics, perceptual, perceptual_weight=perceptual_weight, fft_weight=fft_weight)
            val_avg_metrics = val_metrics.get_average_metrics()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_psnr'].append(train_avg_metrics.get('psnr', 0))
            history['val_psnr'].append(val_avg_metrics.get('psnr', 0))
            
            # Step scheduler (CosineAnnealingLR doesn't need validation loss)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Val', val_loss, epoch)
                
                # Dual-polarization PSNR logging
                writer.add_scalar('PSNR/Train_VV', train_avg_metrics.get('psnr_vv', 0), epoch)
                writer.add_scalar('PSNR/Train_VH', train_avg_metrics.get('psnr_vh', 0), epoch)
                writer.add_scalar('PSNR/Train_Avg', train_avg_metrics.get('psnr_avg', 0), epoch)
                writer.add_scalar('PSNR/Val_VV', val_avg_metrics.get('psnr_vv', 0), epoch)
                writer.add_scalar('PSNR/Val_VH', val_avg_metrics.get('psnr_vh', 0), epoch)
                writer.add_scalar('PSNR/Val_Avg', val_avg_metrics.get('psnr_avg', 0), epoch)
                
                # Legacy PSNR (VV) for backward compatibility
                writer.add_scalar('PSNR/Train', train_avg_metrics.get('psnr', 0), epoch)
                writer.add_scalar('PSNR/Val', val_avg_metrics.get('psnr', 0), epoch)
                
                writer.add_scalar('RMSE/Val', val_avg_metrics.get('rmse', 0), epoch)
                writer.add_scalar('CPIF/Val', val_avg_metrics.get('cpif', 0), epoch)
                writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                # Log images
                if epoch % 5 == 0 or epoch == num_epochs - 1: # Log every 5 epochs or at the end
                    # -------- TensorBoard sample preparation --------
                    num_tb_imgs = 1  # Use only 1 image for cleaner display
                    
                    # Get consistent sample from same patch
                    sample_idx = 0  # Use first validation sample for consistency
                    
                    # Get LR/HR pair from validation dataset (same source)
                    lr_sample, hr_sample = val_loader.dataset[sample_idx]
                    lr_batch_sample = lr_sample.unsqueeze(0).to(device)  # Add batch dimension
                    hr_batch_sample = hr_sample.unsqueeze(0).to(device)  # Add batch dimension
                    
                    # Generate prediction from the same LR data
                    with torch.no_grad():
                        pred_hr_sample = model(lr_batch_sample)
                    
                    # Print amplitude min/max (scale sanity)
                    with torch.no_grad():
                        lr_vv = torch.hypot(lr_batch_sample[:, 0], lr_batch_sample[:, 1])
                        hr_vv = torch.hypot(hr_batch_sample[:, 0], hr_batch_sample[:, 1])
                        pr_vv = torch.hypot(pred_hr_sample[:, 0], pred_hr_sample[:, 1])
                        lr_vh = torch.hypot(lr_batch_sample[:, 2], lr_batch_sample[:, 3])
                        hr_vh = torch.hypot(hr_batch_sample[:, 2], hr_batch_sample[:, 3])
                        pr_vh = torch.hypot(pred_hr_sample[:, 2], pred_hr_sample[:, 3])
                        print(f"  VV amp min/max  - LR: {lr_vv.min().item():.4f}/{lr_vv.max().item():.4f}, HR: {hr_vv.min().item():.4f}/{hr_vv.max().item():.4f}, Pred: {pr_vv.min().item():.4f}/{pr_vv.max().item():.4f}")
                        print(f"  VH amp min/max  - LR: {lr_vh.min().item():.4f}/{lr_vh.max().item():.4f}, HR: {hr_vh.min().item():.4f}/{hr_vh.max().item():.4f}, Pred: {pr_vh.min().item():.4f}/{pr_vh.max().item():.4f}")

                    # Log side-by-side LR → HR → SR (all from same source)
                    log_images_to_tensorboard(writer, lr_batch_sample, hr_batch_sample, pred_hr_sample, epoch, num_images=1)
                    log_complex_statistics(writer, hr_batch_sample, 'HR_Target', epoch)
                    log_complex_statistics(writer, pred_hr_sample, 'SR_Prediction', epoch)
                    
                    # Monitor coherence improvement for dual-pol data
                    # if hr_batch_sample.shape[1] == 2: # Removed coherence logging
                    #     from utils import calculate_cross_pol_coherence
                    #     pred_vv, pred_vh = pred_hr_sample[:, 0], pred_hr_sample[:, 1]
                    #     gt_vv, gt_vh = hr_batch_sample[:, 0], hr_batch_sample[:, 1]
                        
                    #     pred_coherence = calculate_cross_pol_coherence(pred_vv, pred_vh)
                    #     gt_coherence = calculate_cross_pol_coherence(gt_vv, gt_vh)
                        
                    #     writer.add_scalar('Coherence/Predicted', pred_coherence, epoch)
                    #     writer.add_scalar('Coherence/Ground_Truth', gt_coherence, epoch)
                    #     writer.add_scalar('Coherence/Error', abs(pred_coherence - gt_coherence), epoch)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1} Results ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Train PSNR - VV: {train_avg_metrics.get('psnr_vv', 0):.2f} dB, VH: {train_avg_metrics.get('psnr_vh', 0):.2f} dB, Avg: {train_avg_metrics.get('psnr_avg', 0):.2f} dB")
            print(f"  Val PSNR - VV: {val_avg_metrics.get('psnr_vv', 0):.2f} dB, VH: {val_avg_metrics.get('psnr_vh', 0):.2f} dB, Avg: {val_avg_metrics.get('psnr_avg', 0):.2f} dB")
            # Per-epoch amplitude sanity check on a fixed validation sample
            try:
                sample_idx = 0
                lr_sample, hr_sample = val_loader.dataset[sample_idx]
                lr_b = lr_sample.unsqueeze(0).to(device)
                hr_b = hr_sample.unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_b = model(lr_b)
                # VV
                lr_vv = torch.hypot(lr_b[:,0], lr_b[:,1])
                hr_vv = torch.hypot(hr_b[:,0], hr_b[:,1])
                pr_vv = torch.hypot(pred_b[:,0], pred_b[:,1])
                # VH
                lr_vh = torch.hypot(lr_b[:,2], lr_b[:,3])
                hr_vh = torch.hypot(hr_b[:,2], hr_b[:,3])
                pr_vh = torch.hypot(pred_b[:,2], pred_b[:,3])
                print(f"  VV amp min/max  - LR: {lr_vv.min().item():.4f}/{lr_vv.max().item():.4f}, HR: {hr_vv.min().item():.4f}/{hr_vv.max().item():.4f}, Pred: {pr_vv.min().item():.4f}/{pr_vv.max().item():.4f}")
                print(f"  VH amp min/max  - LR: {lr_vh.min().item():.4f}/{lr_vh.max().item():.4f}, HR: {hr_vh.min().item():.4f}/{hr_vh.max().item():.4f}, Pred: {pr_vh.min().item():.4f}/{pr_vh.max().item():.4f}")
            except Exception:
                pass
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best model! Saving to {model_save_path}")
                
                # Ensure directory exists
                Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save model checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'history': history,
                    'model_params': model.count_parameters(),
                    'hyperparameters': {
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'early_stop_patience': early_stop_patience,
                        'early_stop_threshold': early_stop_threshold,
                        'enable_perceptual': enable_perceptual,
                        'max_samples': max_samples,
                        'use_cache': use_cache,
                        'gpu_degrade': gpu_degrade,
                        'num_workers': num_workers
                    },
                    'tensorboard_log_dir': tensorboard_log_dir
                }
                torch.save(checkpoint, model_save_path)
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_save_path}")
    
    # Final validation metrics
    print("\nFinal Validation Metrics:")
    val_metrics.print_metrics("Final")
    
    if writer:
        writer.close()
    
    return model, history


def test_model(model_path: str, data_dir: str, max_samples: int = None):
    """
    Test trained model on test set
    
    Args:
        model_path: Path to saved model
        data_dir: Data directory
        max_samples: Maximum number of samples to use (should match training)
    """
    print(f"\nTesting model from {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    
    if Path(model_path).exists():
        # Fix for PyTorch 2.6 weights_only default change
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        print("Warning: Model file not found, using random weights")
    
    model.to(device)
    model.eval()
    
    # Perceptual disabled by default for test; enable only with weight > 0 via CLI
    perceptual = None
    
    # --- Load test split from cache ---
    # Use appropriate cache file based on max_samples
    if max_samples is not None:
        cache_path = Path(data_dir) / f'file_split_cache_{max_samples}.pkl'
    else:
        cache_path = Path(data_dir) / 'file_split_cache.pkl'
        
    if not cache_path.exists():
        print(f"Error: File split cache not found at {cache_path}. Please run training first to generate it.")
        return
        
    with open(cache_path, 'rb') as f:
        file_splits = pickle.load(f)
    
    # Create test dataset
    test_dataset = SARSuperResDataset(file_splits['test'], data_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Test model
    test_metrics = MetricsCalculator()
    # FIX: Pass perceptual parameter to validate_epoch
    test_loss = validate_epoch(model, test_loader, device, test_metrics, perceptual)
    
    print(f"Test Loss: {test_loss:.6f}")
    test_metrics.print_metrics("Test")


if __name__ == "__main__":
    # PERF: Enable cuDNN benchmark mode for optimized convolution performance.
    torch.backends.cudnn.benchmark = True
    
    # PERF: Add CLI argument parsing
    parser = argparse.ArgumentParser(description="SAR Super-Resolution Training")
    parser.add_argument('--batch-size-auto', action='store_true', help="Auto-adjust batch size based on GPU memory")
    parser.add_argument('--profile', type=int, default=0, help="Number of steps to profile (0 to disable)")
    parser.add_argument('--gpu-degrade', action='store_true', help="Run HR→LR degradation on GPU")
    parser.add_argument('--no-cache', action='store_true', help="Disable LR patch caching")
    parser.add_argument('--num-workers', type=int, default=0, help="DataLoader num_workers (default 0)")
    parser.add_argument('--model-type', default='swin', choices=['swin','base'], help='Model backbone selection (WARNING: base model has channel mismatch issues)')
    # Rebuild LR cache is handled externally by user; keeping CLI clean
    parser.add_argument('--dry-run', action='store_true', help='Run single forward pass and exit')
    parser.add_argument('--no-perceptual', action='store_true', help='Disable perceptual VGG loss (deprecated; use --perceptual-weight=0.0)')
    parser.add_argument('--perceptual-weight', type=float, default=0.0, help='Weight for perceptual loss (default 0.0 disables VGG)')
    parser.add_argument('--fft-weight', type=float, default=0.02, help='Weight for high-frequency weighted FFT loss (0.0 to disable)')
    parser.add_argument('--auto-workers', action='store_true', help='Use optimal DataLoader workers automatically')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to use (None for all data)')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint file')
    args = parser.parse_args()

    # Set multiprocessing start method to avoid CUDA init deadlocks
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select backbone at runtime
    if args.model_type == 'swin':
        from ac_swin_unet_pp import create_model as create_model_fn
    else:
        # Base model (cv_unet) has channel mismatch: expects 3 channels but dataset returns 4
        print("ERROR: --model-type base is currently not supported due to channel mismatch.")
        print("cv_unet expects 3 input channels (VV-Re, VV-Im, |VH|) but dataset returns 4 channels (VV-Re, VV-Im, VH-Re, VH-Im).")
        print("Use --model-type swin instead, or modify cv_unet to accept 4 channels.")
        sys.exit(1)
    globals()['create_model'] = create_model_fn

    # Configuration
    config = {
        'data_dir': r"D:\Sentinel-1\data\patches\zero_filtered",
        'model_save_path': r"D:\Sentinel-1\model\acswin_unet_pp.pth",
        'num_epochs': 100,  # Increased from 50 for better convergence
        'batch_size': 32,   # Increased from 24 to 32 for better convergence
        'learning_rate': 1e-4,
        'early_stop_patience': 10,  # Increased patience for longer training
        'early_stop_threshold': 1e-4,
        'enable_tensorboard': True,
        'use_cache': not args.no_cache,
        'gpu_degrade': args.gpu_degrade,
        'num_workers': get_optimal_workers() if args.auto_workers else args.num_workers,
        'tensorboard_log_dir': None,  # Will use timestamped directory
        'enable_perceptual': not args.no_perceptual,
        'perceptual_weight': args.perceptual_weight,
        'fft_weight': args.fft_weight,
        'max_samples': args.max_samples,
        'resume_checkpoint_path': args.resume,
    }
    
    # PERF: Auto-adjust batch size if flag set
    if args.batch_size_auto:
        config['batch_size'] = auto_adjust_batch_size(config['batch_size'])
        print(f"Auto-adjusted batch size to {config['batch_size']}")
    
    if args.dry_run:
        model = create_model_fn().to(device)
        dummy = torch.randn(2,4,64,128,device=device)
        with torch.no_grad():
            out = model(dummy)
        print("Header  |  PSNR  |  RMSE  |  CPIF")
        exit(0)

    print("Complex U-Net SAR Super-Resolution Training")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    try:
        # Train model with profile_steps
        model, history = train_model(**config, profile_steps=args.profile)
        
        # Test model
        test_model(config['model_save_path'], config['data_dir'], config['max_samples'])
        
        print("\nTraining and testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()