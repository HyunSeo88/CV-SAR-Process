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
from data_cache import load_or_compute_lr
# PERF: Import speed utils
from speed_utils import get_optimal_workers, auto_adjust_batch_size, setup_profiler


class SARSuperResDataset(Dataset):
    """
    SAR Super-Resolution Dataset
    
    Loads dual-pol complex SAR patches for super-resolution training.
    Assumes directory structure:
    data_dir/
    ├── LR/  # Low resolution patches (64x128)
    ├── HR/  # High resolution patches (256x512)
    
    Each patch is a .npy file with shape (2, H, W) complex64
    """
    
    def __init__(self, data_dir: str, split: str = 'train', lr_size: Tuple[int, int] = (128, 64),
                 hr_size: Tuple[int, int] = (512, 256), *, use_cache: bool = True, gpu_degrade: bool = False):  # Corrected HR size

        """
        Initialize SAR dataset
        
        Args:
            data_dir: Base data directory (D:\Sentinel-1\data\processed_2)
            split: Dataset split ('train', 'val', 'test')
            lr_size: Low resolution patch size (height, width)
            hr_size: High resolution patch size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.use_cache = use_cache
        self.gpu_degrade = gpu_degrade
        
        # For now, we'll simulate LR/HR pairs from existing data
        # In practice, you would have pre-generated LR patches
        self.hr_files = self._find_sar_patches()
        
        print(f"Found {len(self.hr_files)} SAR patches for {split} split")
        
        if len(self.hr_files) == 0:
            print(f"Warning: No SAR patches found in {self.data_dir}")
            print("Creating synthetic dataset for testing...")
            self._create_synthetic_data()
    
    def _find_sar_patches(self):
        pattern = "*_dual_pol_complex_*.npy"
        cache_path = self.data_dir / f'valid_hr_files_{self.split}.pkl'
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                hr_files = pickle.load(f)
            print(f"Loaded {len(hr_files)} cached valid patches for {self.split}")
            return hr_files  # No reshuffle needed if cached post-split

        potential_files = list(self.data_dir.rglob(pattern))
        max_total_patches = 20000  
        buffer_size = 30000
        hr_files = []
        filtered_count = 0
        for file in potential_files[buffer_size:max_total_patches+buffer_size]:
            hr_data = np.load(file).astype(np.complex64)
            if hr_data.shape[0] != 2: 
                filtered_count += 1
                continue
            vv, vh = hr_data[0], hr_data[1]
            cross = np.mean(vv * np.conj(vh))
            pow_vv = np.mean(np.abs(vv)**2)
            pow_vh = np.mean(np.abs(vh)**2)
            coherence = np.abs(cross) / np.sqrt(pow_vv * pow_vh + 1e-8)
            if coherence >= 0.0:     # 실제 필터링 제거 # Relaxed filter to avoid data scarcity
                hr_files.append(file)
            else:
                filtered_count += 1
        
        print(f"Filtered {filtered_count} marine patches ({filtered_count/len(potential_files)*100:.1f}%)")
        
        # Shuffle and split
        np.random.shuffle(hr_files)
        n_total = len(hr_files)
        if self.split == 'train':
            split_files = hr_files[:int(0.8 * n_total)]
        elif self.split == 'val':
            split_files = hr_files[int(0.8 * n_total):int(0.9 * n_total)]
        else:  # test
            split_files = hr_files[int(0.9 * n_total):]
        
        # Cache split for reuse
        with open(cache_path, 'wb') as f:
            pickle.dump(split_files, f)
        
        return split_files
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
        assert hr.shape[-2:] == self.hr_size, "HR shape mismatch"

        # 1) 전력합 → √: divisor_override=1 로 ‘합’을 바로 계산
        power = (hr.real**2 + hr.imag**2).unsqueeze(0)       # (1,2,H,W)
        power_sum = F.avg_pool2d(power, scale, stride=scale,
                             divisor_override=1)         # 합 = ∑|z|²
        amp_lr = power_sum.sqrt().squeeze(0)                 # (2,H/4,W/4)
        
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
        
        # FIX: Enforce exact HR size by cropping (handles patches larger than 512x256)
        hr_data = hr_data[:, :self.hr_size[0], :self.hr_size[1]]  # Corrected cropping to include channel dim explicitly
        
        # Ensure correct shape and data type
        if hr_data.shape != (2, *self.hr_size):
            # Transpose if dimensions are swapped (512,256) vs (256,512)
            if hr_data.shape == (2, self.hr_size[1], self.hr_size[0]):
                hr_data = np.transpose(hr_data, (0, 2, 1))  # Swap height/width
            else:
                # Apply padding if needed
                if len(hr_data.shape) == 3 and hr_data.shape[0] == 2:
                    pad_h = max(0, self.hr_size[0] - hr_data.shape[1])
                    pad_w = max(0, self.hr_size[1] - hr_data.shape[2])
                    if pad_h > 0 or pad_w > 0:
                        # Change: Use reflective padding instead of constant zeros
                        # Reason: Reflective padding preserves phase coherence and speckle patterns in SAR images,
                        # avoiding artificial discontinuities that zero-padding introduces, which can create edge artifacts
                        # in super-resolution outputs. This maintains the interferometric properties better.
                        hr_data = np.pad(hr_data, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')
        
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

    def fetch_raw_patch(self, idx):
        """
        Load a raw complex SAR patch, resized to the standard HR size.
        This ensures consistency for visualization tasks like TensorBoard logging.

        Args:
            idx: Dataset index
        Returns:
            np.ndarray with shape (2, H, W) complex64 representing the processed patch.
        """
        if hasattr(self, 'synthetic_data'):
            hr_data = self.synthetic_data[idx]
        else:
            hr_file = self.hr_files[idx]
            hr_data = np.load(hr_file)

        # FIX: Enforce exact HR size by cropping/padding to handle inconsistent source patches
        hr_data = hr_data[:, :self.hr_size[0], :self.hr_size[1]]

        # Ensure correct shape and data type through transpose or padding
        if hr_data.shape != (2, *self.hr_size):
            if hr_data.shape == (2, self.hr_size[1], self.hr_size[0]):
                hr_data = np.transpose(hr_data, (0, 2, 1))
            else:
                if len(hr_data.shape) == 3 and hr_data.shape[0] == 2:
                    pad_h = max(0, self.hr_size[0] - hr_data.shape[1])
                    pad_w = max(0, self.hr_size[1] - hr_data.shape[2])
                    if pad_h > 0 or pad_w > 0:
                        hr_data = np.pad(hr_data, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')

        # Ensure dtype consistency
        if hr_data.dtype != np.complex64:
            hr_data = hr_data.astype(np.complex64)
            
        return hr_data


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


def create_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 0, *, use_cache: bool = True, gpu_degrade: bool = False):
    """
    Create train and validation dataloaders
    """
    # Safety: warn when using worker processes with CUDA
    if num_workers != 0:
        print(f"[Warning] num_workers set to {num_workers}. CUDA DataLoader workers may clash with GPU ops. Consider keeping it 0 unless necessary.")
    

    # Create datasets
    train_dataset = SARSuperResDataset(data_dir, split='train', use_cache=use_cache, gpu_degrade=gpu_degrade)
    val_dataset = SARSuperResDataset(data_dir, split='val', use_cache=use_cache, gpu_degrade=gpu_degrade)
    
    # PERF: Add persistent_workers and pin_memory for faster DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, metrics_calc, perceptual, scaler, profiler=None):
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
        with autocast(device_type='cuda'):
            pred_hr = model(lr_batch)
            
            # FIX: Shape guard to ensure pred_hr and hr_batch dimensions match
            if pred_hr.shape[-2:] != hr_batch.shape[-2:]:
                h = min(pred_hr.shape[-2], hr_batch.shape[-2])
                w = min(pred_hr.shape[-1], hr_batch.shape[-1])
                pred_hr = pred_hr[..., :h, :w]
                hr_batch = hr_batch[..., :h, :w]
            
            # Calculate loss
            loss_components = sr_loss(pred_hr.float(), hr_batch, perceptual)
            
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


def validate_epoch(model, val_loader, device, metrics_calc, perceptual):
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
        with autocast(device_type='cuda'):
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
                loss_components = sr_loss(pred_hr.float(), hr_batch, perceptual)
                
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


def log_images_to_tensorboard(writer, raw_batch, lr_batch, hr_batch, pred_batch, epoch, num_images=4):
    # Convert to amplitude for visualization
        # --- Raw patch amplitude (VV) ---
    raw_amp = torch.abs(raw_batch[:num_images, 0])  # raw_batch is complex64 [B, Pol, H, W]
    # --- LR amplitude (VV) ---
    lr_amp = torch.sqrt(lr_batch[:num_images, 0]**2 + lr_batch[:num_images, 1]**2)
    # FIX: Handle real 2-channel tensors properly for amplitude calculation
    hr_amp = torch.abs(torch.complex(hr_batch[:num_images, 0], hr_batch[:num_images, 1]))
    pred_amp = torch.abs(torch.complex(pred_batch[:num_images, 0], pred_batch[:num_images, 1]))
    
    # Normalize for display (log scale for SAR)
    lr_amp_log = torch.log10(lr_amp + 1e-8)
    hr_amp_log = torch.log10(hr_amp + 1e-8)
    pred_amp_log = torch.log10(pred_amp + 1e-8)
    raw_amp_log  = torch.log10(raw_amp + 1e-8)
    
    # Normalize to [0, 1]
    lr_norm = (lr_amp_log - lr_amp_log.min()) / (lr_amp_log.max() - lr_amp_log.min() + 1e-8)
    hr_norm = (hr_amp_log - hr_amp_log.min()) / (hr_amp_log.max() - hr_amp_log.min() + 1e-8)
    pred_norm = (pred_amp_log - pred_amp_log.min()) / (pred_amp_log.max() - pred_amp_log.min() + 1e-8)
    raw_norm  = (raw_amp_log  - raw_amp_log.min())  / (raw_amp_log.max()  - raw_amp_log.min()  + 1e-8)
    
    # Add channel dim
    lr_norm  = lr_norm.unsqueeze(1)
    hr_norm  = hr_norm.unsqueeze(1)
    pred_norm = pred_norm.unsqueeze(1)
    raw_norm  = raw_norm.unsqueeze(1)
    
    # Convert to uint8 PNG-friendly tensors to reduce log size
    lr_uint8  = (lr_norm  * 255).clamp(0, 255).to(torch.uint8)
    hr_uint8  = (hr_norm  * 255).clamp(0, 255).to(torch.uint8)
    pred_uint8= (pred_norm * 255).clamp(0, 255).to(torch.uint8)
    raw_uint8 = (raw_norm * 255).clamp(0, 255).to(torch.uint8)

    # Log
    writer.add_images('SAR_Images/Raw_VV', raw_uint8, epoch)
    writer.add_images('SAR_Images/LR_Input_VV', lr_uint8, epoch)
    writer.add_images('SAR_Images/HR_Target_VV', hr_uint8, epoch)
    writer.add_images('SAR_Images/SR_Prediction_VV', pred_uint8, epoch)


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
    data_dir: str = r"D:\Sentinel-1\data\processed_2",
    model_save_path: str = r"D:\Sentinel-1\model\cv_unet.pth",
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    early_stop_patience: int = 7,
    early_stop_threshold: float = 1e-4,
    enable_tensorboard: bool = True,
    enable_perceptual: bool = True,
    tensorboard_log_dir: str = None,
    profile_steps: int = 0,  # PERF: Added for profiling control
    *,
    use_cache: bool = True,
    gpu_degrade: bool = False,
    num_workers: int = 0,
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
    
    # Instantiate PerceptualLoss only when enabled
    perceptual = PerceptualLoss().to(device) if enable_perceptual else None
    
    # Log model architecture to TensorBoard
    if writer:
        dummy_input = torch.randn(1, 4, 64, 128).to(device) # Adjusted dummy input to match real-valued shape
        
        try:
            writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    # DataLoader workers already passed via parameter; keeping as is
    
    # Create dataloaders
    print(f"\nLoading data from {data_dir}...")
    train_loader, val_loader = create_dataloaders(data_dir, batch_size, num_workers, use_cache=use_cache, gpu_degrade=gpu_degrade)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup optimizer and scheduler for better convergence
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = sched.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=3,      # Reduce LR after 3 epochs without improvement
        factor=0.5,      # Reduce LR by half
        min_lr=1e-6      # Minimum learning rate

    )
    
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
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_psnr': [],
        'val_psnr': []
    }
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    # PERF: Wrap training loop with profiler (or do nothing)
    with outer_ctx:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Training
            print("Training...")
            train_loss = train_epoch(model, train_loader, optimizer, device, train_metrics, perceptual, scaler, profiler)
            train_avg_metrics = train_metrics.get_average_metrics()
            
            # Validation
            print("Validating...")
            val_loss = validate_epoch(model, val_loader, device, val_metrics, perceptual)
            val_avg_metrics = val_metrics.get_average_metrics()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_psnr'].append(train_avg_metrics.get('psnr', 0))
            history['val_psnr'].append(val_avg_metrics.get('psnr', 0))
            
            # Step scheduler based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Val', val_loss, epoch)
                writer.add_scalar('PSNR/Train', train_avg_metrics.get('psnr', 0), epoch)
                writer.add_scalar('PSNR/Val', val_avg_metrics.get('psnr', 0), epoch)
                writer.add_scalar('RMSE/Val', val_avg_metrics.get('rmse', 0), epoch)
                writer.add_scalar('CPIF/Val', val_avg_metrics.get('cpif', 0), epoch)
                writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                # Log images
                if epoch % 5 == 0 or epoch == num_epochs - 1: # Log every 5 epochs or at the end
                    # -------- TensorBoard sample preparation --------
                    num_tb_imgs = 4  # number of images to log
                    # Raw HR patches (untouched)
                    raw_list = [torch.from_numpy(val_loader.dataset.fetch_raw_patch(i)) for i in range(num_tb_imgs)]
                    raw_batch_sample = torch.stack(raw_list).to(device)

                    # LR / HR pairs from validation loader
                    lr_batch_sample, hr_batch_sample = next(iter(val_loader))  # Get a sample batch
                    lr_batch_sample = lr_batch_sample.to(device)
                    hr_batch_sample = hr_batch_sample.to(device)
                    
                    with torch.no_grad():
                        pred_hr_sample = model(lr_batch_sample)
                    
                    # Log side-by-side Raw → HR → SR
                    log_images_to_tensorboard(writer, raw_batch_sample, lr_batch_sample, hr_batch_sample, pred_hr_sample, epoch)
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
            print(f"  Train PSNR: {train_avg_metrics.get('psnr', 0):.2f} dB")
            print(f"  Val PSNR: {val_avg_metrics.get('psnr', 0):.2f} dB")
            
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
                    'model_params': model.count_parameters()
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


def test_model(model_path: str, data_dir: str):
    """
    Test trained model on test set
    
    Args:
        model_path: Path to saved model
        data_dir: Data directory
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
    
    # FIX: Add perceptual loss instance for validate_epoch
    perceptual = PerceptualLoss().to(device)
    
    # Create test dataset
    test_dataset = SARSuperResDataset(data_dir, split='test')
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
    parser.add_argument('--model-type', default='swin', choices=['swin','base'], help='Model backbone selection')
    parser.add_argument('--dry-run', action='store_true', help='Run single forward pass and exit')
    parser.add_argument('--no-perceptual', action='store_true', help='Disable perceptual VGG loss')
    parser.add_argument('--auto-workers', action='store_true', help='Use optimal DataLoader workers automatically')
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
        from cv_unet import create_model as create_model_fn
    globals()['create_model'] = create_model_fn

    # Configuration
    config = {
        'data_dir': r"D:\Sentinel-1\data\processed_2",
        'model_save_path': r"D:\Sentinel-1\model\cv_unet.pth",
        'num_epochs': 100,  # Increased from 50 for better convergence
        'batch_size': 64,   # Increased from 24 to 32 for better convergence
        'learning_rate': 1e-4,
        'early_stop_patience': 10,  # Increased patience for longer training
        'early_stop_threshold': 1e-4,
        'enable_tensorboard': True,
        'use_cache': not args.no_cache,
        'gpu_degrade': args.gpu_degrade,
        'num_workers': get_optimal_workers() if args.auto_workers else args.num_workers,
        'tensorboard_log_dir': None,  # Will use timestamped directory
        'enable_perceptual': not args.no_perceptual
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
        test_model(config['model_save_path'], config['data_dir'])
        
        print("\nTraining and testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()