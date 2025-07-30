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
from pathlib import Path
from typing import Tuple, Optional
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Import our modules
from cv_unet import create_model
from utils import sr_loss, MetricsCalculator


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
                 hr_size: Tuple[int, int] = (512, 256)):
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
        
        # For now, we'll simulate LR/HR pairs from existing data
        # In practice, you would have pre-generated LR patches
        self.hr_files = self._find_sar_patches()
        
        print(f"Found {len(self.hr_files)} SAR patches for {split} split")
        
        if len(self.hr_files) == 0:
            print(f"Warning: No SAR patches found in {self.data_dir}")
            print("Creating synthetic dataset for testing...")
            self._create_synthetic_data()
    
    def _find_sar_patches(self):
        """Find all dual-pol complex patch files"""
        pattern = "*_dual_pol_complex_*.npy"
        hr_files = list(self.data_dir.rglob(pattern))
        
        # Limit total dataset to 30,000 patches for faster training
        max_total_patches = 30000
        if len(hr_files) > max_total_patches:
            print(f"Limiting dataset from {len(hr_files)} to {max_total_patches} patches")
            hr_files = hr_files[:max_total_patches]
        
        # Split dataset: 80% train, 10% val, 10% test
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(hr_files)
        
        n_total = len(hr_files)
        if self.split == 'train':
            return hr_files[:int(0.8 * n_total)]
        elif self.split == 'val':
            return hr_files[int(0.8 * n_total):int(0.9 * n_total)]
        else:  # test
            return hr_files[int(0.9 * n_total):]
    
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
        """
        Simulate LR patch from HR patch with SAR-specific degradation
        
        Enhanced degradation model:
        1. Multi-look averaging (speckle reduction)
        2. Gaussian blur for range resolution degradation
        3. Downsampling with proper anti-aliasing
        4. Additive noise
        """
        import torch.nn.functional as F
        
        # Convert to torch for processing
        hr_torch = torch.from_numpy(hr_data)
        
        # Separate magnitude and phase for processing
        hr_amp = torch.abs(hr_torch)
        hr_phase = torch.angle(hr_torch)
        
        # Multi-look averaging simulation (reduce speckle)
        # Apply averaging on magnitude to simulate multi-looking
        multilook_kernel_size = 2
        avg_amp = F.avg_pool2d(hr_amp, kernel_size=multilook_kernel_size, stride=1, 
                               padding=multilook_kernel_size//2)
        
        # Gaussian blur for range resolution degradation using Conv2d
        # Create Gaussian kernel
        kernel_size = 3
        sigma = 1.5
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        
        # Apply convolution for each channel
        blurred_amp = torch.zeros_like(avg_amp)
        for c in range(avg_amp.shape[0]):
            # Add batch and channel dimensions for conv2d
            amp_single = avg_amp[c:c+1].unsqueeze(0)  # [1, 1, H, W]
            kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
            blurred_single = F.conv2d(amp_single, kernel, padding=kernel_size//2)
            blurred_amp[c] = blurred_single.squeeze()
        
        # Phase averaging with lower weight to preserve coherence
        cos_phase = torch.cos(hr_phase)
        sin_phase = torch.sin(hr_phase)
        avg_cos = F.avg_pool2d(cos_phase, kernel_size=multilook_kernel_size, stride=1,
                               padding=multilook_kernel_size//2)
        avg_sin = F.avg_pool2d(sin_phase, kernel_size=multilook_kernel_size, stride=1,
                               padding=multilook_kernel_size//2)
        averaged_phase = torch.atan2(avg_sin, avg_cos)
        
        # Recombine amplitude and phase
        processed_complex = blurred_amp * torch.exp(1j * averaged_phase)
        
        # Downsample by factor of 4 in both dimensions
        stride_h = self.hr_size[0] // self.lr_size[0]  # 512 // 128 = 4
        stride_w = self.hr_size[1] // self.lr_size[1]  # 256 // 64 = 4
        
        lr_complex = processed_complex[:, ::stride_h, ::stride_w]
        
        # Add realistic SAR noise (multiplicative + additive)
        noise_level = 0.01
        
        # Multiplicative noise (speckle-like)
        speckle_noise = 1.0 + noise_level * torch.randn_like(lr_complex.real)
        lr_complex = lr_complex * speckle_noise.to(lr_complex.dtype)
        
        # Additive thermal noise
        thermal_noise = noise_level * torch.randn_like(lr_complex)
        lr_complex = lr_complex + thermal_noise
        
        return lr_complex.numpy()
    
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
                        hr_data = np.pad(hr_data, ((0,0), (0,pad_h), (0,pad_w)), mode='constant')
                    else:
                        return None, None
                else:
                    return None, None
        
        # Convert to complex64 if not already
        if hr_data.dtype != np.complex64:
            hr_data = hr_data.astype(np.complex64)
        
        # Generate corresponding LR patch
        lr_data = self._simulate_lr_from_hr(hr_data)
        
        # Convert to torch tensors
        lr_tensor = torch.from_numpy(lr_data)
        hr_tensor = torch.from_numpy(hr_data)
        
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


def create_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 4):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Data directory path
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SARSuperResDataset(data_dir, split='train')
    val_dataset = SARSuperResDataset(data_dir, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, metrics_calc):
    """
    Train for one epoch
    
    Args:
        model: ComplexUNet model
        train_loader: Training data loader
        optimizer: Optimizer
        device: torch device
        metrics_calc: MetricsCalculator instance
        
    Returns:
        Average training loss
    """
    model.train()
    metrics_calc.reset()
    
    total_loss = 0.0
    for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_hr = model(lr_batch)
        
        # Calculate loss
        loss_components = sr_loss(pred_hr, hr_batch)
        total_loss_val, amp_loss, phase_loss = loss_components
        
        # Backward pass
        total_loss_val.backward()
        optimizer.step()
        
        # Update metrics
        metrics_calc.update(pred_hr, hr_batch, loss_components)
        total_loss += total_loss_val.item()
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss_val.item():.6f}')
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, device, metrics_calc):
    """
    Validate for one epoch
    
    Args:
        model: ComplexUNet model
        val_loader: Validation data loader
        device: torch device
        metrics_calc: MetricsCalculator instance
        
    Returns:
        Average validation loss
    """
    model.eval()
    metrics_calc.reset()
    
    total_loss = 0.0
    with torch.no_grad():
        for lr_batch, hr_batch in val_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            # Forward pass
            pred_hr = model(lr_batch)
            
            # Calculate loss
            loss_components = sr_loss(pred_hr, hr_batch)
            total_loss_val, amp_loss, phase_loss = loss_components
            
            # Update metrics
            metrics_calc.update(pred_hr, hr_batch, loss_components)
            total_loss += total_loss_val.item()
    
    return total_loss / len(val_loader)


def log_images_to_tensorboard(writer, lr_batch, hr_batch, pred_batch, epoch, num_images=4):
    """
    Log SAR images to TensorBoard for visualization
    
    Args:
        writer: TensorBoard SummaryWriter
        lr_batch: Low resolution input batch
        hr_batch: High resolution target batch  
        pred_batch: Predicted high resolution batch
        epoch: Current epoch number
        num_images: Number of images to log
    """
    # Convert complex to amplitude for visualization
    lr_amp = torch.abs(lr_batch[:num_images, 0])  # VV channel
    hr_amp = torch.abs(hr_batch[:num_images, 0])  # VV channel
    pred_amp = torch.abs(pred_batch[:num_images, 0])  # VV channel
    
    # Normalize for display (log scale for SAR)
    lr_amp_log = torch.log10(lr_amp + 1e-8)
    hr_amp_log = torch.log10(hr_amp + 1e-8)
    pred_amp_log = torch.log10(pred_amp + 1e-8)
    
    # Normalize to [0, 1] for tensorboard
    lr_norm = (lr_amp_log - lr_amp_log.min()) / (lr_amp_log.max() - lr_amp_log.min() + 1e-8)
    hr_norm = (hr_amp_log - hr_amp_log.min()) / (hr_amp_log.max() - hr_amp_log.min() + 1e-8)
    pred_norm = (pred_amp_log - pred_amp_log.min()) / (pred_amp_log.max() - pred_amp_log.min() + 1e-8)
    
    # Add channel dimension for tensorboard (grayscale)
    lr_norm = lr_norm.unsqueeze(1)
    hr_norm = hr_norm.unsqueeze(1)
    pred_norm = pred_norm.unsqueeze(1)
    
    # Log to tensorboard
    writer.add_images('SAR_Images/LR_Input', lr_norm, epoch)
    writer.add_images('SAR_Images/HR_Target', hr_norm, epoch)
    writer.add_images('SAR_Images/SR_Prediction', pred_norm, epoch)


def log_complex_statistics(writer, complex_tensor, tag, epoch):
    """
    Log complex tensor statistics to TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter
        complex_tensor: Complex tensor to analyze
        tag: Tag for logging (e.g., 'HR_Target', 'SR_Prediction')
        epoch: Current epoch
    """
    # Amplitude and phase statistics
    amplitude = torch.abs(complex_tensor)
    phase = torch.angle(complex_tensor)
    
    # Log amplitude statistics
    writer.add_scalar(f'{tag}/Amplitude_Mean', amplitude.mean().item(), epoch)
    writer.add_scalar(f'{tag}/Amplitude_Std', amplitude.std().item(), epoch)
    writer.add_scalar(f'{tag}/Amplitude_Max', amplitude.max().item(), epoch)
    
    # Log phase statistics (wrapped to [-pi, pi])
    phase_wrapped = torch.atan2(torch.sin(phase), torch.cos(phase))
    writer.add_scalar(f'{tag}/Phase_Mean', phase_wrapped.mean().item(), epoch)
    writer.add_scalar(f'{tag}/Phase_Std', phase_wrapped.std().item(), epoch)
    
    # Log amplitude histogram
    writer.add_histogram(f'{tag}/Amplitude_Distribution', amplitude, epoch)
    writer.add_histogram(f'{tag}/Phase_Distribution', phase_wrapped, epoch)


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
    tensorboard_log_dir: str = None
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
    
    # Log model architecture to TensorBoard
    if writer:
        # Create dummy input for model graph
        dummy_input = torch.randn(1, 2, 128, 64, dtype=torch.complex64).to(device)
        try:
            writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    # Create dataloaders
    print(f"\nLoading data from {data_dir}...")
    train_loader, val_loader = create_dataloaders(data_dir, batch_size)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        print("Training...")
        train_loss = train_epoch(model, train_loader, optimizer, device, train_metrics)
        train_avg_metrics = train_metrics.get_average_metrics()
        
        # Validation
        print("Validating...")
        val_loss = validate_epoch(model, val_loader, device, val_metrics)
        val_avg_metrics = val_metrics.get_average_metrics()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_psnr'].append(train_avg_metrics.get('psnr', 0))
        history['val_psnr'].append(val_avg_metrics.get('psnr', 0))
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('PSNR/Train', train_avg_metrics.get('psnr', 0), epoch)
            writer.add_scalar('PSNR/Val', val_avg_metrics.get('psnr', 0), epoch)
            
            # Log images
            if epoch % 5 == 0 or epoch == num_epochs - 1: # Log every 5 epochs or at the end
                lr_batch_sample, hr_batch_sample = next(iter(val_loader)) # Get a sample batch
                lr_batch_sample = lr_batch_sample.to(device)
                hr_batch_sample = hr_batch_sample.to(device)
                
                with torch.no_grad():
                    pred_hr_sample = model(lr_batch_sample)
                
                log_images_to_tensorboard(writer, lr_batch_sample, hr_batch_sample, pred_hr_sample, epoch)
                log_complex_statistics(writer, hr_batch_sample, 'HR_Target', epoch)
                log_complex_statistics(writer, pred_hr_sample, 'SR_Prediction', epoch)
        
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
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        print("Warning: Model file not found, using random weights")
    
    model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = SARSuperResDataset(data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Test model
    test_metrics = MetricsCalculator()
    test_loss = validate_epoch(model, test_loader, device, test_metrics)
    
    print(f"Test Loss: {test_loss:.6f}")
    test_metrics.print_metrics("Test")


if __name__ == "__main__":
    # Configuration
    config = {
        'data_dir': r"D:\Sentinel-1\data\processed_2",
        'model_save_path': r"D:\Sentinel-1\model\cv_unet.pth",
        'num_epochs': 50,
        'batch_size': 24,  # Increased from 16 to 24 for better GPU utilization
        'learning_rate': 1e-4,
        'early_stop_patience': 7,
        'early_stop_threshold': 1e-4,
        'enable_tensorboard': True,
        'tensorboard_log_dir': None  # Will use timestamped directory
    }
    
    print("Complex U-Net SAR Super-Resolution Training")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    try:
        # Train model
        model, history = train_model(**config)
        
        # Test model
        test_model(config['model_save_path'], config['data_dir'])
        
        print("\nTraining and testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc() 