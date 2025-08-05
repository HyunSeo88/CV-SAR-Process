#!/usr/bin/env python3
"""
Loss Functions and Evaluation Metrics for SAR Super-Resolution
==============================================================

Implements specialized loss functions and metrics for complex-valued SAR data:
- Hybrid loss: amplitude MSE + phase L1 + CPIF Loss
- PSNR/SSIM/RMSE for amplitude
- CPIF (Complex Peak Intensity Factor) for phase quality

Designed for Korean disaster monitoring applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.models as models
from typing import Optional

def _ensure_complex(t: torch.Tensor) -> torch.Tensor:
    """Convert 4-channel real tensor → 2-channel complex tensor if needed."""
    if torch.is_complex(t):
        return t
    if t.shape[1] == 4:
        vv = torch.complex(t[:,0], t[:,1])
        vh = torch.complex(t[:,2], t[:,3])
        return torch.stack([vv, vh], 1)
    # Remove 2-channel support as it creates artificial VH channel
    # Single-polarization experiments should use dedicated functions
    raise ValueError(f"Expected tensor with 4 channels for dual-pol complex conversion, but got {t.shape[1]} channels. Tensor shape: {t.shape}. For single-pol data, use dedicated single-channel loss functions.")


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # PERF: Limit VGG layers to features[:16] for faster computation
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
    
    def forward(self, recon, gt):
        # Convert real-valued 4-channel tensor to complex for magnitude calculation
        recon_complex = torch.complex(recon[:, 0], recon[:, 1])
        gt_complex = torch.complex(gt[:, 0], gt[:, 1])
        
        # Calculate magnitude and ensure it's 4D: (B, 1, H, W)
        recon_mag = torch.abs(recon_complex).unsqueeze(1)
        gt_mag = torch.abs(gt_complex).unsqueeze(1)
        
        # Resize for VGG input and repeat for 3 channels
        recon_mag_resized = F.interpolate(recon_mag, size=(224, 224), mode='bilinear', align_corners=False).repeat(1,3,1,1)
        gt_mag_resized = F.interpolate(gt_mag, size=(224, 224), mode='bilinear', align_corners=False).repeat(1,3,1,1)
        
        # Extract features
        recon_features = self.vgg(recon_mag_resized)
        gt_features = self.vgg(gt_mag_resized)
        
        # Compute perceptual loss as MSE between feature maps
        return F.mse_loss(recon_features, gt_features)


def sr_loss(recon, gt, perceptual=None):
    """
    Computes SR loss: combination of amplitude, phase, and optional perceptual loss for both VV and VH polarizations.
    
    Args:
        recon: Reconstructed 4-channel real tensor (B, 4, H, W) - [VV-Re, VV-Im, VH-Re, VH-Im]
        gt: Ground truth 4-channel real tensor (B, 4, H, W) - [VV-Re, VV-Im, VH-Re, VH-Im]
        perceptual: PerceptualLoss instance
        
    Returns:
        Dictionary of loss components
    """
    # loss weights
    amp_weight = 0.5
    phase_weight = 0.3
    perceptual_weight = 0.05
    # Convert to complex for both VV and VH polarizations
    recon_vv = torch.complex(recon[:, 0], recon[:, 1])
    recon_vh = torch.complex(recon[:, 2], recon[:, 3])
    gt_vv = torch.complex(gt[:, 0], gt[:, 1])
    gt_vh = torch.complex(gt[:, 2], gt[:, 3])
    
    # 1. Amplitude Loss (L1) for both polarizations
    amp_loss_vv = F.l1_loss(torch.abs(recon_vv), torch.abs(gt_vv))
    amp_loss_vh = F.l1_loss(torch.abs(recon_vh), torch.abs(gt_vh))
    amp_loss = amp_weight * (amp_loss_vv + amp_loss_vh)
    
    # 2. Phase Loss (L1) for both polarizations
    phase_loss_vv = F.l1_loss(torch.angle(recon_vv), torch.angle(gt_vv))
    phase_loss_vh = F.l1_loss(torch.angle(recon_vh), torch.angle(gt_vh))
    phase_loss = phase_weight * (phase_loss_vv + phase_loss_vh)
    
    # 3. Perceptual Loss (optional) - uses VV polarization for compatibility with VGG
    p_loss = 0
    if perceptual is not None:
        # Pass the ORIGINAL 4-channel real tensors to perceptual loss
        p_loss = perceptual_weight * perceptual(recon, gt)
    
    # Total loss
    total_loss = amp_loss + phase_loss + p_loss
    
    return {
        'total_loss': total_loss,
        'amp_loss': amp_loss,
        'amp_loss_vv': amp_loss_vv,
        'amp_loss_vh': amp_loss_vh,
        'phase_loss': phase_loss,
        'phase_loss_vv': phase_loss_vv,
        'phase_loss_vh': phase_loss_vh,
        'perceptual_loss': p_loss
    }


def complex_mse_loss(recon, gt):
    return F.mse_loss(recon.real, gt.real) + F.mse_loss(recon.imag, gt.imag)


def complex_l1_loss(recon, gt):
    return F.l1_loss(recon.real, gt.real) + F.l1_loss(recon.imag, gt.imag)


def calculate_rmse(recon: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculate Root Mean Squared Error on the amplitude."""
    recon, gt = _ensure_complex(recon), _ensure_complex(gt)
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    return torch.sqrt(F.mse_loss(recon_amp, gt_amp)).item()


def calculate_psnr(recon: torch.Tensor, gt: torch.Tensor, max_val: Optional[float] = None) -> float:
    """Calculate PSNR for VV polarization (backward compatibility)."""
    recon, gt = _ensure_complex(recon), _ensure_complex(gt)
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    
    mse = F.mse_loss(recon_amp, gt_amp)
    
    # Use fixed max_val for consistent PSNR calculation across batches
    if max_val is None:
        max_val = 1.0  # Fixed reference value for SAR amplitude data
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def calculate_psnr_dual_pol(recon: torch.Tensor, gt: torch.Tensor, max_val: Optional[float] = None) -> dict:
    """Calculate PSNR for both VV and VH polarizations separately."""
    # Convert 4-channel real to complex
    recon_vv = torch.complex(recon[:, 0], recon[:, 1])
    recon_vh = torch.complex(recon[:, 2], recon[:, 3])
    gt_vv = torch.complex(gt[:, 0], gt[:, 1])
    gt_vh = torch.complex(gt[:, 2], gt[:, 3])
    
    recon_vv_amp = torch.abs(recon_vv)
    recon_vh_amp = torch.abs(recon_vh)
    gt_vv_amp = torch.abs(gt_vv)
    gt_vh_amp = torch.abs(gt_vh)
    
    # Use fixed max_val for consistent PSNR calculation
    if max_val is None:
        max_val = 1.0
    
    # Calculate PSNR for each polarization
    mse_vv = F.mse_loss(recon_vv_amp, gt_vv_amp)
    mse_vh = F.mse_loss(recon_vh_amp, gt_vh_amp)
    
    psnr_vv = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse_vv + 1e-8))
    psnr_vh = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse_vh + 1e-8))
    psnr_avg = (psnr_vv + psnr_vh) / 2
    
    return {
        'psnr_vv': psnr_vv.item(),
        'psnr_vh': psnr_vh.item(),
        'psnr_avg': psnr_avg.item()
    }


def calculate_ssim(recon: torch.Tensor, gt: torch.Tensor, window_size=11, sigma=1.5) -> float:
    """Calculate Structural Similarity Index for amplitude. Assumes complex inputs."""
    recon, gt = _ensure_complex(recon), _ensure_complex(gt)
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(recon_amp, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(gt_amp, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(recon_amp ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(gt_amp ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(recon_amp * gt_amp, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return torch.mean(ssim_map).item()


def calculate_cpif(recon: torch.Tensor, gt: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    recon, gt = _ensure_complex(recon), _ensure_complex(gt)
    """
    Calculate Complex Peak Intensity Factor. Assumes complex inputs.
    Uses proper pixel-wise MSE calculation for accurate dB scaling.
    """
    # Calculate pixel-wise complex MSE
    complex_mse = (recon.real - gt.real)**2 + (recon.imag - gt.imag)**2
    
    # Calculate mean MSE first, then apply logarithmic scaling
    if reduction == 'mean':
        mse_mean = torch.mean(complex_mse)
    elif reduction == 'none':
        mse_mean = complex_mse  # Keep per-pixel values
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    
    gt_intensity = torch.abs(gt) ** 2
    peak_intensity = torch.max(gt_intensity)
    
    if torch.all(mse_mean == 0):
        return torch.tensor(float('inf')).to(recon.device)
    
    # Apply logarithmic scaling to the averaged MSE for correct dB calculation
    cpif = 10 * torch.log10(peak_intensity / (mse_mean + 1e-12))
    
    return cpif


def calculate_phase_difference_stats(recon: torch.Tensor, gt: torch.Tensor) -> dict:
    """Calculate phase difference statistics. Assumes complex inputs."""
    recon_phase = torch.angle(recon)
    gt_phase = torch.angle(gt)
    
    phase_diff = recon_phase - gt_phase
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    stats = {
        'mean_phase_error': torch.mean(torch.abs(phase_diff)).item(),
        'std_phase_error': torch.std(phase_diff).item(),
        'max_phase_error': torch.max(torch.abs(phase_diff)).item(),
        'phase_rmse': torch.sqrt(torch.mean(phase_diff ** 2)).item()
    }
    return stats


class MetricsCalculator:
    """Comprehensive metrics calculator for SAR super-resolution."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.metrics = {
            'loss': [], 'amp_loss': [], 'phase_loss': [], 'cpif_loss': [],
            'psnr': [], 'psnr_vv': [], 'psnr_vh': [], 'psnr_avg': [],
            'ssim': [], 'cpif': [], 'rmse': [], 'phase_rmse': []
        }
    
    def update(self, recon: torch.Tensor, gt: torch.Tensor, loss_components: Optional[dict] = None):
        """Update metrics with new batch. Assumes 4-channel real inputs for dual-pol."""
        with torch.no_grad():
            if loss_components is not None:
                for key, value in loss_components.items():
                    if key not in self.metrics:
                        self.metrics[key] = []
                    self.metrics[key].append(value.item())
            
            # Quality metrics - dual polarization
            psnr_results = calculate_psnr_dual_pol(recon, gt)
            self.metrics['psnr_vv'].append(psnr_results['psnr_vv'])
            self.metrics['psnr_vh'].append(psnr_results['psnr_vh'])
            self.metrics['psnr_avg'].append(psnr_results['psnr_avg'])
            
            # Backward compatibility - use VV PSNR for 'psnr' key
            self.metrics['psnr'].append(psnr_results['psnr_vv'])
            
            # Convert to complex for other metrics (using _ensure_complex)
            recon_complex = _ensure_complex(recon)
            gt_complex = _ensure_complex(gt)
            
            self.metrics['ssim'].append(calculate_ssim(recon_complex, gt_complex))
            self.metrics['cpif'].append(calculate_cpif(recon_complex, gt_complex).item())
            self.metrics['rmse'].append(calculate_rmse(recon, gt))
            
            # Phase statistics
            phase_stats = calculate_phase_difference_stats(recon_complex, gt_complex)
            self.metrics['phase_rmse'].append(phase_stats['phase_rmse'])
    
    def get_average_metrics(self):
        """Get average of accumulated metrics."""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        return avg_metrics
    
    def print_metrics(self, prefix=""):
        """Print current average metrics"""
        avg_metrics = self.get_average_metrics()
        
        print(f"\n{prefix} Metrics:")
        print("-" * 50)
        
        if 'loss' in avg_metrics:
            print(f"Loss: {avg_metrics['loss']:.6f} ± {avg_metrics['loss_std']:.6f}")
        
        # Quality metrics - dual polarization
        if 'psnr_vv' in avg_metrics:
            print(f"PSNR VV: {avg_metrics['psnr_vv']:.2f} ± {avg_metrics['psnr_vv_std']:.2f} dB")
        if 'psnr_vh' in avg_metrics:
            print(f"PSNR VH: {avg_metrics['psnr_vh']:.2f} ± {avg_metrics['psnr_vh_std']:.2f} dB")
        if 'psnr_avg' in avg_metrics:
            print(f"PSNR Avg: {avg_metrics['psnr_avg']:.2f} ± {avg_metrics['psnr_avg_std']:.2f} dB")
        
        if 'ssim' in avg_metrics:
            print(f"SSIM: {avg_metrics['ssim']:.4f} ± {avg_metrics['ssim_std']:.4f}")
        if 'cpif' in avg_metrics:
            print(f"CPIF: {avg_metrics['cpif']:.2f} ± {avg_metrics['cpif_std']:.2f} dB")
        if 'rmse' in avg_metrics:
            print(f"RMSE: {avg_metrics['rmse']:.4f} ± {avg_metrics['rmse_std']:.4f}")
        
        # SAR-specific metrics
        if 'phase_rmse' in avg_metrics:
            print(f"Phase RMSE: {avg_metrics['phase_rmse']:.4f} ± {avg_metrics['phase_rmse_std']:.4f} rad")


if __name__ == "__main__":
    print("Testing SAR loss functions and metrics...")
    
    batch_size, channels, height, width = 2, 2, 256, 512
    gt = torch.randn(batch_size, channels, height, width, dtype=torch.cfloat)
    recon = gt + 0.1 * torch.randn_like(gt)
    
    perceptual = PerceptualLoss()
    loss_dict = sr_loss(recon, gt, perceptual)
    
    print("\n--- Loss Components ---")
    for name, val in loss_dict.items():
        print(f"{name.replace('_', ' ').title()}: {val.item():.6f}")

    print("\n--- Evaluation Metrics ---")
    calculator = MetricsCalculator()
    calculator.update(recon, gt, loss_dict)
    calculator.print_metrics("Test Batch")
    
    print("\nLoss functions and metrics test completed successfully!")
