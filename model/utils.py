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

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # PERF: Limit VGG layers to features[:16] for faster computation
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
    
    def forward(self, recon, gt):
        device = recon.device
        # PERF: Guard .to(device) to prevent repeated transfers
        if self.vgg[0].weight.device != device:
            self.vgg = self.vgg.to(device)
        
        # Perceptual loss operates on magnitude (which is real)
        recon_mag = torch.abs(recon)
        gt_mag = torch.abs(gt)
        
        # Resize to 224x224 for VGG and expand to 3 channels
        recon_mag_resized = F.interpolate(recon_mag.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).repeat(1,3,1,1)
        gt_mag_resized = F.interpolate(gt_mag.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).repeat(1,3,1,1)
        
        # Manual normalization for batch tensors
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        recon_mag_norm = (recon_mag_resized - mean) / std
        gt_mag_norm = (gt_mag_resized - mean) / std
        
        return F.mse_loss(self.vgg(recon_mag_norm), self.vgg(gt_mag_norm))


def sr_loss(recon: torch.Tensor, gt: torch.Tensor, perceptual: nn.Module = None, cpif_weight: float = 0.1):
    """
    Enhanced hybrid loss for complex SAR data. Assumes inputs are complex tensors.
    Combines Amplitude MSE, Phase L1, Perceptual loss, and CPIF loss.

    Args:
        recon (torch.Tensor): Reconstructed complex tensor.
        gt (torch.Tensor): Ground truth complex tensor.
        perceptual (nn.Module, optional): Pre-trained perceptual loss module.
        cpif_weight (float, optional): Weight for the CPIF loss component.

    Returns:
        dict: Dictionary containing total loss and its components.
    """
    # 1. Amplitude Loss (MSE)
    amp_loss = F.mse_loss(torch.abs(recon), torch.abs(gt))
    
    # 2. Phase Loss (L1 on the angle difference)
    phase_diff = torch.angle(recon) - torch.angle(gt)
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = torch.mean(torch.abs(phase_diff))
    
    # Base hybrid loss
    total_loss = 0.6 * amp_loss + 0.4 * phase_loss
    loss_dict = {'amp_loss': amp_loss, 'phase_loss': phase_loss}
    
    # 3. Perceptual Loss (optional)
    if perceptual:
        p_loss = 0.1 * perceptual(recon, gt) # Pass complex tensors, handled inside
        total_loss += p_loss
        loss_dict['perceptual_loss'] = p_loss

    # 4. CPIF Loss (optional)
    # We want to maximize CPIF, so we minimize (1 - normalized_cpif)
    cpif_val = calculate_cpif(recon, gt, reduction='mean')
    # Normalize CPIF to a [0, 1] range roughly, assuming typical values are 0-20dB.
    normalized_cpif = torch.sigmoid((cpif_val - 10) / 5) 
    cpif_loss_val = (1.0 - normalized_cpif) * cpif_weight
    total_loss += cpif_loss_val
    loss_dict['cpif_loss'] = cpif_loss_val
    
    loss_dict['total_loss'] = total_loss
    return loss_dict


def complex_mse_loss(recon, gt):
    return F.mse_loss(recon.real, gt.real) + F.mse_loss(recon.imag, gt.imag)


def complex_l1_loss(recon, gt):
    return F.l1_loss(recon.real, gt.real) + F.l1_loss(recon.imag, gt.imag)


def calculate_rmse(recon: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculate Root Mean Squared Error on the amplitude."""
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    return torch.sqrt(F.mse_loss(recon_amp, gt_amp)).item()


def calculate_psnr(recon: torch.Tensor, gt: torch.Tensor, max_val: Optional[float] = None) -> float:
    """Calculate Peak Signal-to-Noise Ratio for amplitude. Assumes complex inputs."""
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    
    mse = F.mse_loss(recon_amp, gt_amp)
    
    if max_val is None:
        max_val = torch.max(gt_amp)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(recon: torch.Tensor, gt: torch.Tensor, window_size=11, sigma=1.5) -> float:
    """Calculate Structural Similarity Index for amplitude. Assumes complex inputs."""
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
    """
    Calculate Complex Peak Intensity Factor. Assumes complex inputs.
    """
    complex_mse = (recon.real - gt.real)**2 + (recon.imag - gt.imag)**2
    if reduction == 'mean':
        complex_mse = torch.mean(complex_mse)
    
    gt_intensity = torch.abs(gt) ** 2
    peak_intensity = torch.max(gt_intensity)
    
    if torch.all(complex_mse == 0):
        return torch.tensor(float('inf')).to(recon.device)
    
    cpif = 10 * torch.log10(peak_intensity / (complex_mse + 1e-12))
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
            'psnr': [], 'ssim': [], 'cpif': [], 'rmse': [], 'phase_rmse': []
        }
    
    def update(self, recon: torch.Tensor, gt: torch.Tensor, loss_components: Optional[dict] = None):
        """Update metrics with new batch. Assumes complex inputs."""
        with torch.no_grad():
            if loss_components is not None:
                for key, value in loss_components.items():
                    if key not in self.metrics:
                        self.metrics[key] = []
                    self.metrics[key].append(value.item())
            
            # Quality metrics
            self.metrics['psnr'].append(calculate_psnr(recon, gt))
            self.metrics['ssim'].append(calculate_ssim(recon, gt))
            self.metrics['cpif'].append(calculate_cpif(recon, gt).item())
            self.metrics['rmse'].append(calculate_rmse(recon, gt))
            
            # Phase statistics
            phase_stats = calculate_phase_difference_stats(recon, gt)
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
        
        # Quality metrics
        if 'psnr' in avg_metrics:
            print(f"PSNR: {avg_metrics['psnr']:.2f} ± {avg_metrics['psnr_std']:.2f} dB")
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
