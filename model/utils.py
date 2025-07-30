#!/usr/bin/env python3
"""
Loss Functions and Evaluation Metrics for SAR Super-Resolution
==============================================================

Implements specialized loss functions and metrics for complex-valued SAR data:
- Hybrid loss: amplitude MSE + phase L1
- PSNR/SSIM for amplitude
- CPIF (Complex Peak Intensity Factor) for phase quality

Designed for Korean disaster monitoring applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def sr_loss(recon, gt, amp_weight=0.7, phase_weight=0.3):
    """
    SAR Super-Resolution hybrid loss function
    
    Combines amplitude and phase losses for optimal SAR reconstruction:
    loss = amp_weight * MSE(|recon|, |gt|) + phase_weight * L1(∠recon, ∠gt)
    
    Args:
        recon: Reconstructed complex tensor [batch, channels, height, width]
        gt: Ground truth complex tensor [batch, channels, height, width]
        amp_weight: Weight for amplitude loss (default: 0.7)
        phase_weight: Weight for phase loss (default: 0.3)
        
    Returns:
        Combined loss value
    """
    # Calculate amplitude (magnitude)
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    
    # Calculate phase (angle)
    recon_phase = torch.angle(recon)
    gt_phase = torch.angle(gt)
    
    # Amplitude loss: MSE on magnitudes
    amp_loss = F.mse_loss(recon_amp, gt_amp)
    
    # Phase loss: L1 on angles (handles phase wrapping better than MSE)
    phase_diff = recon_phase - gt_phase
    # Handle phase wrapping: map differences to [-π, π]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = F.l1_loss(phase_diff, torch.zeros_like(phase_diff))
    
    # Combined loss
    total_loss = amp_weight * amp_loss + phase_weight * phase_loss
    
    return total_loss, amp_loss, phase_loss


def complex_mse_loss(recon, gt):
    """
    Mean Squared Error for complex tensors
    
    Args:
        recon: Reconstructed complex tensor
        gt: Ground truth complex tensor
        
    Returns:
        MSE loss value
    """
    return F.mse_loss(recon.real, gt.real) + F.mse_loss(recon.imag, gt.imag)


def complex_l1_loss(recon, gt):
    """
    L1 loss for complex tensors
    
    Args:
        recon: Reconstructed complex tensor
        gt: Ground truth complex tensor
        
    Returns:
        L1 loss value
    """
    return F.l1_loss(recon.real, gt.real) + F.l1_loss(recon.imag, gt.imag)


def calculate_psnr(recon, gt, max_val=None):
    """
    Calculate Peak Signal-to-Noise Ratio for amplitude
    
    Args:
        recon: Reconstructed complex tensor
        gt: Ground truth complex tensor
        max_val: Maximum possible value (if None, use gt max)
        
    Returns:
        PSNR value in dB
    """
    # Calculate amplitude
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    
    # Calculate MSE
    mse = F.mse_loss(recon_amp, gt_amp)
    
    if max_val is None:
        max_val = torch.max(gt_amp)
    
    # Avoid log(0)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(recon, gt, window_size=11, sigma=1.5):
    """
    Calculate Structural Similarity Index for amplitude
    
    Simplified SSIM implementation for SAR amplitude data
    
    Args:
        recon: Reconstructed complex tensor [batch, channels, height, width]
        gt: Ground truth complex tensor [batch, channels, height, width]
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian window
        
    Returns:
        SSIM value [0, 1]
    """
    # Calculate amplitude
    recon_amp = torch.abs(recon)
    gt_amp = torch.abs(gt)
    
    # Constants for numerical stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate means
    mu1 = F.avg_pool2d(recon_amp, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(gt_amp, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.avg_pool2d(recon_amp ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(gt_amp ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(recon_amp * gt_amp, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return torch.mean(ssim_map).item()


def calculate_cpif(recon, gt):
    """
    Calculate Complex Peak Intensity Factor
    
    CPIF measures quality of complex reconstruction:
    CPIF = 10 * log10(peak_intensity / MSE(complex))
    
    Higher values indicate better preservation of complex structure
    
    Args:
        recon: Reconstructed complex tensor
        gt: Ground truth complex tensor
        
    Returns:
        CPIF value in dB
    """
    # Calculate complex MSE
    complex_mse = complex_mse_loss(recon, gt)
    
    # Calculate peak intensity from ground truth
    gt_intensity = torch.abs(gt) ** 2
    peak_intensity = torch.max(gt_intensity)
    
    # Avoid log(0)
    if complex_mse == 0:
        return float('inf')
    
    cpif = 10 * torch.log10(peak_intensity / complex_mse)
    return cpif.item()


def calculate_cross_pol_coherence(vv, vh):
    """
    Calculate cross-polarization coherence
    
    coherence = |⟨VV * conj(VH)⟩| / sqrt(⟨|VV|²⟩ * ⟨|VH|²⟩)
    
    Args:
        vv: VV polarization complex tensor
        vh: VH polarization complex tensor
        
    Returns:
        Cross-pol coherence value [0, 1]
    """
    # Calculate cross-correlation
    cross_corr = torch.abs(torch.mean(vv * torch.conj(vh)))
    
    # Calculate power
    vv_power = torch.mean(torch.abs(vv) ** 2)
    vh_power = torch.mean(torch.abs(vh) ** 2)
    
    # Calculate coherence
    if vv_power > 1e-10 and vh_power > 1e-10:
        coherence = cross_corr / torch.sqrt(vv_power * vh_power)
        return coherence.item()
    else:
        return 0.0


def calculate_phase_difference_stats(recon, gt):
    """
    Calculate phase difference statistics
    
    Args:
        recon: Reconstructed complex tensor
        gt: Ground truth complex tensor
        
    Returns:
        Dictionary with phase difference statistics
    """
    # Calculate phase difference
    recon_phase = torch.angle(recon)
    gt_phase = torch.angle(gt)
    
    phase_diff = recon_phase - gt_phase
    # Handle phase wrapping
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    # Calculate statistics
    stats = {
        'mean_phase_error': torch.mean(torch.abs(phase_diff)).item(),
        'std_phase_error': torch.std(phase_diff).item(),
        'max_phase_error': torch.max(torch.abs(phase_diff)).item(),
        'phase_rmse': torch.sqrt(torch.mean(phase_diff ** 2)).item()
    }
    
    return stats


class MetricsCalculator:
    """
    Comprehensive metrics calculator for SAR super-resolution
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.metrics = {
            'loss': [],
            'amp_loss': [],
            'phase_loss': [],
            'psnr': [],
            'ssim': [],
            'cpif': [],
            'cross_pol_coherence': [],
            'phase_rmse': []
        }
    
    def update(self, recon, gt, loss_components=None):
        """
        Update metrics with new batch
        
        Args:
            recon: Reconstructed complex tensor
            gt: Ground truth complex tensor
            loss_components: Tuple of (total_loss, amp_loss, phase_loss)
        """
        with torch.no_grad():
            # Loss components
            if loss_components is not None:
                total_loss, amp_loss, phase_loss = loss_components
                self.metrics['loss'].append(total_loss.item())
                self.metrics['amp_loss'].append(amp_loss.item())
                self.metrics['phase_loss'].append(phase_loss.item())
            
            # Quality metrics
            psnr = calculate_psnr(recon, gt)
            ssim = calculate_ssim(recon, gt)
            cpif = calculate_cpif(recon, gt)
            
            self.metrics['psnr'].append(psnr)
            self.metrics['ssim'].append(ssim)
            self.metrics['cpif'].append(cpif)
            
            # Dual-pol specific metrics (if 2 channels)
            if recon.shape[1] == 2:
                vv_recon, vh_recon = recon[:, 0], recon[:, 1]
                vv_gt, vh_gt = gt[:, 0], gt[:, 1]
                
                coherence = calculate_cross_pol_coherence(vv_recon, vh_recon)
                self.metrics['cross_pol_coherence'].append(coherence)
            
            # Phase statistics
            phase_stats = calculate_phase_difference_stats(recon, gt)
            self.metrics['phase_rmse'].append(phase_stats['phase_rmse'])
    
    def get_average_metrics(self):
        """
        Get average of accumulated metrics
        
        Returns:
            Dictionary of average metrics
        """
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
        
        # Loss metrics
        if 'loss' in avg_metrics:
            print(f"Loss: {avg_metrics['loss']:.6f} ± {avg_metrics['loss_std']:.6f}")
            print(f"  Amplitude: {avg_metrics['amp_loss']:.6f} ± {avg_metrics['amp_loss_std']:.6f}")
            print(f"  Phase: {avg_metrics['phase_loss']:.6f} ± {avg_metrics['phase_loss_std']:.6f}")
        
        # Quality metrics
        if 'psnr' in avg_metrics:
            print(f"PSNR: {avg_metrics['psnr']:.2f} ± {avg_metrics['psnr_std']:.2f} dB")
        if 'ssim' in avg_metrics:
            print(f"SSIM: {avg_metrics['ssim']:.4f} ± {avg_metrics['ssim_std']:.4f}")
        if 'cpif' in avg_metrics:
            print(f"CPIF: {avg_metrics['cpif']:.2f} ± {avg_metrics['cpif_std']:.2f} dB")
        
        # SAR-specific metrics
        if 'cross_pol_coherence' in avg_metrics:
            print(f"Cross-pol Coherence: {avg_metrics['cross_pol_coherence']:.4f} ± {avg_metrics['cross_pol_coherence_std']:.4f}")
        if 'phase_rmse' in avg_metrics:
            print(f"Phase RMSE: {avg_metrics['phase_rmse']:.4f} ± {avg_metrics['phase_rmse_std']:.4f} rad")


if __name__ == "__main__":
    # Test loss functions and metrics
    print("Testing SAR loss functions and metrics...")
    
    # Create test data
    batch_size, channels, height, width = 2, 2, 256, 512
    gt_real = torch.randn(batch_size, channels, height, width)
    gt_imag = torch.randn(batch_size, channels, height, width)
    gt = torch.complex(gt_real, gt_imag)
    
    # Simulate reconstructed data with some noise
    noise_real = 0.1 * torch.randn_like(gt_real)
    noise_imag = 0.1 * torch.randn_like(gt_imag)
    recon = torch.complex(gt_real + noise_real, gt_imag + noise_imag)
    
    # Test loss function
    total_loss, amp_loss, phase_loss = sr_loss(recon, gt)
    print(f"Total Loss: {total_loss:.6f}")
    print(f"Amplitude Loss: {amp_loss:.6f}")
    print(f"Phase Loss: {phase_loss:.6f}")
    
    # Test metrics
    psnr = calculate_psnr(recon, gt)
    ssim = calculate_ssim(recon, gt)
    cpif = calculate_cpif(recon, gt)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"CPIF: {cpif:.2f} dB")
    
    # Test dual-pol metrics
    if channels == 2:
        vv_recon, vh_recon = recon[:, 0], recon[:, 1]
        coherence = calculate_cross_pol_coherence(vv_recon, vh_recon)
        print(f"Cross-pol Coherence: {coherence:.4f}")
    
    # Test metrics calculator
    calculator = MetricsCalculator()
    calculator.update(recon, gt, (total_loss, amp_loss, phase_loss))
    calculator.print_metrics("Test")
    
    print("\nLoss functions and metrics test completed successfully!") 