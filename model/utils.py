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
    
    def forward(self, recon_single_pol, gt_single_pol):
        # Input shape: (B, 2, H, W) -> [Real, Imag]
        recon_complex = torch.complex(recon_single_pol[:, 0], recon_single_pol[:, 1])
        gt_complex = torch.complex(gt_single_pol[:, 0], gt_single_pol[:, 1])
        
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


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def _tv_l1(x: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dx + dy


def sr_loss(recon, gt, perceptual=None, *, perceptual_weight: float = 0.0, fft_weight: float = 0.0):
    """
    Phase-preserving SR loss set for dual-pol SAR (VV, VH):
    - Log-amplitude Charbonnier (multiplicative speckle friendly)
    - Circular phase loss: 1 - cos(Δφ)
    - Complex L1 on (Real, Imag)
    - FFT(Amplitude) spectrum L1 to suppress grid frequencies
    - Small TV on amplitude
    - Optional perceptual (amplitude-only, small weight)
    
    Args:
        recon: (B, 4, H, W) real tensor [VV-Re, VV-Im, VH-Re, VH-Im]
        gt:    (B, 4, H, W) real tensor
        perceptual: Optional PerceptualLoss (amplitude-only)
    Returns:
        Dict of loss components
    """
    # Weights (phase preservation first)
    w_log_amp = 0.45
    w_phase = 0.40
    w_cplx_l1 = 0.10
    w_tv = 1e-3

    # Convert to complex
    recon_vv = torch.complex(recon[:, 0], recon[:, 1])
    recon_vh = torch.complex(recon[:, 2], recon[:, 3])
    gt_vv = torch.complex(gt[:, 0], gt[:, 1])
    gt_vh = torch.complex(gt[:, 2], gt[:, 3])

    # Amplitudes and phases
    amp_r_vv = torch.abs(recon_vv)
    amp_r_vh = torch.abs(recon_vh)
    amp_g_vv = torch.abs(gt_vv)
    amp_g_vh = torch.abs(gt_vh)

    phase_r_vv = torch.angle(recon_vv)
    phase_r_vh = torch.angle(recon_vh)
    phase_g_vv = torch.angle(gt_vv)
    phase_g_vh = torch.angle(gt_vh)

    # 1) Log-amplitude Charbonnier
    log_r_vv = torch.log(amp_r_vv + 1e-8)
    log_g_vv = torch.log(amp_g_vv + 1e-8)
    log_r_vh = torch.log(amp_r_vh + 1e-8)
    log_g_vh = torch.log(amp_g_vh + 1e-8)
    log_amp_charb_vv = _charbonnier(log_r_vv - log_g_vv).mean()
    log_amp_charb_vh = _charbonnier(log_r_vh - log_g_vh).mean()
    log_amp_loss = w_log_amp * (log_amp_charb_vv + log_amp_charb_vh)

    # 2) Circular phase loss: 1 - cos(Δφ)
    dphi_vv = phase_r_vv - phase_g_vv
    dphi_vh = phase_r_vh - phase_g_vh
    phase_loss_vv = (1 - torch.cos(dphi_vv)).mean()
    phase_loss_vh = (1 - torch.cos(dphi_vh)).mean()
    phase_loss = w_phase * (phase_loss_vv + phase_loss_vh)

    # 3) Complex L1 on (real, imag)
    cplx_l1_vv = F.l1_loss(recon_vv.real, gt_vv.real) + F.l1_loss(recon_vv.imag, gt_vv.imag)
    cplx_l1_vh = F.l1_loss(recon_vh.real, gt_vh.real) + F.l1_loss(recon_vh.imag, gt_vh.imag)
    cplx_l1 = w_cplx_l1 * (cplx_l1_vv + cplx_l1_vh)

    # 4) FFT (amplitude) spectrum L1 with radial high-frequency weighting
    def amp_fft_l1_weighted(a_pred: torch.Tensor, a_gt: torch.Tensor) -> torch.Tensor:
        Fp = torch.fft.rfft2(a_pred.float())
        Fg = torch.fft.rfft2(a_gt.float())
        diff = torch.abs(torch.abs(Fp) - torch.abs(Fg))  # (B, H, W//2+1)
        H, W_r = diff.shape[-2], diff.shape[-1]
        fy = torch.fft.fftfreq(H, d=1.0, device=diff.device).abs().view(1, H, 1)
        fx = torch.fft.rfftfreq((W_r - 1) * 2, d=1.0, device=diff.device).view(1, 1, W_r)
        r = torch.sqrt(fy * fy + fx * fx)
        r = r / (r.max() + 1e-8)
        return (diff * r).mean()
    fft_loss = fft_weight * (amp_fft_l1_weighted(amp_r_vv, amp_g_vv) + amp_fft_l1_weighted(amp_r_vh, amp_g_vh))

    # 5) Tiny TV on amplitude
    amp_stack = torch.stack([amp_r_vv, amp_r_vh], dim=1)  # (B,2,H,W)
    tv_loss = w_tv * _tv_l1(amp_stack)

    # 6) Perceptual (amplitude-only) - weighted by flag
    p_loss = torch.tensor(0.0, device=recon.device)
    if perceptual is not None and perceptual_weight > 0.0:
        p_vv = perceptual(recon[:, :2], gt[:, :2])
        p_vh = perceptual(recon[:, 2:], gt[:, 2:])
        p_loss = perceptual_weight * (p_vv + p_vh)

    total_loss = log_amp_loss + phase_loss + cplx_l1 + fft_loss + tv_loss + p_loss

    return {
        'total_loss': total_loss,
        'log_amp_loss': log_amp_loss,
        'phase_loss': phase_loss,
        'phase_loss_vv': w_phase * phase_loss_vv,
        'phase_loss_vh': w_phase * phase_loss_vh,
        'complex_l1': cplx_l1,
        'fft_loss': fft_loss,
        'tv_loss': tv_loss,
        'perceptual_loss': p_loss,
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


def calculate_psnr_dual_pol(recon: torch.Tensor, gt: torch.Tensor) -> dict:
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
    
    # Use max value from ground truth for PSNR calculation
    max_val_vv = gt_vv_amp.max()
    max_val_vh = gt_vh_amp.max()
    
    # Calculate PSNR for each polarization
    mse_vv = F.mse_loss(recon_vv_amp, gt_vv_amp)
    mse_vh = F.mse_loss(recon_vh_amp, gt_vh_amp)
    
    # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon) 추가
    epsilon = 1e-9
    
    psnr_vv = 20 * torch.log10(max_val_vv / torch.sqrt(mse_vv + epsilon))
    psnr_vh = 20 * torch.log10(max_val_vh / torch.sqrt(mse_vh + epsilon))
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
