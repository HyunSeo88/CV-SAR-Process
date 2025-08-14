#!/usr/bin/env python3
"""
Loss Functions and Evaluation Metrics for SAR Super-Resolution
==============================================================

This module implements physically consistent objectives aligned with
연구질문.md (A-계열 제약):
- [§4.1] Data Consistency via forward operator H and decimator D (implemented in degradations.py)
- [§4.2] Spectral band constraint using frequency support mask M
- [§4.3] Phase-safe metrics including interferometric coherence |γ|

We also provide amplitude (log-magnitude) and phase (circular) losses and
evaluation metrics on log-intensity, as required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.models as models
from typing import Optional, Tuple

"""
Fixed PSNR reference values for VV and VH amplitude when --psnr-ref=fixed.
Tune if a different fixed scaling is desired; defaults keep backward behavior.
"""
PSNR_REF_VV: float = 1.0
PSNR_REF_VH: float = 1.0

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


# -----------------------------------------------------------------------------
# Helpers for physically consistent losses (연구질문.md mapping)
# -----------------------------------------------------------------------------

def _global_phase_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Align global phase of pred to gt per-sample and per-channel by
    Δφ = angle(sum pred * conj(gt)).
    [연구질문.md §2, objective aligns statistically; here we remove global offset
    for circular phase MAE computation.]
    """
    # pred, gt: complex tensors of shape [B,H,W] or [B,2,H,W]
    if pred.dim() == 4:  # [B, H, W] or [B, C, H, W]?
        # Interpret as [B, H, W]; do nothing
        pass
    ph = pred
    gh = gt
    # Compute Δφ per sample (and channel if present)
    # Support shapes [B, H, W] and [B, C, H, W]
    if ph.dim() == 3:
        num = (ph * gh.conj()).sum(dim=(-2, -1), keepdim=True)
        dphi = torch.angle(num)
        return ph * torch.exp(-1j * dphi)
    elif ph.dim() == 4:
        num = (ph * gh.conj()).sum(dim=(-2, -1), keepdim=True)  # [B,C,1,1]
        dphi = torch.angle(num)
        return ph * torch.exp(-1j * dphi)
    else:
        return pred


_FREQ_MASK_CACHE: dict = {}

def _get_frequency_mask(H: int, W: int, cutoff_rng: float, cutoff_az: float, device: torch.device) -> torch.Tensor:
    """
    Cached elliptical passband mask M on target device. [연구질문.md §4.2]
    """
    key = (int(H), int(W), float(round(cutoff_rng, 4)), float(round(cutoff_az, 4)), str(device))
    if key in _FREQ_MASK_CACHE:
        return _FREQ_MASK_CACHE[key]
    fy = torch.fft.fftfreq(H, d=1.0, device=device)
    fx = torch.fft.fftfreq(W, d=1.0, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    nyq = 0.5
    nr = (FY.abs() / nyq) / max(1e-8, float(cutoff_rng))
    na = (FX.abs() / nyq) / max(1e-8, float(cutoff_az))
    mask = (nr**2 + na**2 <= 1.0).to(torch.float32)
    _FREQ_MASK_CACHE[key] = mask
    return mask


def sr_loss(
    recon: torch.Tensor,
    gt: torch.Tensor,
    perceptual=None,
    *,
    # New physical weights (연구질문.md)
    w_mag: float = 1.0,
    w_phase: float = 1.0,
    w_coh: float = 0.0,
    w_spec: float = 0.0,
    # Data-consistency (optional, §4.1)
    w_dc: float = 0.0,
    lr_input: Optional[torch.Tensor] = None,
    dc_scale: Optional[int] = None,
    dc_lp_params: Optional[object] = None,
    # Spectral mask params (§4.2)
    spec_cutoff_rng: float = 0.45,
    spec_cutoff_az: float = 0.45,
    # Legacy/optional extras
    perceptual_weight: float = 0.0,
    fft_weight: float = 0.0,
    phase_window: int = 7,
    eps: float = 1e-8,
):
    """
    Physically motivated SR loss set (VV, VH), aligned with 연구질문.md:
    - Magnitude loss: L1(log|S_SR|, log|S_HR|) [§C Magnitude loss]
    - Phase loss: circular L1 after global phase alignment [§C Phase loss]
    - Coherence loss: 1 - |γ| (local coherence) [§C Coherence]
    - Spectral band loss: || (1-M) ⊙ F{u_hat} ||_2^2 [§4.2]
    - Optional Data Consistency: || Down_H(u_hat) - y ||_1 [§4.1]
    """
    device = recon.device
    # Convert 4-real to complex per-pol
    recon_vv = torch.complex(recon[:, 0], recon[:, 1])
    recon_vh = torch.complex(recon[:, 2], recon[:, 3])
    gt_vv = torch.complex(gt[:, 0], gt[:, 1])
    gt_vh = torch.complex(gt[:, 2], gt[:, 3])

    # 1) Magnitude loss on log-amplitude
    def log_mag_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(torch.log(a + eps), torch.log(b + eps))
    amp_r_vv, amp_r_vh = torch.abs(recon_vv), torch.abs(recon_vh)
    amp_g_vv, amp_g_vh = torch.abs(gt_vv), torch.abs(gt_vh)
    mag_loss = w_mag * (log_mag_l1(amp_r_vv, amp_g_vv) + log_mag_l1(amp_r_vh, amp_g_vh))

    # 2) Phase loss (circular MAE) after global alignment
    recon_vv_al = _global_phase_align(recon_vv, gt_vv)
    recon_vh_al = _global_phase_align(recon_vh, gt_vh)
    dphi_vv = torch.atan2(torch.sin(torch.angle(recon_vv_al) - torch.angle(gt_vv)),
                          torch.cos(torch.angle(recon_vv_al) - torch.angle(gt_vv)))
    dphi_vh = torch.atan2(torch.sin(torch.angle(recon_vh_al) - torch.angle(gt_vh)),
                          torch.cos(torch.angle(recon_vh_al) - torch.angle(gt_vh)))
    phase_loss_vv = torch.mean(torch.abs(dphi_vv))
    phase_loss_vh = torch.mean(torch.abs(dphi_vh))
    phase_loss = w_phase * (phase_loss_vv + phase_loss_vh)

    # 3) Interferometric coherence regularizer L_coh = 1 - |γ|
    def local_coherence(c1: torch.Tensor, c2: torch.Tensor, win: int) -> torch.Tensor:
        pad = win // 2
        def ap(x: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(x, kernel_size=win, stride=1, padding=pad)
        r1, i1 = c1.real, c1.imag
        r2, i2 = c2.real, c2.imag
        real_num = ap(r1 * r2 + i1 * i2)
        imag_num = ap(i1 * r2 - r1 * i2)
        num = torch.sqrt(real_num**2 + imag_num**2 + 1e-16)
        den = torch.sqrt(ap(r1*r1 + i1*i1) * ap(r2*r2 + i2*i2) + 1e-16)
        gamma = (num / (den + 1e-12)).clamp(0.0, 1.0)
        return gamma
    gamma_vv = local_coherence(recon_vv, gt_vv, phase_window)
    gamma_vh = local_coherence(recon_vh, gt_vh, phase_window)
    coh_loss = w_coh * (1.0 - gamma_vv.mean() + 1.0 - gamma_vh.mean())

    # 4) Spectral band loss (mask outside support)
    H, W = recon_vv.shape[-2], recon_vv.shape[-1]
    M = _get_frequency_mask(H, W, spec_cutoff_rng, spec_cutoff_az, device)
    def spec_band_loss(c: torch.Tensor) -> torch.Tensor:
        U = torch.fft.fft2(c)
        out = (1.0 - M) * torch.abs(U)
        return torch.sum(out * out) / (H * W)
    spec_loss = w_spec * (spec_band_loss(recon_vv) + spec_band_loss(recon_vh))

    # 5) Optional Data Consistency with LR input and forward model (vectorized)
    dc_loss = torch.tensor(0.0, device=device)
    if w_dc > 0.0 and lr_input is not None and dc_scale is not None and dc_lp_params is not None:
        import degradations as _deg
        B = recon.shape[0]
        recon_c = torch.stack([recon_vv, recon_vh], dim=1)  # [B,2,H,W]
        kernel = _deg.make_psf(dc_lp_params.kind, dc_lp_params.sigma_px, dc_lp_params.size, dc_lp_params.beta, device=device, cutoff=dc_lp_params.cutoff)
        pad_h = kernel.shape[-2] // 2
        pad_w = kernel.shape[-1] // 2
        # Depthwise over 2 channels (VV,VH) with the same PSF per channel
        weight_dw = kernel.repeat(2, 1, 1, 1)  # [2,1,kH,kW]
        real = recon_c.real  # [B,2,H,W]
        imag = recon_c.imag  # [B,2,H,W]
        real = F.pad(real, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        imag = F.pad(imag, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        real = F.conv2d(real, weight_dw, stride=1, padding=0, groups=2)
        imag = F.conv2d(imag, weight_dw, stride=1, padding=0, groups=2)
        c_lp = torch.complex(real.float(), imag.float())  # [B,2,H,W]
        y_hat = c_lp[:, :, ::dc_scale, ::dc_scale]
        y_hat_real4 = torch.stack([y_hat[:, 0].real, y_hat[:, 0].imag, y_hat[:, 1].real, y_hat[:, 1].imag], dim=1)
        y_tgt = lr_input
        if y_hat_real4.shape[-2:] != y_tgt.shape[-2:]:
            h = min(y_hat_real4.shape[-2], y_tgt.shape[-2])
            w = min(y_hat_real4.shape[-1], y_tgt.shape[-1])
            y_hat_real4 = y_hat_real4[..., :h, :w]
            y_tgt = y_tgt[..., :h, :w]
        dc_loss = w_dc * F.l1_loss(y_hat_real4, y_tgt)

    # Optional extras for stability/legacy
    fft_loss = torch.tensor(0.0, device=device)
    phase_fft_loss = torch.tensor(0.0, device=device)
    tv_loss = torch.tensor(0.0, device=device)
    p_loss = torch.tensor(0.0, device=device)
    if fft_weight > 0.0:
        def complex_fft_l1_weighted(c_pred: torch.Tensor, c_gt: torch.Tensor) -> torch.Tensor:
            Fp = torch.fft.fft2(c_pred)
            Fg = torch.fft.fft2(c_gt)
            diff = torch.abs(Fp - Fg)
            fy = torch.fft.fftfreq(H, d=1.0, device=diff.device).abs().view(1, H, 1)
            fx = torch.fft.fftfreq(W, d=1.0, device=diff.device).abs().view(1, 1, W)
            r = torch.sqrt(fy * fy + fx * fx)
            r = r / (r.max() + 1e-8)
            return (diff * r).mean()
        fft_loss = fft_weight * (complex_fft_l1_weighted(recon_vv, gt_vv) + complex_fft_l1_weighted(recon_vh, gt_vh))
        # Small TV on amplitude for stability
        amp_stack = torch.stack([amp_r_vv, amp_r_vh], dim=1)
        tv_loss = 8e-3 * _tv_l1(amp_stack)
    if perceptual is not None and perceptual_weight > 0.0:
        p_vv = perceptual(recon[:, :2], gt[:, :2])
        p_vh = perceptual(recon[:, 2:], gt[:, 2:])
        p_loss = perceptual_weight * (p_vv + p_vh)

    total_loss = mag_loss + phase_loss + coh_loss + spec_loss + dc_loss + fft_loss + phase_fft_loss + tv_loss + p_loss

    return {
        'total_loss': total_loss,
        'mag_loss': mag_loss,
        'phase_loss': phase_loss,
        'coh_loss': coh_loss,
        'spec_loss': spec_loss,
        'dc_loss': dc_loss,
        'fft_loss': fft_loss,
        'phase_fft_loss': phase_fft_loss,
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


def calculate_psnr_dual_pol(recon: torch.Tensor, gt: torch.Tensor, ref: str = 'fixed') -> dict:
    """Calculate PSNR for both VV and VH with reference selection.
    ref: 'fixed' uses PSNR_REF constants; 'per_dataset' uses gt max amplitude
    """
    # Convert 4-channel real to complex
    recon_vv = torch.complex(recon[:, 0], recon[:, 1])
    recon_vh = torch.complex(recon[:, 2], recon[:, 3])
    gt_vv = torch.complex(gt[:, 0], gt[:, 1])
    gt_vh = torch.complex(gt[:, 2], gt[:, 3])

    recon_vv_amp = torch.abs(recon_vv)
    recon_vh_amp = torch.abs(recon_vh)
    gt_vv_amp = torch.abs(gt_vv)
    gt_vh_amp = torch.abs(gt_vh)

    if ref == 'fixed':
        # Use globally defined constants for consistent PSNR comparison
        max_val_vv = torch.tensor(PSNR_REF_VV, device=recon.device)
        max_val_vh = torch.tensor(PSNR_REF_VH, device=recon.device)
    else:
        max_val_vv = gt_vv_amp.max()
        max_val_vh = gt_vh_amp.max()

    mse_vv = F.mse_loss(recon_vv_amp, gt_vv_amp)
    mse_vh = F.mse_loss(recon_vh_amp, gt_vh_amp)

    epsilon = 1e-9
    psnr_vv = 20 * torch.log10(max_val_vv / torch.sqrt(mse_vv + epsilon))
    psnr_vh = 20 * torch.log10(max_val_vh / torch.sqrt(mse_vh + epsilon))
    psnr_avg = (psnr_vv + psnr_vh) / 2

    return {
        'psnr_vv': psnr_vv.item(),
        'psnr_vh': psnr_vh.item(),
        'psnr_avg': psnr_avg.item()
    }


def _to_log_intensity(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return log-intensity image (natural log). Input can be 4-real or complex [B,2,H,W]."""
    c = _ensure_complex(t)
    I = torch.abs(c) ** 2
    return torch.log(I + eps)

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


def calculate_ssim_log(recon: torch.Tensor, gt: torch.Tensor, window_size=11) -> float:
    """SSIM computed on log-intensity images (연구질문.md metrics)."""
    recon_log = _to_log_intensity(recon)
    gt_log = _to_log_intensity(gt)
    # Reuse amplitude SSIM by treating logs as "amplitudes"
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(recon_log, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(gt_log, window_size, stride=1, padding=window_size//2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(recon_log ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(gt_log ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(recon_log * gt_log, window_size, stride=1, padding=window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
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


def calculate_local_coherence_avg(recon: torch.Tensor, gt: torch.Tensor, window: int = 11) -> float:
    """Average local coherence magnitude |γ| per 연구질문.md [§C]."""
    c = _ensure_complex(recon)
    g = _ensure_complex(gt)
    B, C, H, W = c.shape
    # Compute per-channel then average
    pad = window // 2
    def ap(x):
        return F.avg_pool2d(x, kernel_size=window, stride=1, padding=pad)
    vals = []
    for ch in range(C):
        r1, i1 = c[:, ch].real, c[:, ch].imag
        r2, i2 = g[:, ch].real, g[:, ch].imag
        real_num = ap(r1 * r2 + i1 * i2)
        imag_num = ap(i1 * r2 - r1 * i2)
        num = torch.sqrt(real_num**2 + imag_num**2 + 1e-16)
        den = torch.sqrt(ap(r1*r1 + i1*i1) * ap(r2*r2 + i2*i2) + 1e-16)
        vals.append((num / (den + 1e-12)).clamp(0.0, 1.0).mean())
    return torch.stack(vals).mean().item()


def radial_spectrum_deviation(recon: torch.Tensor, gt: torch.Tensor) -> float:
    """Optional: Radial spectrum deviation on amplitude spectra (lower is better)."""
    c = _ensure_complex(recon)
    g = _ensure_complex(gt)
    # Use only VV channel for simplicity
    rp = torch.abs(torch.fft.fftshift(torch.fft.fft2(c[:, 0])))
    rg = torch.abs(torch.fft.fftshift(torch.fft.fft2(g[:, 0])))
    H, W = rp.shape[-2:]
    cy, cx = H // 2, W // 2
    R = min(cy, cx)
    # Sample rings and compute mean per radius
    r_vals = torch.linspace(1, R, steps=32, device=rp.device)
    diffs = []
    yy, xx = torch.meshgrid(torch.arange(H, device=rp.device), torch.arange(W, device=rp.device), indexing='ij')
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    for r in r_vals:
        band = (rr >= r - 0.5) & (rr < r + 0.5)
        if band.any():
            diffs.append((rp[..., band].mean() - rg[..., band].mean()).abs())
    if len(diffs) == 0:
        return 0.0
    return torch.stack(diffs).mean().item()


class MetricsCalculator:
    """Comprehensive metrics calculator for SAR super-resolution."""
    
    def __init__(self, psnr_ref: str = 'fixed', eval_window: int = 11):
        self.psnr_ref = psnr_ref
        self.eval_window = eval_window
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
            
            # Quality metrics - dual polarization (amplitude reference)
            psnr_results = calculate_psnr_dual_pol(recon, gt, ref=self.psnr_ref)
            self.metrics['psnr_vv'].append(psnr_results['psnr_vv'])
            self.metrics['psnr_vh'].append(psnr_results['psnr_vh'])
            self.metrics['psnr_avg'].append(psnr_results['psnr_avg'])
            
            # Backward compatibility - use VV PSNR for 'psnr' key
            self.metrics['psnr'].append(psnr_results['psnr_vv'])
            
            # Convert to complex for other metrics (using _ensure_complex)
            recon_complex = _ensure_complex(recon)
            gt_complex = _ensure_complex(gt)
            
            self.metrics['ssim'].append(calculate_ssim(recon_complex, gt_complex, window_size=self.eval_window))
            # PSNR/SSIM on log-intensity (연구질문.md)
            # Define PSNR on log-intensity with fixed ref of 1.0 in log-space
            log_recon = _to_log_intensity(recon_complex)
            log_gt = _to_log_intensity(gt_complex)
            mse_log = F.mse_loss(log_recon, log_gt)
            self.metrics.setdefault('psnr_log', [])
            self.metrics['psnr_log'].append((20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse_log + 1e-9))).item())
            self.metrics.setdefault('ssim_log', [])
            self.metrics['ssim_log'].append(calculate_ssim_log(recon_complex, gt_complex, window_size=self.eval_window))
            self.metrics['cpif'].append(calculate_cpif(recon_complex, gt_complex).item())
            self.metrics['rmse'].append(calculate_rmse(recon, gt))

            # Amplitude maxima for summary
            amp_pred = torch.abs(recon_complex)
            amp_gt = torch.abs(gt_complex)
            self.metrics.setdefault('amp_max_pred', [])
            self.metrics.setdefault('amp_max_hr', [])
            self.metrics['amp_max_pred'].append(amp_pred.max().item())
            self.metrics['amp_max_hr'].append(amp_gt.max().item())
            
            # Phase statistics
            phase_stats = calculate_phase_difference_stats(recon_complex, gt_complex)
            self.metrics['phase_rmse'].append(phase_stats['phase_rmse'])
            # Back-compat: provide wrapped_phase_rmse alias
            self.metrics.setdefault('wrapped_phase_rmse', [])
            self.metrics['wrapped_phase_rmse'].append(phase_stats['phase_rmse'])
            self.metrics.setdefault('mean_phase_error', [])
            self.metrics['mean_phase_error'].append(phase_stats['mean_phase_error'])
            # Local coherence magnitude average
            self.metrics.setdefault('coherence_avg', [])
            self.metrics['coherence_avg'].append(calculate_local_coherence_avg(recon_complex, gt_complex, window=self.eval_window))
            # Optional spectral deviation
            self.metrics.setdefault('radial_spectrum_dev', [])
            self.metrics['radial_spectrum_dev'].append(radial_spectrum_deviation(recon_complex, gt_complex))
    
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
