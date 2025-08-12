#!/usr/bin/env python3
"""
Degradation operators for synthetic LR generation from HR complex SAR patches
-----------------------------------------------------------------------------

Two LR generation modes controlled via CLI:
- complex_lp: Convolve complex HR field with a low-pass PSF then decimate
- mean_amp_phase: Baseline amplitude mean + circular phase mean (ablation)

All convolutions use reflect padding and kernels are energy-normalized.
The decimation factor is inferred from (hr_size / lr_size), which must be
an integer ratio in both height and width.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LPParams:
    kind: str = 'gaussian'  # 'gaussian' | 'sinc'
    sigma_px: float = 1.2
    size: int = 9           # odd kernel size
    beta: float = 12.0      # kaiser window beta (used for 'sinc')

    def to_meta(self) -> Dict:
        d = asdict(self)
        d['meta_version'] = 1
        return d


def _ensure_odd(n: int) -> int:
    return int(n) if int(n) % 2 == 1 else int(n) + 1


def make_psf(kind: str, sigma_px: float, size: int, beta: float = 12.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Create a 2D low-pass PSF kernel.

    - kind='gaussian': isotropic Gaussian with std=sigma_px
    - kind='sinc': ideal low-pass sinc with Kaiser window (beta)

    Returns float32 tensor [1,1,kH,kW], sum normalized to 1.0
    """
    size = _ensure_odd(size)
    half = size // 2
    yy, xx = torch.meshgrid(torch.arange(-half, half + 1, device=device),
                            torch.arange(-half, half + 1, device=device), indexing='ij')
    rr = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)

    if kind.lower() == 'gaussian':
        sigma = max(1e-6, float(sigma_px))
        k = torch.exp(-(rr ** 2) / (2.0 * (sigma ** 2)))
    elif kind.lower() == 'sinc':
        # Ideal low-pass with cutoff ~ 1/(pi*sigma)
        # Avoid div by zero at center: sinc(0) = 1
        eps = 1e-8
        # Empirical mapping: smaller sigma -> wider passband
        cutoff = 1.0 / max(1.0, float(np.pi * sigma_px))
        r = rr * cutoff
        k = torch.sin(np.pi * r + eps) / (np.pi * r + eps)
        # Kaiser window to control sidelobes
        # I0-based Kaiser window approximation (torch has no direct I0)
        a = beta
        t = (rr / (half + 1e-8)).clamp(0, 1)
        # Approximate I0 using a truncated series (sufficient for windowing)
        def i0(x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            y2 = (y / 2) ** 2
            out = torch.ones_like(y)
            term = torch.ones_like(y)
            for m in range(1, 6):  # 5 terms
                term = term * y2 / (m ** 2)
                out = out + term
            return out
        w = i0(a * torch.sqrt(1 - t ** 2)) / i0(torch.tensor(a, device=device))
        k = k * w
    else:
        raise ValueError(f"Unknown PSF kind: {kind}")

    k = k / (k.sum() + 1e-12)
    return k.view(1, 1, size, size).float()


def _conv2d_reflect(x: torch.Tensor, kernel: torch.Tensor, groups: int) -> torch.Tensor:
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    assert x.dim() == 4, f"expected 4D (N,C,H,W), got {x.shape}"
    x = x.to(kernel.device)
    pad = torch.nn.ReflectionPad2d((pad_w, pad_w, pad_h, pad_h))
    x = pad(x)
    weight = kernel.repeat(groups, 1, 1, 1)
    return F.conv2d(x, weight, stride=1, padding=0, groups=groups)


def degrade_complex_lp(hr_complex: torch.Tensor, scale: int, lp_params: LPParams, device: torch.device) -> torch.Tensor:
    """
    hr_complex: Tensor [2,H,W] complex or stacked real-imag as complex dtype
    Returns: complex tensor [2, H/scale, W/scale]
    """
    assert hr_complex.ndim == 3 and hr_complex.shape[0] == 2, f"Expected [2,H,W], got {hr_complex.shape}"
    if not torch.is_complex(hr_complex):
        raise ValueError("hr_complex must be complex tensor of shape [2,H,W]")
    c = hr_complex  # [2,H,W] complex, channels are polarizations
    # Guard: autocast(bf16) may produce bfloat16 real/imag; promote to float32
    if c.real.dtype not in (torch.float32, torch.float64):
        c = torch.complex(c.real.float(), c.imag.float())
    # Convolve real and imag with same PSF (depthwise over channels)
    kernel = make_psf(lp_params.kind, lp_params.sigma_px, lp_params.size, lp_params.beta, device=device)
    real = _conv2d_reflect(c.real.unsqueeze(0).float(), kernel, groups=c.shape[0]).squeeze(0)  # [2,H,W]
    imag = _conv2d_reflect(c.imag.unsqueeze(0).float(), kernel, groups=c.shape[0]).squeeze(0)  # [2,H,W]
    c_lp = torch.complex(real.float(), imag.float())  # [2,H,W]
    # Decimate by integer factor (spatial dims only)
    lr = c_lp[:, ::scale, ::scale]  # [2,H/scale,W/scale]
    return lr


def degrade_mean_amp_phase(hr_complex: torch.Tensor, scale: int) -> torch.Tensor:
    """Baseline: amplitude mean + circular mean of phase within scale×scale blocks per polarization."""
    assert hr_complex.ndim == 3 and hr_complex.shape[0] == 2
    if not torch.is_complex(hr_complex):
        raise ValueError("hr_complex must be complex tensor of shape [2,H,W]")
    c = hr_complex  # [2,H,W]
    # Avg pool on power and cos/sin of phase per-channel
    power = (c.real.float() ** 2 + c.imag.float() ** 2).unsqueeze(0)  # [1,2,H,W]
    power_avg = F.avg_pool2d(power, scale, stride=scale)  # [1,2,h,w]
    amp_lr = torch.sqrt(power_avg + 1e-12).squeeze(0)     # [2,h,w]

    phase = torch.angle(torch.complex(c.real.float(), c.imag.float())).unsqueeze(0)  # [1,2,H,W]
    cos_p = F.avg_pool2d(torch.cos(phase), scale, stride=scale)  # [1,2,h,w]
    sin_p = F.avg_pool2d(torch.sin(phase), scale, stride=scale)  # [1,2,h,w]
    phase_lr = torch.atan2(sin_p, cos_p).squeeze(0)  # [2,h,w]

    lr = amp_lr * torch.exp(1j * phase_lr)  # [2,h,w] complex
    return lr


def build_meta(lr_mode: str, scale: int, lp_params: LPParams | None) -> Dict:
    meta = {
        'meta_version': 2,
        'lr_mode': lr_mode,
        'scale': int(scale),
    }
    if lr_mode == 'complex_lp' and lp_params is not None:
        meta.update(lp_params.to_meta())
    return meta


def degrade_from_params(hr_np: np.ndarray,
                        lr_size: Tuple[int, int],
                        hr_size: Tuple[int, int],
                        lr_mode: str,
                        lp_params: LPParams,
                        device: torch.device = torch.device('cpu')) -> np.ndarray:
    """Entry point used by dataset. Returns np.complex64 (2,h,w)."""
    # Inputs are (2,H,W) complex64 np array
    hr_t = torch.from_numpy(hr_np).to(device)
    H, W = hr_size
    h, w = lr_size
    assert H % h == 0 and W % w == 0, f"Non-integer scale: HR {hr_size} -> LR {lr_size}"
    scale_h = H // h
    scale_w = W // w
    assert scale_h == scale_w, f"Anisotropic scale not supported: {scale_h} vs {scale_w}"
    scale = scale_h

    if lr_mode == 'complex_lp':
        lr_t = degrade_complex_lp(hr_t, scale, lp_params, device=device)
    elif lr_mode == 'mean_amp_phase':
        lr_t = degrade_mean_amp_phase(hr_t, scale)
    else:
        raise ValueError(f"Unsupported lr_mode: {lr_mode}")

    return lr_t.cpu().numpy().astype(np.complex64)


