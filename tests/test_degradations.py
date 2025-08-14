#!/usr/bin/env python3
import math
import numpy as np
import torch

# Implicit namespace package import of model/* modules
from model.degradations import degrade_complex_lp, LPParams


def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_anti_alias_decimation_passband():
    """Verify low-pass filtering suppresses near-Nyquist energy before decimation.
    [연구질문.md §4.1] anti-aliased decimation via H then D.
    """
    set_seed(0)

    H, W = 128, 128
    # Build a sinusoid near Nyquist in azimuth (horizontal stripes)
    y = torch.arange(H).view(H, 1)
    x = torch.arange(W).view(1, W)
    freq = 0.45  # cycles/pixel near Nyquist 0.5
    phase = 2 * math.pi * (x * freq)
    img = torch.cos(phase).float()
    c = torch.complex(img.clone(), torch.zeros_like(img))  # single-pol pattern
    c = torch.stack([c, c], dim=0)  # [2,H,W]

    # Apply sinc low-pass with cutoff below freq before decimation by 2
    lp = LPParams(kind='sinc', sigma_px=1.2, size=15, beta=12.0, cutoff=0.35)
    lr = degrade_complex_lp(c.to(torch.complex64), scale=2, lp_params=lp, device=torch.device('cpu'))

    # Expect strong attenuation since freq (0.45) > cutoff (0.35)
    # Measure amplitude reduction between HR and upsampled LR back to HxW
    amp_hr = torch.abs(c)
    amp_lr = torch.abs(lr)
    amp_lr_up = torch.nn.functional.interpolate(amp_lr.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    ratio = (amp_lr_up.mean() / (amp_hr.mean() + 1e-8)).item()
    assert ratio < 0.3, f"Expected strong attenuation, got ratio={ratio:.3f}"


def test_speckle_gamma_enl_variance():
    """Check multiplicative speckle stats: Var(G)≈1/L on intensity.
    [연구질문.md §4.3]
    """
    set_seed(1)
    H, W = 128, 128
    # Constant intensity scene
    amp = torch.ones(H, W)
    phase = torch.zeros(H, W)
    c = amp * torch.exp(1j * phase)
    c = torch.stack([c, c], dim=0).to(torch.complex64)
    L = 4.0
    lp = LPParams(kind='gaussian', sigma_px=1.0, size=9)
    lr = degrade_complex_lp(c, scale=2, lp_params=lp, device=torch.device('cpu'), enl=L, noise_std=None)
    I = torch.abs(lr) ** 2
    G_hat = (I / (torch.ones_like(I))).view(-1)
    var_est = G_hat.var().item()
    assert abs(var_est - (1.0 / L)) < 0.1, f"Var(G) ~ 1/L expected, got {var_est:.3f}"


def test_additive_thermal_noise_floor():
    """Verify additive complex noise raises noise floor.
    [연구질문.md §4.3]
    """
    set_seed(2)
    H, W = 64, 64
    c = torch.zeros(2, H, W, dtype=torch.complex64)
    lp = LPParams(kind='gaussian', sigma_px=1.0, size=7)
    lr_no = degrade_complex_lp(c, scale=2, lp_params=lp, device=torch.device('cpu'), enl=None, noise_std=None)
    lr_n = degrade_complex_lp(c, scale=2, lp_params=lp, device=torch.device('cpu'), enl=None, noise_std=0.1)
    p0 = (torch.abs(lr_no) ** 2).mean().item()
    p1 = (torch.abs(lr_n) ** 2).mean().item()
    assert p1 > p0 + 1e-6, "Thermal noise should elevate mean power"


