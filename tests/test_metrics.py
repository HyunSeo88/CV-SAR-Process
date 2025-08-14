#!/usr/bin/env python3
import numpy as np
import torch

from model.utils import calculate_phase_difference_stats, calculate_local_coherence_avg


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_plane_wave(B=2, H=64, W=64, kx=0.1, ky=0.0, phase_offset=0.0):
    y = torch.arange(H).view(1, H, 1)
    x = torch.arange(W).view(1, 1, W)
    phi = 2 * np.pi * (kx * x + ky * y) + phase_offset
    c = torch.exp(1j * phi).repeat(B, 1, 1)
    return c


def test_phase_error_zero_when_identical():
    set_seed(0)
    c = make_plane_wave()
    stats = calculate_phase_difference_stats(c.unsqueeze(1), c.unsqueeze(1))
    assert stats['mean_phase_error'] < 1e-6
    assert stats['phase_rmse'] < 1e-6


def test_phase_error_matches_offset():
    set_seed(0)
    c1 = make_plane_wave(phase_offset=0.0)
    c2 = make_plane_wave(phase_offset=np.pi / 4)
    stats = calculate_phase_difference_stats(c1.unsqueeze(1), c2.unsqueeze(1))
    # Due to wrapping, mean absolute should be close to pi/4
    assert abs(stats['mean_phase_error'] - np.pi / 4) < 0.05


def test_local_coherence_reduces_with_noise():
    set_seed(0)
    B, H, W = 2, 64, 64
    c_gt = make_plane_wave(B=B, H=H, W=W)
    # Add complex Gaussian noise
    noise = 0.5 * (torch.randn_like(c_gt.real) + 1j * torch.randn_like(c_gt.real))
    c_noisy = c_gt + noise
    coh_clean = calculate_local_coherence_avg(c_gt.unsqueeze(1), c_gt.unsqueeze(1), window=11)
    coh_noisy = calculate_local_coherence_avg(c_noisy.unsqueeze(1), c_gt.unsqueeze(1), window=11)
    assert coh_noisy < coh_clean


