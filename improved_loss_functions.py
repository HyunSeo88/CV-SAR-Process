#!/usr/bin/env python3
"""
Improved Loss Functions for SAR Super-Resolution
================================================

Enhanced loss functions to address:
1. Speckle over-smoothing (current preservation: 0.0)
2. Phase information loss (current RMSE: 1.06 rad, correlation: 0.34)
3. Perceptual quality for disaster monitoring applications

References:
- Johnson et al. "Perceptual Losses for Real-Time Style Transfer" (2016)
- Zhao et al. "Loss Functions for Image Restoration with Neural Networks" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpecklePreservingLoss(nn.Module):
    """
    Speckle pattern preservation loss to maintain SAR texture characteristics
    
    Rationale: Current speckle preservation is 0.0, indicating over-smoothing.
    This loss encourages local variance preservation essential for SAR interpretation.
    """
    
    def __init__(self, window_size: int = 7, weight: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.weight = weight
        
    def forward(self, sr_complex: torch.Tensor, gt_complex: torch.Tensor) -> torch.Tensor:
        """
        Calculate speckle preservation loss
        
        Args:
            sr_complex: Super-resolved complex tensor [B, C, H, W]
            gt_complex: Ground truth complex tensor [B, C, H, W]
        """
        sr_amp = torch.abs(sr_complex)
        gt_amp = torch.abs(gt_complex)
        
        # Calculate local variance using avg_pool2d
        kernel_size = self.window_size
        
        # Local mean
        sr_mean = F.avg_pool2d(sr_amp, kernel_size, stride=1, padding=kernel_size//2)
        gt_mean = F.avg_pool2d(gt_amp, kernel_size, stride=1, padding=kernel_size//2)
        
        # Local variance
        sr_var = F.avg_pool2d(sr_amp**2, kernel_size, stride=1, padding=kernel_size//2) - sr_mean**2
        gt_var = F.avg_pool2d(gt_amp**2, kernel_size, stride=1, padding=kernel_size//2) - gt_mean**2
        
        # Speckle index (coefficient of variation)
        sr_speckle = sr_var / (sr_mean + 1e-8)
        gt_speckle = gt_var / (gt_mean + 1e-8)
        
        # L1 loss between speckle indices
        speckle_loss = F.l1_loss(sr_speckle, gt_speckle)
        
        return self.weight * speckle_loss


class CircularPhaseLoss(nn.Module):
    """
    Circular phase loss for better phase preservation
    
    Rationale: Current phase RMSE is 1.06 rad and correlation is 0.34.
    Standard MSE doesn't handle phase wrapping properly for SAR applications.
    """
    
    def __init__(self, weight: float = 0.3):
        super().__init__()
        self.weight = weight
        
    def forward(self, sr_complex: torch.Tensor, gt_complex: torch.Tensor) -> torch.Tensor:
        """
        Calculate circular phase loss with proper wrapping
        
        Args:
            sr_complex: Super-resolved complex tensor [B, C, H, W]
            gt_complex: Ground truth complex tensor [B, C, H, W]
        """
        sr_phase = torch.angle(sr_complex)
        gt_phase = torch.angle(gt_complex)
        
        # Circular difference: handles phase wrapping
        phase_diff = sr_phase - gt_phase
        circular_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        # Circular MSE
        circular_mse = torch.mean(circular_diff**2)
        
        return self.weight * circular_mse


class CoherenceLoss(nn.Module):
    """
    Cross-polarization coherence preservation loss
    
    Rationale: Current coherence correlation is -0.7063 (negative).
    Essential for maintaining polarimetric relationships in SAR data.
    """
    
    def __init__(self, weight: float = 0.2, window_size: int = 5):
        super().__init__()
        self.weight = weight
        self.window_size = window_size
        
    def forward(self, sr_complex: torch.Tensor, gt_complex: torch.Tensor) -> torch.Tensor:
        """
        Calculate coherence preservation loss
        
        Args:
            sr_complex: Super-resolved complex tensor [B, 2, H, W] (VV, VH)
            gt_complex: Ground truth complex tensor [B, 2, H, W] (VV, VH)
        """
        if sr_complex.shape[1] != 2:
            return torch.tensor(0.0, device=sr_complex.device)
        
        # Extract VV and VH channels
        sr_vv, sr_vh = sr_complex[:, 0], sr_complex[:, 1]
        gt_vv, gt_vh = gt_complex[:, 0], gt_complex[:, 1]
        
        # Calculate coherence for SR and GT
        sr_coherence = self._calculate_coherence(sr_vv, sr_vh)
        gt_coherence = self._calculate_coherence(gt_vv, gt_vh)
        
        # L1 loss between coherence maps
        coherence_loss = F.l1_loss(sr_coherence, gt_coherence)
        
        return self.weight * coherence_loss
    
    def _calculate_coherence(self, pol1: torch.Tensor, pol2: torch.Tensor) -> torch.Tensor:
        """Calculate coherence between two polarizations"""
        # Cross-product
        cross_prod = pol1 * torch.conj(pol2)
        power1 = torch.abs(pol1)**2
        power2 = torch.abs(pol2)**2
        
        # Spatial averaging
        kernel_size = self.window_size
        avg_cross = F.avg_pool2d(cross_prod.real, kernel_size, stride=1, padding=kernel_size//2) + \
                   1j * F.avg_pool2d(cross_prod.imag, kernel_size, stride=1, padding=kernel_size//2)
        avg_pow1 = F.avg_pool2d(power1, kernel_size, stride=1, padding=kernel_size//2)
        avg_pow2 = F.avg_pool2d(power2, kernel_size, stride=1, padding=kernel_size//2)
        
        # Coherence magnitude
        coherence = torch.abs(avg_cross) / torch.sqrt(avg_pow1 * avg_pow2 + 1e-8)
        
        return coherence


class PerceptualLoss(nn.Module):
    """
    Perceptual loss for SAR amplitude images
    
    Rationale: Standard pixel-wise losses don't capture perceptual quality
    important for visual interpretation in disaster monitoring.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
        # Use simple gradient-based features for SAR
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, sr_complex: torch.Tensor, gt_complex: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss based on edge features
        
        Args:
            sr_complex: Super-resolved complex tensor [B, C, H, W]
            gt_complex: Ground truth complex tensor [B, C, H, W]
        """
        sr_amp = torch.abs(sr_complex)
        gt_amp = torch.abs(gt_complex)
        
        # Calculate edge features for each channel
        perceptual_loss = 0.0
        num_channels = sr_amp.shape[1]
        
        for ch in range(num_channels):
            sr_ch = sr_amp[:, ch:ch+1]
            gt_ch = gt_amp[:, ch:ch+1]
            
            # Edge detection
            sr_edge_x = F.conv2d(sr_ch, self.sobel_x, padding=1)
            sr_edge_y = F.conv2d(sr_ch, self.sobel_y, padding=1)
            sr_edges = torch.sqrt(sr_edge_x**2 + sr_edge_y**2)
            
            gt_edge_x = F.conv2d(gt_ch, self.sobel_x, padding=1)
            gt_edge_y = F.conv2d(gt_ch, self.sobel_y, padding=1)
            gt_edges = torch.sqrt(gt_edge_x**2 + gt_edge_y**2)
            
            # L1 loss on edge features
            perceptual_loss += F.l1_loss(sr_edges, gt_edges)
        
        perceptual_loss /= num_channels
        
        return self.weight * perceptual_loss


class EnhancedSARLoss(nn.Module):
    """
    Combined enhanced loss function for SAR super-resolution
    
    Addresses all identified issues:
    - Over-smoothing (speckle preservation)
    - Phase information loss
    - Perceptual quality
    - Coherence preservation
    """
    
    def __init__(self, 
                 amp_weight: float = 0.4,
                 phase_weight: float = 0.2,
                 speckle_weight: float = 0.1,
                 coherence_weight: float = 0.2,
                 perceptual_weight: float = 0.1):
        super().__init__()
        
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight
        
        self.speckle_loss = SpecklePreservingLoss(weight=speckle_weight)
        self.phase_loss = CircularPhaseLoss(weight=phase_weight)
        self.coherence_loss = CoherenceLoss(weight=coherence_weight)
        self.perceptual_loss = PerceptualLoss(weight=perceptual_weight)
        
    def forward(self, sr_complex: torch.Tensor, gt_complex: torch.Tensor) -> dict:
        """
        Calculate enhanced SAR loss
        
        Returns:
            dict: Loss components for monitoring
        """
        # Basic amplitude loss
        sr_amp = torch.abs(sr_complex)
        gt_amp = torch.abs(gt_complex)
        amp_loss = F.mse_loss(sr_amp, gt_amp)
        
        # Enhanced loss components
        speckle_loss = self.speckle_loss(sr_complex, gt_complex)
        phase_loss = self.phase_loss(sr_complex, gt_complex)
        coherence_loss = self.coherence_loss(sr_complex, gt_complex)
        perceptual_loss = self.perceptual_loss(sr_complex, gt_complex)
        
        # Total loss
        total_loss = (self.amp_weight * amp_loss + 
                     speckle_loss + phase_loss + 
                     coherence_loss + perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'amp_loss': amp_loss,
            'speckle_loss': speckle_loss,
            'phase_loss': phase_loss,
            'coherence_loss': coherence_loss,
            'perceptual_loss': perceptual_loss
        }


if __name__ == "__main__":
    # Test enhanced loss functions
    print("Testing Enhanced SAR Loss Functions...")
    
    # Create test data
    batch_size, channels, height, width = 2, 2, 64, 128
    sr_real = torch.randn(batch_size, channels, height, width)
    sr_imag = torch.randn(batch_size, channels, height, width)
    sr_complex = torch.complex(sr_real, sr_imag)
    
    gt_real = torch.randn(batch_size, channels, height, width)
    gt_imag = torch.randn(batch_size, channels, height, width)
    gt_complex = torch.complex(gt_real, gt_imag)
    
    # Test enhanced loss
    enhanced_loss = EnhancedSARLoss()
    loss_dict = enhanced_loss(sr_complex, gt_complex)
    
    print("Loss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("\nEnhanced loss functions test completed successfully!") 