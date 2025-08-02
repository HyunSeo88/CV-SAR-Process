#!/usr/bin/env python3
"""
Complex U-Net for SAR Super-Resolution
=====================================

Lightweight Complex U-Net implementation for 4x SAR super-resolution.
Designed for Korean disaster monitoring with dual-pol (VV+VH) Sentinel-1 data.

Key features:
- Complex-valued convolutions preserving phase information
- U-Net architecture with skip connections
- Residual learning for better convergence
- ~1M parameters for efficient training

Input: (2, 64, 128) dual-pol LR complex patches
Output: (2, 256, 512) HR complex patches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplexUNet(nn.Module):
    """
    Complex U-Net for SAR Super-Resolution
    
    Architecture:
    - Encoder: 3 ComplexConv layers with MaxPool (2→64→128→256 channels)
    - Decoder: 3 Upsample + ComplexConv layers (256→128→64→2 channels)
    - Skip connections between encoder-decoder pairs
    - Residual connection with upsampled input
    """
    
    def __init__(self):
        super(ComplexUNet, self).__init__()
        
        # Encoder now takes 3 real channels
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Real conv
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder for VV features (output 2 channels: real/imag)
        self.dec3 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.dec1 = nn.Conv2d(64 + 3, 2, kernel_size=3, stride=1, padding=1)  # Output 2 channels
        
        # Attention using VH mag (enhanced to two-layer with ReLU for richer representation)
        self.attention = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1)
        )   # Enhanced attention with 3x3 convs for better texture capture
        
        # SE attention (replacing vh_mag attention)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64//16, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        print(f"ComplexUNet initialized with {self.count_parameters()} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):  # x: [B,3,H,W] real
        # Encoder
        enc1_out = F.relu(self.enc1(x))
        enc1_pooled = self.pool(enc1_out)
        
        enc2_out = F.relu(self.enc2(enc1_pooled))
        enc2_pooled = self.pool(enc2_out)
        
        enc3_out = F.relu(self.enc3(enc2_pooled))
        
        # Decoder with attention from VH mag
        dec3_up = F.interpolate(enc3_out, size=enc2_out.shape[2:], mode='bilinear', align_corners=False)
        dec3_skip = torch.cat([dec3_up, enc2_out], dim=1)
        dec3_out = F.relu(self.dec3(dec3_skip))
        
        dec2_up = F.interpolate(dec3_out, size=enc1_out.shape[2:], mode='bilinear', align_corners=False)
        dec2_skip = torch.cat([dec2_up, enc1_out], dim=1)
        dec2_out = F.relu(self.dec2(dec2_skip))
        
        # SE attention (replacing vh_mag attention)
        dec2_out = dec2_out * self.se(dec2_out)
        
        # Upsample VH mag for attention
        vh_mag_up = F.interpolate(x[:,2:3,:,:], size=dec2_out.shape[2:], mode='bicubic', align_corners=False) # Unified bicubic mode for consistency
        
        attn = torch.sigmoid(self.attention(vh_mag_up))
        se_out = self.se(dec2_out)
        dec2_out += attn * se_out  # Additive fusion
        
        dec1_up = F.interpolate(dec2_out, scale_factor=4, mode='bilinear', align_corners=False)
        input_upsampled = F.interpolate(x, size=dec1_up.shape[2:], mode='bilinear', align_corners=False)
        dec1_skip = torch.cat([dec1_up, input_upsampled], dim=1)
        output = self.dec1(dec1_skip)  # [B,2,H',W']
        
        # Residual connection
        vv_input = torch.complex(x[:,0], x[:,1]).unsqueeze(1)  # [B,1,H,W] complex
        vv_r = F.interpolate(vv_input.real, size=output.shape[-2:], mode='bicubic', align_corners=False)
        vv_i = F.interpolate(vv_input.imag, size=output.shape[-2:], mode='bicubic', align_corners=False)
        residual = torch.cat([vv_r, vv_i], dim=1)  # (B,2,H,W)
        # FIX: Use in-place addition for better memory efficiency
        output.add_(residual)
        
        return output  # Keep output as float tensor (2 channels: real & imag) # Keep as real tensor to match visualization
    
    
    

def create_model():
    """Create and return ComplexUNet model"""
    model = ComplexUNet()
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing ComplexUNet...")
    
    # Create model
    model = create_model()
    
    test_input = torch.randn(1, 3, 64, 128).float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_input = test_input.to(device)
    model.to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    print(f"Output is complex: {torch.is_complex(output)}")
    print(f"Output dtype: {output.dtype}")
    
    print("ComplexUNet test completed successfully!")