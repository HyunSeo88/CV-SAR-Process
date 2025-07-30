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


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D convolution layer
    
    Implements complex convolution as:
    out_real = conv_real(real) - conv_imag(imag)
    out_imag = conv_real(imag) + conv_imag(real)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ComplexConv2d, self).__init__()
        
        # Separate convolutions for real and imaginary parts
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        nn.init.xavier_uniform_(self.conv_real.weight)
        nn.init.xavier_uniform_(self.conv_imag.weight)
        if self.conv_real.bias is not None:
            nn.init.zeros_(self.conv_real.bias)
            nn.init.zeros_(self.conv_imag.bias)
    
    def forward(self, x):
        """
        Forward pass for complex convolution
        
        Args:
            x: Complex tensor as [batch, channel, height, width, 2] where [..., 0] is real, [..., 1] is imag
            
        Returns:
            Complex output tensor in same format
        """
        if x.dim() == 4:
            # Assume input is real-only, convert to complex format
            real_part = x
            imag_part = torch.zeros_like(x)
        else:
            # Extract real and imaginary parts
            real_part = x[..., 0]
            imag_part = x[..., 1]
        
        # Complex convolution operations
        out_real = self.conv_real(real_part) - self.conv_imag(imag_part)
        out_imag = self.conv_real(imag_part) + self.conv_imag(real_part)
        
        # Stack back to complex format
        output = torch.stack([out_real, out_imag], dim=-1)
        return output


class ComplexReLU(nn.Module):
    """
    Complex ReLU activation applied to magnitude
    
    Preserves phase while applying ReLU to magnitude:
    output = ReLU(|z|) * exp(i * angle(z))
    """
    
    def forward(self, x):
        """Apply complex ReLU activation"""
        real_part = x[..., 0]
        imag_part = x[..., 1]
        
        # Calculate magnitude and phase
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)
        
        # Apply ReLU to magnitude
        activated_magnitude = F.relu(magnitude)
        
        # Reconstruct complex number
        out_real = activated_magnitude * torch.cos(phase)
        out_imag = activated_magnitude * torch.sin(phase)
        
        return torch.stack([out_real, out_imag], dim=-1)


class ComplexConvBlock(nn.Module):
    """
    Complex convolution block with activation
    
    ComplexConv2d -> ComplexReLU
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ComplexConvBlock, self).__init__()
        
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = ComplexReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


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
        
        # Encoder layers
        self.enc1 = ComplexConvBlock(2, 64)      # Input: 2 channels (VV, VH)
        self.enc2 = ComplexConvBlock(64, 128)
        self.enc3 = ComplexConvBlock(128, 256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder layers
        self.dec3 = ComplexConvBlock(256 + 128, 128)  # +128 from skip connection
        self.dec2 = ComplexConvBlock(128 + 64, 64)    # +64 from skip connection
        self.dec1 = ComplexConvBlock(64 + 2, 2)       # +2 from skip connection (upsampled input)
        
        # Final output layer
        self.final_conv = ComplexConv2d(2, 2, kernel_size=1, padding=0)
        
        print(f"ComplexUNet initialized with {self.count_parameters()} parameters")
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _convert_to_complex_format(self, x):
        """Convert standard complex tensor to internal format"""
        if torch.is_complex(x):
            # Convert from complex64/128 to real tensor with last dim=2
            real_part = x.real
            imag_part = x.imag
            return torch.stack([real_part, imag_part], dim=-1)
        else:
            # Assume input is real-only
            imag_part = torch.zeros_like(x)
            return torch.stack([x, imag_part], dim=-1)
    
    def _convert_from_complex_format(self, x):
        """Convert internal format back to complex tensor"""
        real_part = x[..., 0]
        imag_part = x[..., 1]
        return torch.complex(real_part, imag_part)
    
    def forward(self, x):
        """
        Forward pass of Complex U-Net
        
        Args:
            x: Input complex tensor [batch, 2, 64, 128] (complex64/128 or real tensor)
            
        Returns:
            Super-resolved complex tensor [batch, 2, 256, 512]
        """
        # Store original input format
        input_is_complex = torch.is_complex(x)
        
        # Convert to internal complex format [batch, channel, height, width, 2]
        x_complex = self._convert_to_complex_format(x)
        
        # Store original input for residual connection
        original_input = x_complex
        
        # Encoder path
        # Level 1: (2, 64, 128) -> (64, 64, 128)
        enc1_out = self.enc1(x_complex)
        enc1_pooled = self._pool_complex(enc1_out)  # -> (64, 32, 64)
        
        # Level 2: (64, 32, 64) -> (128, 32, 64)
        enc2_out = self.enc2(enc1_pooled)
        enc2_pooled = self._pool_complex(enc2_out)  # -> (128, 16, 32)
        
        # Level 3: (128, 16, 32) -> (256, 16, 32)
        enc3_out = self.enc3(enc2_pooled)
        
        # Decoder path
        # Level 3: (256, 16, 32) -> (256, 32, 64)
        dec3_up = self._upsample_complex(enc3_out, scale_factor=2)
        
        # Ensure spatial dimensions match for skip connection
        if dec3_up.shape[2:4] != enc2_out.shape[2:4]:
            # Resize enc2_out to match dec3_up
            target_h, target_w = dec3_up.shape[2], dec3_up.shape[3]
            enc2_real = F.interpolate(enc2_out[..., 0], size=(target_h, target_w), mode='bilinear', align_corners=False)
            enc2_imag = F.interpolate(enc2_out[..., 1], size=(target_h, target_w), mode='bilinear', align_corners=False)
            enc2_out = torch.stack([enc2_real, enc2_imag], dim=-1)
        
        dec3_skip = torch.cat([dec3_up, enc2_out], dim=1)  # Concatenate along channel dim
        dec3_out = self.dec3(dec3_skip)  # -> (128, 32, 64)
        
        # Level 2: (128, 32, 64) -> (128, 64, 128)
        dec2_up = self._upsample_complex(dec3_out, scale_factor=2)
        
        # Ensure spatial dimensions match for skip connection
        if dec2_up.shape[2:4] != enc1_out.shape[2:4]:
            # Resize enc1_out to match dec2_up
            target_h, target_w = dec2_up.shape[2], dec2_up.shape[3]
            enc1_real = F.interpolate(enc1_out[..., 0], size=(target_h, target_w), mode='bilinear', align_corners=False)
            enc1_imag = F.interpolate(enc1_out[..., 1], size=(target_h, target_w), mode='bilinear', align_corners=False)
            enc1_out = torch.stack([enc1_real, enc1_imag], dim=-1)
        
        dec2_skip = torch.cat([dec2_up, enc1_out], dim=1)
        dec2_out = self.dec2(dec2_skip)  # -> (64, 64, 128)
        
        # Level 1: (64, 64, 128) -> (64, 256, 512)
        dec1_up = self._upsample_complex(dec2_out, scale_factor=4)
        
        # Upsample original input for residual connection
        input_upsampled = self._upsample_complex(original_input, scale_factor=4)  # -> (2, 256, 512)
        
        # Ensure spatial dimensions match for skip connection
        if dec1_up.shape[2:4] != input_upsampled.shape[2:4]:
            # Resize input_upsampled to match dec1_up
            target_h, target_w = dec1_up.shape[2], dec1_up.shape[3]
            input_real = F.interpolate(input_upsampled[..., 0], size=(target_h, target_w), mode='bilinear', align_corners=False)
            input_imag = F.interpolate(input_upsampled[..., 1], size=(target_h, target_w), mode='bilinear', align_corners=False)
            input_upsampled = torch.stack([input_real, input_imag], dim=-1)
        
        dec1_skip = torch.cat([dec1_up, input_upsampled], dim=1)
        dec1_out = self.dec1(dec1_skip)  # -> (2, 256, 512)
        
        # Final convolution
        output = self.final_conv(dec1_out)
        
        # Residual connection: output = decoder_output + upsampled_input
        output = output + input_upsampled
        
        # Convert back to original format
        if input_is_complex:
            return self._convert_from_complex_format(output)
        else:
            # Return real part only if input was real
            return output[..., 0]
    
    def _pool_complex(self, x):
        """Apply magnitude-based pooling to complex tensor"""
        real_part = x[..., 0]
        imag_part = x[..., 1]
        
        # Calculate magnitude for pooling decision
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        
        # Apply max pooling on magnitude to get indices
        pooled_magnitude, indices = F.max_pool2d(magnitude, kernel_size=2, stride=2, return_indices=True)
        
        # Use those indices to pool the complex values
        # Flatten spatial dimensions for advanced indexing
        batch_size, channels, h, w = real_part.shape
        pooled_h, pooled_w = h // 2, w // 2
        
        # Reshape for index selection
        real_flat = real_part.view(batch_size, channels, -1)
        imag_flat = imag_part.view(batch_size, channels, -1)
        indices_flat = indices.view(batch_size, channels, -1)
        
        # Gather values using indices
        pooled_real = torch.gather(real_flat, 2, indices_flat).view(batch_size, channels, pooled_h, pooled_w)
        pooled_imag = torch.gather(imag_flat, 2, indices_flat).view(batch_size, channels, pooled_h, pooled_w)
        
        return torch.stack([pooled_real, pooled_imag], dim=-1)
    
    def _upsample_complex(self, x, scale_factor=2):
        """Apply bicubic upsampling to complex tensor with exact size specification"""
        # Calculate exact target dimensions to avoid floating-point rounding issues
        current_h, current_w = x.shape[2], x.shape[3]
        target_h = current_h * scale_factor
        target_w = current_w * scale_factor
        
        # Use explicit size instead of scale_factor for precise dimensions
        real_part = F.interpolate(x[..., 0], size=(target_h, target_w), mode='bicubic', align_corners=False)
        imag_part = F.interpolate(x[..., 1], size=(target_h, target_w), mode='bicubic', align_corners=False)
        return torch.stack([real_part, imag_part], dim=-1)


def create_model():
    """Create and return ComplexUNet model"""
    model = ComplexUNet()
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing ComplexUNet...")
    
    # Create model
    model = create_model()
    
    # Test input (batch_size=1, channels=2, height=64, width=128)
    test_input_real = torch.randn(1, 2, 64, 128)
    test_input_imag = torch.randn(1, 2, 64, 128)
    test_input_complex = torch.complex(test_input_real, test_input_imag)
    
    # Test forward pass
    print(f"Input shape: {test_input_complex.shape}")
    
    with torch.no_grad():
        output = model(test_input_complex)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Verify complex output
    print(f"Output is complex: {torch.is_complex(output)}")
    print(f"Output dtype: {output.dtype}")
    
    print("ComplexUNet test completed successfully!") 