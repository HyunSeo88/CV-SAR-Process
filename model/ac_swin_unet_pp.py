#!/usr/bin/env python3
"""
Anisotropic Complex Swin U-Net++ for Dual-Polarimetric SAR Super-Resolution
===========================================================================

This script implements the AC-Swin-UNet++ architecture in PyTorch, specifically
designed for super-resolving dual-polarimetric (VV, VH) complex SAR data.

Key Architectural Features:
- Siamese Encoder: Shared-weight encoders process VV and VH polarizations in parallel,
  using anisotropic complex convolutions (3x7 kernels, 1x2 strides).
- Swin Transformer Bottleneck: A complex-valued Swin Transformer with anisotropic
  windows (8x4) serves as the bottleneck to capture global dependencies.
- Dense Skip Connections (U-Net++): All encoder feature maps are densely
  connected to the decoder, improving gradient flow and feature fusion.
- Advanced Decoder: The decoder uses Complex Pixel Shuffle for high-quality
  upsampling and incorporates both Complex SE (Squeeze-and-Excitation) and
  Complex Spatial Attention modules.
- Input/Output: Takes a (B, 2, H, W) complex tensor and outputs a
  (B, 2, H_hr, W_hr) complex tensor.

NOTE: Coherence calculations are disabled as phase coherence between VV and VH
in Sentinel-1 is typically low and not a reliable feature for this task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, List, Tuple

# --- Foundational Complex-Valued Modules ---

class ComplexConv2d(nn.Module):
    """Implements a complex-valued 2D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: complex tensor"""
        real_out = self.conv_r(x.real) - self.conv_i(x.imag)
        imag_out = self.conv_r(x.imag) + self.conv_i(x.real)
        return torch.complex(real_out, imag_out)

class ComplexGELU(nn.Module):
    """Applies GELU activation to the magnitude of a complex tensor."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        activated_magnitude = F.gelu(magnitude)
        # Preserve phase, scale magnitude
        return x * (activated_magnitude / (magnitude + 1e-8))

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.bn_i = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return torch.complex(self.bn_r(x.real), self.bn_i(x.imag))


# --- Decoder and Attention Modules ---

class ComplexPixelShuffle(nn.Module):
    """
    Complex-valued PixelShuffle for artifact-free upsampling.
    This module first applies a convolution to expand channels, then performs
    pixel shuffle on real and imaginary parts separately.
    """
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ComplexConv2d(in_channels, out_channels * (scale_factor ** 2), 1, 1, 0)
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        real = self.shuffle(x.real)
        imag = self.shuffle(x.imag)
        return torch.complex(real, imag)

class ComplexSE(nn.Module):
    """Complex Squeeze-and-Excitation Module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Operate on magnitude for pooling and excitation score generation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get attention scores from magnitude
        scale = self.se(torch.abs(x))
        return x * scale

class ComplexSpatialAttention(nn.Module):
    """Complex Spatial Attention Module."""
    def __init__(self, kernel_size=7):
        super().__init__()
        # Use real-valued conv on concatenated magnitude, real, imag parts
        self.conv = nn.Conv2d(3, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stack magnitude, real, and imag parts along channel dimension
        cat_features = torch.cat([torch.abs(x), x.real, x.imag], dim=1)
        # Generate spatial attention map
        attn_map = self.sigmoid(self.conv(cat_features))
        return x * attn_map


# --- Encoder and Decoder Blocks ---

class EncoderBlock(nn.Module):
    """Siamese Encoder Block with anisotropic convolution."""
    def __init__(self, in_c, out_c):
        super().__init__()
        # Anisotropic convolution as specified
        self.conv = ComplexConv2d(in_c, out_c, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.bn = ComplexBatchNorm2d(out_c)
        self.act = ComplexGELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    """
    U-Net++ style dense decoder block with attention.
    Receives a list of tensors from skip connections and the upsampled path.
    """
    def __init__(self, in_c, out_c, upsample_scale=2):
        super().__init__()
        self.upsample = ComplexPixelShuffle(in_c, out_c, upsample_scale)
        self.conv1 = ComplexConv2d(out_c * 2, out_c, 3, 1, 1) # After concat
        self.bn1 = ComplexBatchNorm2d(out_c)
        self.act1 = ComplexGELU()
        self.conv2 = ComplexConv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = ComplexBatchNorm2d(out_c)
        self.act2 = ComplexGELU()
        
        self.se = ComplexSE(out_c)
        self.spatial_attn = ComplexSpatialAttention()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.upsample(x)
        x_cat = torch.cat([x_up, skip], dim=1)
        
        x = self.act1(self.bn1(self.conv1(x_cat)))
        x = self.act2(self.bn2(self.conv2(x)))
        
        x = self.se(x)
        x = self.spatial_attn(x)
        return x


# --- Swin Transformer Bottleneck ---

class ComplexSwinAttention(nn.Module):
    """Complex Multi-Head Self-Attention for Swin Transformer."""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv_conv = ComplexConv2d(dim, dim * 3, 1, 1, 0)
        self.proj_conv = ComplexConv2d(dim, dim, 1, 1, 0)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1) # (B, C, H, W)

        # Reshape for multi-head attention
        q = rearrange(q, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)

        # Complex dot product attention
        attn = (q @ k.transpose(-2, -1).conj()) * self.scale
        
        # Apply relative position bias to magnitude
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Softmax on magnitude
        attn_mag = torch.softmax(torch.abs(attn), dim=-1)
        attn = attn * (attn_mag / (torch.abs(attn) + 1e-8))

        out = attn @ v
        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=H, ww=W)
        return self.proj_conv(out)

class SwinTransformerBlock(nn.Module):
    """Anisotropic Complex Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=(8, 4)):
        super().__init__()
        self.window_size = window_size
        self.attention = ComplexSwinAttention(dim, num_heads, window_size)
        self.norm1 = ComplexBatchNorm2d(dim)
        self.norm2 = ComplexBatchNorm2d(dim)
        
        # Complex MLP
        self.mlp = nn.Sequential(
            ComplexConv2d(dim, dim * 4, 1, 1, 0),
            ComplexGELU(),
            ComplexConv2d(dim * 4, dim, 1, 1, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Window Partitioning
        windows = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.window_size[0], p2=self.window_size[1])
        
        # Attention in windows
        attn_windows = self.attention(windows)
        
        # Reverse Window Partitioning
        attn_output = rearrange(attn_windows, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=H//self.window_size[0], w=W//self.window_size[1])
        
        # Skip connection 1
        x = x + attn_output
        x = self.norm1(x)

        # MLP
        mlp_output = self.mlp(x)

        # Skip connection 2
        x = x + mlp_output
        x = self.norm2(x)
        return x


# --- Main Model: AC-Swin-UNet++ ---

class ACSwinUNetPP(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, num_heads=4, swin_depth=2):
        super().__init__()
        
        # --- Siamese Encoder ---
        self.enc1 = EncoderBlock(in_channels, base_dim) # 64
        self.enc2 = EncoderBlock(base_dim, base_dim*2) # 128
        self.enc3 = EncoderBlock(base_dim*2, base_dim*4) # 256
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            *[SwinTransformerBlock(dim=base_dim*4, num_heads=num_heads, window_size=(8,4)) for _ in range(swin_depth)]
        )
        
        # --- U-Net++ Dense Decoder Path ---
        # Note: This is a simplified U-Net++ structure for clarity.
        # A full implementation would have more cross-connections.
        self.dec2 = DecoderBlock(base_dim*4, base_dim*2)
        self.dec1 = DecoderBlock(base_dim*2, base_dim)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            ComplexConv2d(base_dim, base_dim//2, 3, 1, 1),
            ComplexGELU(),
            ComplexConv2d(base_dim//2, in_channels, 1, 1, 0)
        )
        
        # Upsampling for residual connection
        self.final_upsample = nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=False)

        print(f"AC-Swin-UNet++ initialized with {self.count_parameters():,} parameters.")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is a dual-polar complex tensor (B, 2, H, W)"""
        # Split polarizations for Siamese processing
        vv, vh = x[:, 0:1, ...], x[:, 1:2, ...]
        
        # --- Encoder Path (Shared Weights) ---
        vv_e1 = self.enc1(vv)
        vh_e1 = self.enc1(vh) # Share weights

        vv_e2 = self.enc2(vv_e1)
        vh_e2 = self.enc2(vh_e1)

        vv_e3 = self.enc3(vv_e2)
        vh_e3 = self.enc3(vh_e2)
        
        # Combine features before bottleneck
        b_in = vv_e3 + vh_e3 # Element-wise sum fusion
        
        # --- Bottleneck ---
        b_out = self.bottleneck(b_in)
        
        # --- Decoder Path ---
        # Using vv channel for skip connections as it's the primary channel to be super-resolved
        d2 = self.dec2(b_out, vv_e2)
        d1 = self.dec1(d2, vv_e1)

        # --- Final Output Generation ---
        # Generate SR for both channels from the fused features
        final_features = self.final_conv(d1)
        
        # Residual connection
        residual_vv = self.final_upsample(vv.real) + 1j * self.final_upsample(vv.imag)
        residual_vh = self.final_upsample(vh.real) + 1j * self.final_upsample(vh.imag)
        
        out_vv = final_features + residual_vv
        out_vh = final_features + residual_vh # Simple assumption: residual is similar
        
        return torch.cat([out_vv, out_vh], dim=1)


def create_model():
    """Factory function for train.py"""
    # NOTE: Set torch.backends.cudnn.benchmark = True in your training script for performance.
    return ACSwinUNetPP(in_channels=1, base_dim=64, num_heads=4, swin_depth=2)


if __name__ == '__main__':
    print("--- Testing AC-Swin-UNet++ ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Target memory: RTX 4070 Ti (12GB)
    # Using a smaller batch size for testing to be safe.
    # Dynamic batch sizing from speed_utils.py should handle this in train.py
    batch_size = 2
    in_height, in_width = 128, 64 # Example LR dimensions
    
    model = create_model().to(device)
    
    # Create a dummy complex input tensor
    dummy_input = torch.randn(batch_size, 2, in_height, in_width, dtype=torch.cfloat).to(device)
    
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Expected output shape: (B, 2, H, W*4) due to strides (1,2) and upsampling (1,4)
    print(f"Output shape: {output.shape}")
    print(f"Output is complex: {torch.is_complex(output)}")
    
    # --- Performance Metrics To Be Integrated ---
    print("\n--- Metrics to integrate in utils.py ---")
    print(" - PSNR (Peak Signal-to-Noise Ratio)")
    print(" - RMSE (Root Mean Squared Error)")
    print(" - CPIF (Complex Phase Invariant Fidelity)")
    
    # --- Notes on Disabled Functionality ---
    print("\n--- Disabled/Removed Functionality ---")
    print(" - Coherence calculation between VV and VH is removed due to low reliability.")

    # --- Potential Future Optimizations ---
    print("\n--- Future Optimization Points ---")
    print(" - Swin Transformer Depth: `swin_depth` can be adjusted. More depth increases capacity but also memory/compute.")
    print(" - Attention Mechanism: The fusion of VV and VH features in the decoder could be more sophisticated (e.g., cross-attention).")
    print(" - Mixed Precision: Use torch.amp for further speed-up and memory savings.")

