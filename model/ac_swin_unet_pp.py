#!/usr/bin/env python3
"""
Anisotropic Complex Swin U-Net++ (AC-Swin-UNet++)
================================================
Dual-polarimetric SAR 4-channel (VV-Re, VV-Im, VH-Re, VH-Im) super-resolution network.

Key points vs. prototype version:
1. 4-channel real I/O  → 내부 2-channel complex 전환.
2. Shifted-window Swin Transformer(8×4) : 짝수 stage 시 cyclic shift = window/2.
3. Dense skip(U-Net++), Complex SE + Spatial Attention, Complex PixelShuffle upsampling.

Input  : (B,4,H, W) real
Output : (B,4,4H,4W) real (isotropic 4× upsampling in both height and width)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, List

# ================================================================
# Basic complex layers
# ================================================================

class ComplexConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k, s, p):
        super().__init__()
        self.real = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, padding_mode='reflect')
        self.imag = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, padding_mode='reflect')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.real(x.real) - self.imag(x.imag)
        i = self.real(x.imag) + self.imag(x.real)
        return torch.complex(r, i)

class ComplexBN(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.r = nn.BatchNorm2d(c)
        self.i = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor):
        return torch.complex(self.r(x.real), self.i(x.imag))

class ComplexGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        mag = torch.abs(x)
        return x * (F.gelu(mag) / (mag + 1e-8))

# ================================================================
# Anti-aliased downsampling (BlurPool) for complex tensors
# ================================================================

def _create_blur_kernel(size: int = 5) -> torch.Tensor:
    """Create a 2D binomial (approx Gaussian) blur kernel of given odd size."""
    assert size % 2 == 1, "Blur kernel size must be odd"
    if size == 3:
        base = torch.tensor([1., 2., 1.])
    elif size == 5:
        base = torch.tensor([1., 4., 6., 4., 1.])
    elif size == 7:
        base = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    else:
        # Fallback to size 5 pattern for unsupported sizes
        base = torch.tensor([1., 4., 6., 4., 1.])
    kernel_1d = base / base.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

class ComplexBlurPool(nn.Module):
    """
    Depthwise blur + strided subsampling applied to real/imag separately.
    This reduces aliasing before decimation.
    """
    def __init__(self, channels: int, kernel_size: int = 5, stride: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.channels = channels
        self.stride = stride
        kernel = _create_blur_kernel(kernel_size).float()
        # Register as buffer to move with module device/dtype
        self.register_buffer('kernel', kernel[None, None, :, :])  # [1,1,kH,kW]
        self.pad_h = kernel.shape[-2] // 2
        self.pad_w = kernel.shape[-1] // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = F.pad(x.real, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode='replicate')
        imag = F.pad(x.imag, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode='replicate')
        # Depthwise: repeat kernel per channel
        weight = self.kernel.repeat(self.channels, 1, 1, 1)  # [C,1,kH,kW]
        real = F.conv2d(real, weight, stride=self.stride, groups=self.channels)
        imag = F.conv2d(imag, weight, stride=self.stride, groups=self.channels)
        return torch.complex(real, imag)

# ================================================================
# Complex Resize-based upsampling (anti-checkerboard)
# ================================================================

class ComplexResizeUpsample(nn.Module):
    def __init__(self, in_c: int, out_c: int, scale):
        super().__init__()
        if isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = scale
        self.conv = ComplexConv2d(in_c, out_c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = F.interpolate(x.real, scale_factor=self.scale, mode='bilinear', align_corners=False, antialias=True)
        imag = F.interpolate(x.imag, scale_factor=self.scale, mode='bilinear', align_corners=False, antialias=True)
        x_up = torch.complex(real, imag)
        return self.conv(x_up)

# ================================================================
# Attention blocks
# ================================================================

class ComplexSE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        
        # ★★★★★ 핵심 수정 사항 ★★★★★
        # 채널 수가 r보다 작아져 0이 되는 것을 방지합니다.
        squeezed_ch = max(1, c // r)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, squeezed_ch, 1, bias=False), # c//r -> squeezed_ch
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed_ch, c, 1, bias=False), # c//r -> squeezed_ch
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(torch.abs(x))

class ComplexSpatialAttn(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, k, padding=k//2, bias=False)
    def forward(self, x):
        # Create spatial attention map using aggregated channel information
        abs_feat = torch.mean(torch.abs(x), dim=1, keepdim=True)      # [B, 1, H, W]
        real_feat = torch.mean(x.real, dim=1, keepdim=True)          # [B, 1, H, W] 
        imag_feat = torch.mean(x.imag, dim=1, keepdim=True)          # [B, 1, H, W]
        feat = torch.cat([abs_feat, real_feat, imag_feat], 1)        # [B, 3, H, W]
        m = torch.sigmoid(self.conv(feat))
        return x * m

# ================================================================
# Encoder / Decoder blocks
# ================================================================

class EncBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Anti-aliased downsampling: blur + decimate, then stride-1 conv
        self.blur = ComplexBlurPool(in_c, kernel_size=5, stride=(2, 2))
        self.conv = ComplexConv2d(in_c, out_c, (7, 3), (1, 1), (3, 1))
        self.bn   = ComplexBN(out_c)
        self.act  = ComplexGELU()
    def forward(self,x):
        x = self.blur(x)
        return self.act(self.bn(self.conv(x)))

class DecBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=2, skip_ch=None):
        super().__init__()
        self.up   = ComplexResizeUpsample(in_c, out_c, scale)
        # Calculate input channels for first conv: upsampled + skip features
        if skip_ch is None or skip_ch == 0:
            conv_in_ch = out_c  # No skip connection
        else:
            conv_in_ch = out_c + skip_ch  # With skip connection
        self.c1   = ComplexConv2d(conv_in_ch, out_c, 3,1,1)
        self.b1   = ComplexBN(out_c)
        self.c2   = ComplexConv2d(out_c, out_c, 3,1,1)
        self.b2   = ComplexBN(out_c)
        self.act  = ComplexGELU()
        self.se   = ComplexSE(out_c)
        self.sa   = ComplexSpatialAttn()
    def forward(self,x,skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x,skip],1)
        x = self.act(self.b1(self.c1(x)))
        x = self.act(self.b2(self.c2(x)))
        x = self.sa(self.se(x))
        return x

class DecBlockResize(nn.Module):
    """Decoder block using bilinear resize + ComplexConv (anti-checkerboard)."""
    def __init__(self, in_c, out_c, scale=(2, 2), skip_ch=None):
        super().__init__()
        self.scale = scale
        # Calculate input channels for first conv: upsampled + skip features
        if skip_ch is None or skip_ch == 0:
            conv_in_ch = in_c  # After upsample, channels unchanged
        else:
            conv_in_ch = in_c + skip_ch
        self.c1   = ComplexConv2d(conv_in_ch, out_c, 3,1,1)
        self.b1   = ComplexBN(out_c)
        self.c2   = ComplexConv2d(out_c, out_c, 3,1,1)
        self.b2   = ComplexBN(out_c)
        self.act  = ComplexGELU()
        self.se   = ComplexSE(out_c)
        self.sa   = ComplexSpatialAttn()

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        real = F.interpolate(x.real, scale_factor=self.scale, mode='bilinear', align_corners=False, antialias=True)
        imag = F.interpolate(x.imag, scale_factor=self.scale, mode='bilinear', align_corners=False, antialias=True)
        return torch.complex(real, imag)

    def forward(self, x, skip):
        x = self._resize(x)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        x = self.act(self.b1(self.c1(x)))
        x = self.act(self.b2(self.c2(x)))
        x = self.sa(self.se(x))
        return x

# ================================================================
# Swin transformer (complex)
# ================================================================

class ComplexWindowAttn(nn.Module):
    def __init__(self, dim:int, heads:int, win:Tuple[int,int]):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.scale = (dim//heads)**-0.5
        self.win = win
        self.qkv = ComplexConv2d(dim, dim*3, 1,1,0)
        self.proj= ComplexConv2d(dim, dim, 1,1,0)
        # relative position bias (real-valued) – kept simple
        self.rel = nn.Parameter(torch.zeros(heads, (2*win[0]-1)*(2*win[1]-1)))
        idxs = torch.stack(torch.meshgrid(torch.arange(win[0]),torch.arange(win[1]),indexing='ij')).view(2,-1)
        rel = idxs[:, :,None]-idxs[:,None,:]
        rel[0]+=win[0]-1; rel[1]+=win[1]-1
        rel[0]*=2*win[1]-1
        self.register_buffer('rel_idx',(rel[0]+rel[1]).long())
    def forward(self,x):
        B,C,H,W = x.shape
        q,k,v = torch.chunk(self.qkv(x),3,1)
        q = rearrange(q,'b (h d) hh ww -> b h (hh ww) d',h=self.heads)
        k = rearrange(k,'b (h d) hh ww -> b h (hh ww) d',h=self.heads)
        v = rearrange(v,'b (h d) hh ww -> b h (hh ww) d',h=self.heads)
        attn = (q@k.transpose(-2,-1).conj())*self.scale
        bias = self.rel[:,self.rel_idx.view(-1)].view(self.heads,*self.rel_idx.shape)
        attn = attn + bias.unsqueeze(0)
        w = torch.softmax(torch.abs(attn),-1)
        attn = attn*(w/(torch.abs(attn)+1e-8))
        out = attn@v
        out = rearrange(out,'b h (hh ww) d -> b (h d) hh ww',hh=self.win[0],ww=self.win[1])
        return self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self,dim,heads,win=(8,4),stride=(4,2),shift:bool=False):
        super().__init__()
        self.win,self.stride,self.shift=win,stride,shift
        self.attn = ComplexWindowAttn(dim,heads,win)
        self.n1 = ComplexBN(dim); self.n2 = ComplexBN(dim)
        self.mlp= nn.Sequential(ComplexConv2d(dim,dim*4,1,1,0),ComplexGELU(),ComplexConv2d(dim*4,dim,1,1,0))
    def forward(self,x):
        B,C,H,W = x.shape
        if self.shift:
            x = torch.roll(x, shifts=(-self.stride[0]//2, -self.stride[1]//2), dims=(2,3))
        
        # Extract overlapping windows with stride
        # Unfold creates overlapping patches
        x_unfold = F.unfold(x, kernel_size=self.win, stride=self.stride)  # [B, C*win_h*win_w, num_windows]
        num_windows = x_unfold.size(-1)
        
        # Reshape to [B*num_windows, C, win_h, win_w]
        x_windows = x_unfold.view(B, C, self.win[0], self.win[1], num_windows)
        x_windows = x_windows.permute(0, 4, 1, 2, 3).contiguous()  # [B, num_windows, C, win_h, win_w]
        x_windows = x_windows.view(B * num_windows, C, self.win[0], self.win[1])
        
        # Apply attention to each window
        x_attn = self.attn(x_windows)  # [B*num_windows, C, win_h, win_w]
        
        # Reshape back and fold to reconstruct feature map
        x_attn = x_attn.view(B, num_windows, C, self.win[0], self.win[1])
        x_attn = x_attn.permute(0, 2, 3, 4, 1).contiguous()  # [B, C, win_h, win_w, num_windows]
        x_attn = x_attn.view(B, C * self.win[0] * self.win[1], num_windows)
        
        # Fold back to original spatial dimensions
        output_size = (H, W)
        x_ = F.fold(x_attn, output_size=output_size, kernel_size=self.win, stride=self.stride)
        
        # Handle overlapping regions by normalizing
        ones = torch.ones_like(x)
        ones_unfold = F.unfold(ones, kernel_size=self.win, stride=self.stride)
        ones_fold = F.fold(ones_unfold, output_size=output_size, kernel_size=self.win, stride=self.stride)
        x_ = x_ / (ones_fold + 1e-8)  # Normalize overlapping regions
        
        if self.shift:
            x_ = torch.roll(x_, shifts=(self.stride[0]//2, self.stride[1]//2), dims=(2,3))
        
        x = self.n1(x+x_)
        x = self.n2(x+self.mlp(x))
        return x

# ================================================================
# Main Network
# ================================================================

class ACSwinUNetPP(nn.Module):
    def __init__(self, base_dim:int=64, heads:int=4, depth:int=4):
        super().__init__()
        # 4-real → 2-complex helper handled in forward
        self.enc1 = EncBlock(2, base_dim)
        self.enc2 = EncBlock(base_dim, base_dim*2)
        self.enc3 = EncBlock(base_dim*2, base_dim*4)
        blocks: List[nn.Module] = []
        for i in range(depth):
            blocks.append(SwinBlock(base_dim*4, heads, win=(8,4), stride=(4,2), shift=(i%2==1)))
        self.bottleneck = nn.Sequential(*blocks)
        # Three decoder layers to match three encoder layers
        self.dec3 = DecBlock(base_dim*4, base_dim*2, scale=(2,2), skip_ch=base_dim*2)  # 1/8 -> 1/4, skip e2 
        self.dec2 = DecBlock(base_dim*2, base_dim, scale=(2,2), skip_ch=base_dim)      # 1/4 -> 1/2, skip e1
        self.dec1 = DecBlock(base_dim, base_dim//2, scale=(2,2), skip_ch=0)           # 1/2 -> 1/1, no skip
        # Staged upsampling for super-resolution to reduce grid artifacts
        # 1/1 -> 2x resolution (resize-conv)
        self.up1 = DecBlockResize(base_dim//2, base_dim//4, scale=(2,2), skip_ch=0)
        # 2x -> 4x resolution (resize-conv)
        self.up2 = DecBlockResize(base_dim//4, base_dim//8, scale=(2,2), skip_ch=0)
        # Final output layer
        self.final = ComplexConv2d(base_dim//8, 2, 3, 1, 1)
    def _real4_to_c2(self,x:torch.Tensor):
        vv = torch.complex(x[:,0],x[:,1]); vh = torch.complex(x[:,2],x[:,3])
        return torch.stack([vv,vh],1)
    def _c2_to_real4(self,x:torch.Tensor):
        vv,vh = x[:,0],x[:,1]
        return torch.stack([vv.real,vv.imag,vh.real,vh.imag],1)
    def forward(self, x: torch.Tensor):
        c = self._real4_to_c2(x)
        e1 = self.enc1(c)          # 1/2 resolution, base_dim channels
        e2 = self.enc2(e1)         # 1/4 resolution, base_dim*2 channels  
        e3 = self.enc3(e2)         # 1/8 resolution, base_dim*4 channels
        b  = self.bottleneck(e3)   # 1/8 resolution, base_dim*4 channels
        # Decoder with U-Net skip connections - each decoder upsamples and connects to corresponding encoder
        d3 = self.dec3(b, e2)      # 1/4 resolution, skip from e2 (same resolution after upsampling)
        d2 = self.dec2(d3, e1)     # 1/2 resolution, skip from e1 (same resolution after upsampling)  
        d1 = self.dec1(d2, None)   # 1/1 resolution, no skip connection
        # Staged super-resolution upsampling (2x + 2x = 4x total)
        sr1 = self.up1(d1, None)   # 2x resolution
        sr2 = self.up2(sr1, None)  # 4x resolution
        out = self.final(sr2)      # 4x resolution, 2 complex channels
        # Residual connection with scaled output for training stability
        residual_real = F.interpolate(x, size=out.shape[-2:], mode='bilinear', align_corners=False)
        residual = self._real4_to_c2(residual_real)
        out = out * 0.1 + residual
        return self._c2_to_real4(out)


    def count_parameters(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model():
    return ACSwinUNetPP()

if __name__ == "__main__":
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = create_model().to(dev)  # Use float32 instead of half() for complex support
    x = torch.randn(2,4,64,128,device=dev)  # Use float32 instead of float16
    with torch.no_grad():
        y = m(x)
    print('out',y.shape)