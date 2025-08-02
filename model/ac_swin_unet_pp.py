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
Output : (B,4,H,4·W) real (anisotropic 4× only in width, LR height 유지)
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
        self.real = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.imag = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)

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
# Attention blocks
# ================================================================

class ComplexSE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(c//r, c, 1, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(torch.abs(x))

class ComplexSpatialAttn(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, k, padding=k//2, bias=False)
    def forward(self, x):
        feat = torch.cat([torch.abs(x), x.real, x.imag], 1)
        m = torch.sigmoid(self.conv(feat))
        return x * m

# ================================================================
# Encoder / Decoder blocks
# ================================================================

class EncBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ComplexConv2d(in_c, out_c, (3,7), (1,2), (1,3))
        self.bn   = ComplexBN(out_c)
        self.act  = ComplexGELU()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class DecBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.up   = ComplexPixelShuffle(in_c, out_c, scale)
        self.c1   = ComplexConv2d(out_c*2, out_c, 3,1,1)
        self.b1   = ComplexBN(out_c)
        self.c2   = ComplexConv2d(out_c, out_c, 3,1,1)
        self.b2   = ComplexBN(out_c)
        self.act  = ComplexGELU()
        self.se   = ComplexSE(out_c)
        self.sa   = ComplexSpatialAttn()
    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat([x,skip],1)
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
    def __init__(self,dim,heads,win=(8,4),shift:bool=False):
        super().__init__()
        self.win,self.shift=win,shift
        self.attn = ComplexWindowAttn(dim,heads,win)
        self.n1 = ComplexBN(dim); self.n2 = ComplexBN(dim)
        self.mlp= nn.Sequential(ComplexConv2d(dim,dim*4,1,1,0),ComplexGELU(),ComplexConv2d(dim*4,dim,1,1,0))
    def forward(self,x):
        B,C,H,W = x.shape
        if self.shift:
            x = torch.roll(x, shifts=(-self.win[0]//2, -self.win[1]//2), dims=(2,3))
        # partition
        assert H%self.win[0]==0 and W%self.win[1]==0, "Input size must be multiple of window"
        x_ = rearrange(x,'b c (h p1) (w p2)-> (b h w) c p1 p2',p1=self.win[0],p2=self.win[1])
        x_ = self.attn(x_)
        x_ = rearrange(x_,'(b h w) c p1 p2 -> b c (h p1) (w p2)',h=H//self.win[0],w=W//self.win[1])
        if self.shift:
            x_ = torch.roll(x_, shifts=(self.win[0]//2, self.win[1]//2), dims=(2,3))
        x = self.n1(x+x_)
        x = self.n2(x+self.mlp(x))
        return x

# ================================================================
# Main Network
# ================================================================

class ACSwinUNetPP(nn.Module):
    def __init__(self, base_dim:int=64, heads:int=4, depth:int=2):
        super().__init__()
        # 4-real → 2-complex helper handled in forward
        self.enc1 = EncBlock(2, base_dim)
        self.enc2 = EncBlock(base_dim, base_dim*2)
        self.enc3 = EncBlock(base_dim*2, base_dim*4)
        blocks: List[nn.Module] = []
        for i in range(depth):
            blocks.append(SwinBlock(base_dim*4, heads, win=(8,4), shift=(i%2==1)))
        self.bottleneck = nn.Sequential(*blocks)
        self.dec2 = DecBlock(base_dim*4, base_dim*2)
        self.dec1 = DecBlock(base_dim*2, base_dim)
        self.final = nn.Sequential(
            ComplexConv2d(base_dim, base_dim//2,3,1,1), ComplexGELU(),
            ComplexConv2d(base_dim//2,2,1,1,0)
        )
    def _real4_to_c2(self,x:torch.Tensor):
        vv = torch.complex(x[:,0],x[:,1]); vh = torch.complex(x[:,2],x[:,3])
        return torch.stack([vv,vh],1)
    def _c2_to_real4(self,x:torch.Tensor):
        vv,vh = x[:,0],x[:,1]
        return torch.stack([vv.real,vv.imag,vh.real,vh.imag],1)
    def forward(self, x: torch.Tensor):
        c = self._real4_to_c2(x)
        e1 = self.enc1(c)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        d2 = self.dec2(b, e2)
        d1 = self.dec1(d2, e1)
        out = self.final(d1)
        out = out + F.interpolate(c, size=out.shape[-2:], mode='bilinear', align_corners=False)
        return self._c2_to_real4(out)


def create_model():
    return ACSwinUNetPP()

if __name__ == "__main__":
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = create_model().to(dev).half()
    x = torch.randn(2,4,64,128,device=dev,dtype=torch.float16)
    with torch.no_grad():
        y = m(x)
    print('out',y.shape)
