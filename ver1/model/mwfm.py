"""
MWFM: Multi-scale Wavelet Feature Modulation
- two conv branches (small, large)
- 2-level Haar DWT on small branch -> HF subbands
- learnable per-channel soft-shrinkage (threshold predicted by GAP+1x1 conv)
- GateNet produces spatial-channel gains
- upsample HF features and produce multi-scale HF residuals for decoder stages
- produces e_in (feature tensor) to feed the encoder and hf_residuals list for each decoder stage

Design choices:
- out_channels: number of channels returned as e_in (this must match Encoder first-conv in_channels)
- base_ch: internal base width for conv branches and HF processing
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import haar_dwt2

class SoftShrink(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        # predict per-channel threshold using GAP -> Conv1x1
        self.thresh = nn.Conv2d(ch, ch, kernel_size=1)
        nn.init.constant_(self.thresh.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # compute per-channel threshold with spatial avg
        t = F.relu(self.thresh(x.mean(dim=(-2, -1), keepdim=True)))
        return torch.sign(x) * F.relu(torch.abs(x) - t)


class GateNet(nn.Module):
    def __init__(self, ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MWFM(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, out_ch: int = 1, num_decoder_stages: int = 3):
        """
        in_ch: input image channels (usually 1 for thermal)
        base_ch: width for intermediate features
        out_ch: channel count for e_in (should match Encoder in_ch)
        num_decoder_stages: how many hf residuals to emit (coarse->fine)
        """
        super().__init__()
        self.num_decoder_stages = int(num_decoder_stages)
        # convolutional branches
        self.conv_small = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        self.conv_large = nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3)
        # we'll extract HF channels from small branch (6 subbands from two-level DWT)
        hf_ch = base_ch * 6 // base_ch * base_ch  # effectively 6*base_ch? Keep consistent below.
        # Actually we create HF by concatenating 6 subbands: each subband has base_ch channels.
        hf_ch = base_ch * 6
        self.soft = SoftShrink(hf_ch)
        self.gnet = GateNet(hf_ch, hidden=max(32, base_ch))
        # smoothing conv after upsample HF
        self.smooth = nn.Conv2d(hf_ch, base_ch, kernel_size=3, padding=1)
        # final projection to produce e_in of out_ch
        self.proj_ein = nn.Conv2d(base_ch * 3, out_ch, kernel_size=1)
        # produce multi-scale hf residuals for decoder stages: project to stage channels
        self.hf_projs = nn.ModuleList([nn.Conv2d(hf_ch, base_ch, kernel_size=3, padding=1) for _ in range(self.num_decoder_stages)])

        # init
        nn.init.kaiming_normal_(self.conv_small.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_large.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.smooth.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj_ein.weight, nonlinearity="relu")
        for p in self.hf_projs:
            nn.init.kaiming_normal_(p.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: (B, in_ch, H, W)
        returns:
            e_in: (B, out_ch, H, W) to feed encoder
            hf_residuals: list of tensors length num_decoder_stages,
                          each (B, base_ch, H_s, W_s) where spatial sizes can be same as feature map (we upsample to x size)
        """
        B, _, H, W = x.shape
        f_small = F.relu(self.conv_small(x))
        f_large = F.relu(self.conv_large(x))

        # two-level Haar DWT on f_small -> produce HF subbands
        ll1, (lh1, hl1, hh1) = haar_dwt2(f_small)   # each (B, base_ch, H/2, W/2)
        ll2, (lh2, hl2, hh2) = haar_dwt2(ll1)       # each (B, base_ch, H/4, W/4)

        # upsample all HF to same resolution as ll1 (H/2, W/2) then concat, then upsample to full
        # Bring hf_subbands to a common spatial size (H/2, W/2)
        # lh2,hl2,hh2 are H/4 - upsample to H/2
        lh2_up = F.interpolate(lh2, size=lh1.shape[-2:], mode="bilinear", align_corners=False)
        hl2_up = F.interpolate(hl2, size=lh1.shape[-2:], mode="bilinear", align_corners=False)
        hh2_up = F.interpolate(hh2, size=lh1.shape[-2:], mode="bilinear", align_corners=False)

        hf = torch.cat([lh1, hl1, hh1, lh2_up, hl2_up, hh2_up], dim=1)  # (B, base_ch*6, H/2, W/2)
        hf = self.soft(hf)   # denoise via soft shrinkage
        gm = self.gnet(hf)   # gains in (0,1)
        hf = hf * (1.0 + gm)  # amplify

        # upsample HF to full resolution H,W
        hf_up = F.interpolate(hf, size=(H, W), mode="bilinear", align_corners=False)
        hf_up = F.relu(self.smooth(hf_up))  # (B, base_ch, H, W)

        # produce multi-scale HF residuals (one per decoder stage)
        hf_residuals = []
        for p in self.hf_projs:
            r = p(F.interpolate(hf, size=(H, W), mode="bilinear", align_corners=False))
            hf_residuals.append(F.relu(r))

        # merge f_small, f_large, hf_up -> project to out_ch
        merged = torch.cat([f_small, f_large, hf_up], dim=1)  # (B, base_ch*3, H, W)
        e_in = self.proj_ein(merged)
        return e_in, hf_residuals
