"""
HPG-Mamba stage:
- projects fused feature map and HF map to token_dim
- serializes tokens to a sequence (row-major)
- predicts convex weights alpha from HF tokens (per time-step)
- runs SelectiveSSM and maps output tokens back to spatial map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .mamba_ssm import SelectiveSSM


class HPGMambaStage(nn.Module):
    def __init__(self, in_ch: int, token_dim: int = 256, state_dim: int = 512, K: int = 4):
        """
        in_ch: channels of input feature map (and hf map)
        token_dim: token embedding dimension
        state_dim: SSM state size
        K: number of base SSM tuples
        """
        super().__init__()
        self.token_dim = token_dim
        self.proj_f = nn.Conv2d(in_ch, token_dim, kernel_size=1)
        self.proj_hf = nn.Conv2d(in_ch, token_dim, kernel_size=1)
        self.alpha_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, K)
        )
        self.ssm = SelectiveSSM(input_dim=token_dim, state_dim=state_dim, K=K)
        self.out_proj = nn.Conv2d(token_dim, in_ch, kernel_size=1)

    def forward(self, feat_map: torch.Tensor, hf_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: (B, C, H, W)
        hf_map:   (B, C, H, W)
        returns: out_map (B, C, H, W) same spatial size as inputs
        """
        B, C, H, W = feat_map.shape
        # project to token dim
        t_f = self.proj_f(feat_map)   # (B, d, H, W)
        t_hf = self.proj_hf(hf_map)   # (B, d, H, W)

        # serialize to sequences: (B, L, d)
        L = H * W
        t_f_seq = t_f.view(B, self.token_dim, L).permute(0, 2, 1).contiguous()
        t_hf_seq = t_hf.view(B, self.token_dim, L).permute(0, 2, 1).contiguous()

        # compute alpha logits per token/time-step from hf tokens
        # use linear head on each token
        alpha_logits = self.alpha_head(t_hf_seq)  # (B, L, K)
        alpha = F.softmax(alpha_logits, dim=-1)

        # run SelectiveSSM -> outputs y_seq (B, L, d)
        y_seq, _ = self.ssm(t_f_seq, alpha)

        # reshape back to (B, d, H, W)
        y_map = y_seq.permute(0, 2, 1).contiguous().view(B, self.token_dim, H, W)

        # project to in_ch and return
        out = self.out_proj(y_map)
        return out
