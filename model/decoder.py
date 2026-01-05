"""

Cascaded decoder that applies multiple HPGMambaStage blocks and injects HF residuals.
Produces final reconstruction map (single-channel).
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .hpg_mamba import HPGMambaStage


class Decoder(nn.Module):
    def __init__(self, in_ch: int, num_stages: int = 3, token_dim: int = 256, state_dim: int = 512, K: int = 4):
        """
        in_ch: channel dimension of fused encoder feature map
        num_stages: number of cascaded HPG-Mamba stages
        """
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([HPGMambaStage(in_ch=in_ch, token_dim=token_dim, state_dim=state_dim, K=K) for _ in range(num_stages)])
        # light refinement head and single-channel output
        self.output_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1)
        )

    def forward(self, fused_feat: torch.Tensor, hf_residuals: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        fused_feat: (B, in_ch, H, W)
        hf_residuals: None or list of tensors length num_stages, each (B, in_ch_hf, H, W)
                      If hf_residuals provided but channel dims differ, we project/resample inside stage.
        """
        x = fused_feat
        for i, stage in enumerate(self.stages):
            hf = None
            if hf_residuals is not None:
                # pick corresponding hf residual if available, otherwise use last
                idx = min(i, len(hf_residuals) - 1)
                hf = hf_residuals[idx]
                # if hf channels differ, adapt via 1x1 conv on the fly (register not ideal, but safe fallback)
                if hf.shape[1] != x.shape[1]:
                    # project hf to match x channels
                    proj = nn.Conv2d(hf.shape[1], x.shape[1], kernel_size=1).to(hf.device)
                    nn.init.kaiming_normal_(proj.weight, nonlinearity='relu')
                    hf = proj(hf)
            else:
                # if none provided, use zeros
                hf = torch.zeros_like(x)
            out_stage = stage(x, hf)
            x = x + out_stage  # additive residual fusion
        recon = self.output_head(x)
        return recon
