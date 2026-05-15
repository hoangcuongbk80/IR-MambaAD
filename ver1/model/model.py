"""
Top-level IR-MambaAD model wiring:
MWFM -> Encoder -> Decoder

- mwfm_out_ch must match encoder in_ch.
- encoder out_dim must match decoder in_ch.
"""

import torch
import torch.nn as nn
from .mwfm import MWFM
from .encoder import Encoder
from .decoder import Decoder
from typing import Optional


class IRMambaAD(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 mwfm_base_ch: int = 32,
                 mwfm_out_ch: int = 1,
                 encoder_out_dim: int = 256,
                 num_decoder_stages: int = 3,
                 token_dim: int = 256,
                 state_dim: int = 512,
                 pretrained_encoder: bool = True):
        """
        Parameters:
        - input_channels: input image channels
        - mwfm_out_ch: channels produced by MWFM and consumed by Encoder
        - encoder_out_dim: fused feature channels output by Encoder (and consumed by Decoder)
        """
        super().__init__()
        self.mwfm = MWFM(in_ch=input_channels, base_ch=mwfm_base_ch, out_ch=mwfm_out_ch, num_decoder_stages=num_decoder_stages)
        self.encoder = Encoder(in_ch=mwfm_out_ch, out_dim=encoder_out_dim, pretrained=pretrained_encoder)
        self.decoder = Decoder(in_ch=encoder_out_dim, num_stages=num_decoder_stages, token_dim=token_dim, state_dim=state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_channels, H, W)
        returns: recon (B, 1, Hf, Wf) where Hf, Wf are spatial dims of decoder output (note: decoder uses encoder fused size)
        """
        e_in, hf_residuals = self.mwfm(x)
        feat = self.encoder(e_in)
        recon = self.decoder(feat, hf_residuals)
        # Optionally upsample recon to input resolution if spatial dims differ
        if recon.shape[-2:] != x.shape[-2:]:
            recon = nn.functional.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon
