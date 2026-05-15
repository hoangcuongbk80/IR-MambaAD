"""

Backbone encoder (ResNet-derived) and a Half-FPN fusion module.

- Accepts arbitrary in_channels for the first conv (so it can accept MWFM outputs).
- Returns a fused feature map with out_dim channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List

# compatibility for torchvision older/newer apis
def _load_resnet34(pretrained: bool):
    try:
        # torchvision >= 0.13
        return models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    except Exception:
        return models.resnet34(pretrained=pretrained)


class HalfFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lat_convs = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels])
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feats: List[torch.Tensor]):
        # feats: list of feature maps [c1, c2, c3, c4]
        # project each to out_channels and upsample to the spatial size of the first tensor
        target_size = feats[0].shape[-2:]
        outs = []
        for i, f in enumerate(feats):
            p = self.lat_convs[i](f)
            if p.shape[-2:] != target_size:
                p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            outs.append(p)
        fused = sum(outs)
        return self.smooth(fused)


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 1, out_dim: int = 256, pretrained: bool = True):
        """
        in_ch: channels fed to first conv (e.g. MWFM.out_ch)
        out_dim: final fused feature channels
        """
        super().__init__()
        base = _load_resnet34(pretrained)
        # adapt first conv to in_ch by averaging pre-trained weights if necessary
        conv1 = base.conv1
        if conv1.in_channels != in_ch:
            w = conv1.weight.data
            if conv1.in_channels == 3 and in_ch == 1:
                new_w = w.mean(dim=1, keepdim=True)
            else:
                # repeat weights to match in_ch or initialize new
                if in_ch < conv1.in_channels:
                    new_w = w[:, :in_ch, :, :].clone()
                else:
                    # tile existing weights
                    times = int((in_ch + conv1.in_channels - 1) / conv1.in_channels)
                    new_w = w.repeat(1, times, 1, 1)[:, :in_ch, :, :].clone()
            base.conv1 = nn.Conv2d(in_ch, conv1.out_channels,
                                   kernel_size=conv1.kernel_size,
                                   stride=conv1.stride,
                                   padding=conv1.padding,
                                   bias=(conv1.bias is not None))
            base.conv1.weight.data = new_w
        # keep useful layers
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # out channels 64
        self.layer2 = base.layer2   # out channels 128
        self.layer3 = base.layer3   # out channels 256
        self.layer4 = base.layer4   # out channels 512

        self.hfpn = HalfFPN(in_channels=[64, 128, 256, 512], out_channels=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_ch, H, W)
        returns: fused feature map (B, out_dim, Hf, Wf) where Hf,Wf are resolution of layer1 (after stem and layer1)
        """
        x = self.stem(x)          # downsampled
        c1 = self.layer1(x)       # smallest downsample factor
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        fused = self.hfpn([c1, c2, c3, c4])
        return fused
