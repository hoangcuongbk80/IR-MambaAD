"""
Small helpers for unit-testing and sanity checks.
"""

import torch

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sanity_forward_test(model, device="cpu"):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, getattr(model, "input_channels", 1), 256, 256, device=device)
        y = model(x)
    return x.shape, y.shape
