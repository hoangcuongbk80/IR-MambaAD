import torch
import torch.nn.functional as F
from typing import Tuple

def reconstruction_map_loss(recon: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-pixel MSE map and scalar loss.
    recon, target: (B,1,H,W)
    returns: (loss_scalar, mse_map (B,H,W))
    """
    mse = F.mse_loss(recon, target, reduction="none")  # (B,1,H,W)
    mse_map = mse.mean(dim=1)  # (B,H,W)
    if reduction == "mean":
        return mse_map.mean(), mse_map
    elif reduction == "sum":
        return mse_map.sum(), mse_map
    else:
        # return per-sample mean as vector
        return mse_map.view(mse_map.shape[0], -1).mean(dim=1), mse_map
