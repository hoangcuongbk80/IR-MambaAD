"""
Small Haar DWT / inverse DWT helpers implemented with torch ops. These are not heavily optimized
but are sufficient for prototyping MWFM. Assumes H and W are even.
"""

import torch
import torch.nn.functional as F


def haar_dwt2(x: torch.Tensor):
    """
    Single-level 2D Haar DWT.
    x: (B, C, H, W) with even H,W
    returns: ll, (lh, hl, hh) each (B,C,H//2,W//2)
    """
    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        # pad by 1 if odd
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="reflect")

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, (lh, hl, hh)


def haar_idwt2(ll: torch.Tensor, lh_hl_hh):
    """
    Inverse of single-level haar_dwt2.
    ll: (B,C,Hc,Wc), lh_hl_hh: tuple of (lh,hl,hh)
    returns: (B,C,Hc*2,Wc*2)
    """
    lh, hl, hh = lh_hl_hh
    # make sure shapes align
    B, C, Hc, Wc = ll.shape
    x00 = 0.5 * (ll + lh + hl + hh)
    x01 = 0.5 * (ll - lh + hl - hh)
    x10 = 0.5 * (ll + lh - hl - hh)
    x11 = 0.5 * (ll - lh - hl + hh)
    H = Hc * 2
    W = Wc * 2
    out = ll.new_zeros((B, C, H, W))
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out
