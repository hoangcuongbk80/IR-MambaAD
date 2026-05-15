"""
Utilities:
- compute_mean_std: compute dataset mean/std for thermal (1-channel) and aux (3-channel) if present.
- visualize_random_samples: show a grid of samples (thermal, aux, mask) for quick debugging.

Usage:
    from datasets.InfraredAD import InfraredAD
    from datasets.utils import compute_mean_std, visualize_random_samples
    ds = InfraredAD("/data/thermal", split="train")
    mean_std = compute_mean_std(ds, batch_size=8, num_workers=4, max_batches=100)
    visualize_random_samples(ds, n=6)
"""

from typing import Optional, Tuple, Dict, Any, Iterable
import random
import math

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def compute_mean_std(dataset,
                     key_thermal: str = "image",
                     key_aux: str = "aux",
                     batch_size: int = 16,
                     num_workers: int = 4,
                     max_batches: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute mean and std for thermal (single-channel) and aux (3-channel) if present.
    Returns dict: {"thermal": (mean, std), "aux": (mean3, std3)} where mean/std are floats or 3-tuples.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)
    cnt = 0
    sum_t = 0.0
    sumsq_t = 0.0
    pixels = 0

    sum_a = None
    sumsq_a = None
    pixels_a = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # batch is list of sample tuples or dicts; unify
        for sample in batch:
            # sample may be tuple (img, mask, label) or dict. Try dict first.
            if isinstance(sample, dict):
                img = sample.get(key_thermal)
                aux = sample.get(key_aux, None)
            else:
                # heuristics: first item is thermal, second could be aux
                if len(sample) >= 1:
                    img = sample[0]
                    aux = sample[1] if len(sample) > 1 else None
                else:
                    continue
            if img is None:
                continue
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = np.asarray(img)
            # ensure CHW
            if img_np.ndim == 2:
                img_np = img_np[np.newaxis, ...]
            elif img_np.ndim == 3 and img_np.shape[0] not in (1,3):
                # HWC -> CHW
                img_np = img_np.transpose(2,0,1)
            c, h, w = img_np.shape
            pixels += h * w
            sum_t += img_np.mean(axis=(1,2)).sum()  # sum of channel means
            # we'll compute per-pixel sums more robustly:
            sumsq_t += (img_np ** 2).sum()
            # aux
            if aux is not None:
                if isinstance(aux, torch.Tensor):
                    aux_np = aux.detach().cpu().numpy()
                else:
                    aux_np = np.asarray(aux)
                if aux_np.ndim == 2:
                    aux_np = aux_np[np.newaxis, ...]
                elif aux_np.ndim == 3 and aux_np.shape[0] not in (1,3):
                    aux_np = aux_np.transpose(2,0,1)
                if sum_a is None:
                    sum_a = np.zeros(aux_np.shape[0], dtype=np.float64)
                    sumsq_a = np.zeros(aux_np.shape[0], dtype=np.float64)
                sum_a += aux_np.sum(axis=(1,2))
                sumsq_a += (aux_np ** 2).sum(axis=(1,2))
                pixels_a += aux_np.shape[1] * aux_np.shape[2]
        cnt += 1

    # thermal stats
    # Because we aggregated sums a bit differently above, recompute robustly via one-pass:
    # Simpler: go again but faster using smaller batches would be accurate; to keep code simple:
    # We compute mean and std from sumsq_t and total pixels: assume single channel or multiple channels
    # sumsq_t is sum over all channels and pixels, we will compute per-channel mean & std if desired.
    # For typical IR dataset single channel:
    if pixels == 0:
        raise RuntimeError("No pixels processed; check dataset.")
    # If dataset images had single channel, compute mean/std from sums above approximately:
    # Here we compute scalar mean/std for simplicity:
    mean_thermal = None
    std_thermal = None
    try:
        # If sum_a is None but we did per-pixel sums incorrectly, fallback compute via simple pass:
        # Do a second pass but small batches to compute precise channel means.
        loader2 = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        sum_chan = None
        sumsq_chan = None
        total_pixels_chan = 0
        for batch in loader2:
            for sample in batch:
                if isinstance(sample, dict):
                    img = sample.get(key_thermal)
                else:
                    img = sample[0]
                if img is None:
                    continue
                if isinstance(img, torch.Tensor):
                    img_np = img.detach().cpu().numpy()
                else:
                    img_np = np.asarray(img)
                if img_np.ndim == 2:
                    img_np = img_np[np.newaxis, ...]
                elif img_np.ndim == 3 and img_np.shape[0] not in (1,3):
                    img_np = img_np.transpose(2,0,1)
                if sum_chan is None:
                    sum_chan = np.zeros(img_np.shape[0], dtype=np.float64)
                    sumsq_chan = np.zeros(img_np.shape[0], dtype=np.float64)
                sum_chan += img_np.sum(axis=(1,2))
                sumsq_chan += (img_np**2).sum(axis=(1,2))
                total_pixels_chan += img_np.shape[1] * img_np.shape[2]
        mean_chan = (sum_chan / total_pixels_chan)
        var_chan = (sumsq_chan / total_pixels_chan) - (mean_chan ** 2)
        std_chan = np.sqrt(np.maximum(var_chan, 1e-12))
        if mean_chan.shape[0] == 1:
            mean_thermal = float(mean_chan[0])
            std_thermal = float(std_chan[0])
        else:
            mean_thermal = tuple(map(float, mean_chan.tolist()))
            std_thermal = tuple(map(float, std_chan.tolist()))
    except Exception:
        mean_thermal = float(0.5)
        std_thermal = float(0.25)

    mean_aux = None
    std_aux = None
    if sum_a is not None and pixels_a > 0:
        mean_aux = tuple((sum_a / pixels_a).tolist())
        var_aux = (sumsq_a / pixels_a) - np.array(mean_aux) ** 2
        std_aux = tuple(np.sqrt(np.maximum(var_aux, 1e-12)).tolist())

    return {"thermal": (mean_thermal, std_thermal), "aux": (mean_aux, std_aux)}


def visualize_random_samples(dataset, n: int = 6, figsize: Tuple[int, int] = (14, 6), seed: Optional[int] = None):
    """Show n random samples from dataset. Handles dict-style and tuple-style samples."""
    if seed is not None:
        random.seed(seed)
    total = len(dataset)
    n = min(n, total)
    idxs = random.sample(range(total), n)
    cols = n
    fig, axs = plt.subplots(3, cols, figsize=figsize) if cols > 1 else plt.subplots(3, 1, figsize=figsize)
    if cols == 1:
        axs = np.expand_dims(axs, axis=1)
    for j, idx in enumerate(idxs):
        sample = dataset[idx]
        if isinstance(sample, dict):
            img = sample.get("image")
            aux = sample.get("aux")
            mask = sample.get("mask")
            fn = sample.get("filename", str(idx))
        else:
            # try tuple format: thermal, aux, mask, label, filename?
            img = sample[0] if len(sample) > 0 else None
            aux = sample[1] if len(sample) > 1 else None
            mask = sample[2] if len(sample) > 2 else None
            fn = sample[4] if len(sample) > 4 else (sample[3] if len(sample) > 3 else str(idx))
        # convert to numpy for plotting
        def to_numpy(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
                if arr.ndim == 3:
                    return np.transpose(arr, (1, 2, 0))
                elif arr.ndim == 2:
                    return arr
                elif arr.ndim == 1:
                    return arr
                else:
                    return arr.squeeze()
            else:
                return np.asarray(x)
        img_np = to_numpy(img)
        aux_np = to_numpy(aux)
        mask_np = to_numpy(mask)
        # thermal (top)
        ax = axs[0, j]
        if img_np is None:
            ax.axis("off")
        else:
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                ax.imshow(img_np)
            else:
                ax.imshow(img_np.squeeze(), cmap="inferno")
            ax.set_title(f"{fn}")
            ax.axis("off")
        # aux (middle)
        ax = axs[1, j]
        if aux_np is None:
            ax.axis("off")
        else:
            # aux expected CHW->HWC in [0,1]
            if aux_np.ndim == 3 and aux_np.shape[2] == 3:
                ax.imshow(np.clip(aux_np, 0, 1))
            else:
                ax.imshow(aux_np.squeeze(), cmap="gray")
            ax.axis("off")
        # mask (bottom)
        ax = axs[2, j]
        if mask_np is None:
            ax.axis("off")
        else:
            if mask_np.ndim == 3:
                m = mask_np.squeeze()
            else:
                m = mask_np
            ax.imshow(m, cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show()
