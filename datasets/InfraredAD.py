"""
Expected directory layout (defaults):
root/
  images/
    train/
    val/
    test/
  masks/                 # optional
    train/
    val/
    test/
OR
root/
  train/
    images/
    masks/
  val/
  test/

Features:
- Robust reading of 8/16-bit images (PIL / imageio fallback).
- Optional radiometric scaling for 16-bit thermal images.
- Optional per-dataset normalization (mean/std) or automatic compute utility.
- Supports returning (image, mask, label, filename) where mask may be None.
- Optional torchvision transforms or albumentations (if provided).
"""

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union, Dict
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Try using imageio for broader format support; fallback to PIL
try:
    import imageio.v2 as imageio_read
except Exception:
    imageio_read = None


def _read_image(path: Union[str, Path]) -> np.ndarray:
    """Read image as numpy array (HW) or (HWC). Uses imageio if available, otherwise PIL.
    Returns dtype-preserving numpy array.
    """
    path = str(path)
    if imageio_read is not None:
        img = imageio_read.imread(path)
        return np.asarray(img)
    # fallback PIL
    with Image.open(path) as im:
        return np.asarray(im)


def _to_gray_tensor(img_np: np.ndarray,
                    radiometric: bool = False,
                    clip_range: Optional[Tuple[float, float]] = None,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    Convert HxW numpy image to tensor normalized to [0,1] (float32) and shape (1,H,W).
    If radiometric=True and dtype is integer, scale by 2^bitdepth - 1 unless clip_range provided.
    clip_range: tuple(min_val, max_val) before scaling to [0,1]
    """
    arr = img_np.astype(np.float32)
    # If image is multi-channel with >1 channels -> convert to grayscale by taking first channel
    if arr.ndim == 3 and arr.shape[2] > 1:
        # Some thermal datasets store as HxWx3 (duplicated channels). We'll average channels.
        arr = arr.mean(axis=2)

    if clip_range is not None:
        mn, mx = clip_range
        arr = np.clip(arr, mn, mx)
        arr = (arr - mn) / max((mx - mn), eps)
    elif radiometric:
        # scale depending on bit depth
        if np.issubdtype(img_np.dtype, np.integer):
            maxv = float(np.iinfo(img_np.dtype).max)
            arr = np.clip(arr, 0.0, maxv) / maxv
        else:
            # float images assumed already scaled
            arr = (arr - arr.min()) / max((arr.max() - arr.min()), eps)
    else:
        # fallback: scale based on min/max to [0,1]
        arr = (arr - arr.min()) / max((arr.max() - arr.min()), eps)

    tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
    return tensor


def _read_mask(path: Union[str, Path]) -> torch.Tensor:
    """Read mask image and return binary torch.Tensor of shape (1, H, W) with values {0,1}."""
    arr = _read_image(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8)
    return torch.from_numpy(arr).unsqueeze(0).float()


class InfraredAD(Dataset):
    """
    Generic dataset for infrared anomaly detection.

    Parameters
    ----------
    root : str or Path
        Root directory of dataset.
    split : str
        One of {"train","val","test"}. Default "train".
    image_dir : Optional[str]
        Relative folder containing images under root (default "images").
    mask_dir : Optional[str]
        Relative folder containing masks under root (default "masks"). If None, masks are optional.
    radiometric : bool
        If True, treat integer images as radiometric 16-bit and scale accordingly.
    clip_range : Optional[tuple]
        If provided, clip raw image values to (min,max) before scaling to [0,1].
    transform : Optional[Callable]
        Transform applied to image tensors (expects torch Tensor). If transform expects PIL/numpy,
        provide a wrapper. By default, simple normalization to 0 mean / 1 std is not applied here.
    return_filename : bool
        If True, dataset __getitem__ returns filename as well.
    examples_list : Optional[Iterable[str]]
        Optional list of image paths (absolute or relative to root) to use instead of scanning directory.
    """

    def __init__(self,
                 root: Union[str, Path],
                 split: str = "train",
                 image_dir: str = "images",
                 mask_dir: Optional[str] = "masks",
                 radiometric: bool = True,
                 clip_range: Optional[Tuple[float, float]] = None,
                 transform: Optional[Callable] = None,
                 return_filename: bool = False,
                 examples_list: Optional[Iterable[str]] = None):
        self.root = Path(root)
        self.split = split
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.radiometric = radiometric
        self.clip_range = clip_range
        self.transform = transform
        self.return_filename = return_filename

        # Resolve paths
        # First try root/images/{split}, otherwise root/{split}/images, otherwise root/{split}
        candidates = [
            self.root / image_dir / split,
            self.root / split / image_dir,
            self.root / split
        ]
        self.images_path = None
        for c in candidates:
            if c.exists():
                self.images_path = c
                break
        if self.images_path is None:
            raise FileNotFoundError(f"Cannot find images folder for split '{split}' in {root}.")

        # mask folder
        self.masks_path = None
        if mask_dir is not None:
            candidates_m = [
                self.root / mask_dir / split,
                self.root / split / mask_dir,
                self.root / split
            ]
            for c in candidates_m:
                if c.exists():
                    self.masks_path = c
                    break
            # if mask_dir provided but not found we will still allow None and warn later

        # Build list of image files
        if examples_list:
            # allow either absolute or relative paths
            self.image_files = [Path(p) if Path(p).is_absolute() else self.images_path / p for p in examples_list]
        else:
            exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
            files = []
            for e in exts:
                files.extend(sorted(self.images_path.rglob(e)))
            if len(files) == 0:
                raise RuntimeError(f"No image files found under {self.images_path}.")
            self.image_files = files

        # If masks exist, check mapping by filename
        if self.masks_path is not None:
            # map image filename -> mask path if exists
            self.mask_map = {}
            for img_path in self.image_files:
                mask_candidate = self.masks_path / img_path.name
                if mask_candidate.exists():
                    self.mask_map[str(img_path.name)] = mask_candidate
            # note: not required that all images have masks (e.g., only anomalies annotated)
        else:
            self.mask_map = {}

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        img_np = _read_image(img_path)
        img_t = _to_gray_tensor(img_np, radiometric=self.radiometric, clip_range=self.clip_range)

        mask_t = None
        filename = img_path.name
        if filename in self.mask_map:
            mask_t = _read_mask(self.mask_map[filename])
            # ensure same spatial size
            if mask_t.shape[1:] != img_t.shape[1:]:
                # resize mask to image size using nearest
                mask_pil = Image.fromarray((mask_t.squeeze(0).numpy().astype(np.uint8) * 255))
                mask_pil = mask_pil.resize((img_t.shape[2], img_t.shape[1]), resample=Image.NEAREST)
                mask_t = torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float() / 255.0

        sample: Dict[str, Union[torch.Tensor, str, int, None]] = {"image": img_t, "mask": mask_t, "label": 1 if mask_t is not None and mask_t.any() else 0, "filename": filename}

        if self.transform is not None:
            # Typical torchvision transforms expect PIL images or tensors; user should provide transform compatible
            sample_update = self.transform(sample) if callable(self.transform) else sample
            if isinstance(sample_update, dict):
                sample = sample_update
            else:
                # assume transform returns only image tensor
                sample["image"] = sample_update

        if self.return_filename:
            return sample["image"], sample["mask"], sample["label"], sample["filename"]
        return sample["image"], sample["mask"], sample["label"]


# Example usage:
# ds = InfraredAD("/data/thermals", split="train", radiometric=True, transform=None)
# img, mask, label = ds[0]
