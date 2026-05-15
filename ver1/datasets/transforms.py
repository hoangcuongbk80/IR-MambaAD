"""

Transforms adapted to dict-style samples used by the dataset classes in IR-MambaAD.

Main classes:
- DictCompose: compose transforms that accept/return dict samples.
- JointRandomCrop, JointRandomHorizontalFlip: geometric transforms applied jointly to
  thermal/rgb and masks to keep alignment.
- AddGaussianNoise: additive noise for thermal channel.
- PhotometricJitter: brightness/contrast/saturation jitter applied to RGB/aux only.
- ToNormalize: normalize tensors with provided mean/std.
- ToTensorIfNeeded: ensure values are torch.Tensor in [0,1].

Usage:
    from datasets.transforms import DictCompose, JointRandomCrop, AddGaussianNoise, PhotometricJitter
    tr = DictCompose([
        JointRandomCrop((256,256)),
        JointRandomHorizontalFlip(0.5),
        AddGaussianNoise(0.0, 0.01),
        PhotometricJitter(0.4,0.4,0.2),
        ToNormalize(mean_thermal, std_thermal, mean_rgb, std_rgb)
    ])
    sample = tr(sample_dict)
"""

from typing import Callable, Dict, List, Tuple, Optional, Any
import random
import math

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T


def _ensure_tensor(img):
    """Ensure sample image is a torch.FloatTensor in [0,1] with shape (C,H,W)."""
    if isinstance(img, torch.Tensor):
        return img.float()
    arr = np.asarray(img)
    if arr.ndim == 2:  # H,W -> 1,H,W
        arr = arr.astype("float32") / 255.0 if arr.max() > 1.0 else arr.astype("float32")
        t = torch.from_numpy(arr).unsqueeze(0).float()
        return t
    if arr.ndim == 3:  # H,W,C -> C,H,W
        arr = arr.astype("float32") / 255.0 if arr.max() > 1.0 else arr.astype("float32")
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        return t
    raise ValueError("Unsupported image type for conversion to tensor.")


class DictCompose:
    """Compose transforms that accept and return dict-like sample objects."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        s = sample
        for t in self.transforms:
            s = t(s)
        return s


class ToTensorIfNeeded:
    """Ensure thermal/rgb and mask are tensors in expected shapes."""

    def __init__(self, thermal_key: str = "image", aux_key: str = "aux", mask_key: str = "mask"):
        self.thermal_key = thermal_key
        self.aux_key = aux_key
        self.mask_key = mask_key

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.thermal_key in sample and sample[self.thermal_key] is not None:
            sample[self.thermal_key] = _ensure_tensor(sample[self.thermal_key])
        if self.aux_key in sample and sample[self.aux_key] is not None:
            sample[self.aux_key] = _ensure_tensor(sample[self.aux_key])
        if self.mask_key in sample and sample[self.mask_key] is not None:
            # mask should be 1,H,W
            m = sample[self.mask_key]
            if not isinstance(m, torch.Tensor):
                m = torch.from_numpy(np.asarray(m)).float()
            if m.ndim == 2:
                m = m.unsqueeze(0)
            sample[self.mask_key] = (m > 0).float()
        return sample


class JointRandomCrop:
    """Random crop that applies to thermal, aux and mask jointly.

    size: (H, W)
    keys: which keys in the sample to crop (default: 'image','aux','mask')
    """

    def __init__(self, size: Tuple[int, int], keys: Tuple[str, ...] = ("image", "aux", "mask")):
        self.size = size
        self.keys = keys

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        H, W = None, None
        for k in self.keys:
            if k in sample and sample[k] is not None:
                data = sample[k]
                if isinstance(data, torch.Tensor):
                    H, W = data.shape[-2], data.shape[-1]
                else:
                    arr = np.asarray(data)
                    H, W = arr.shape[0], arr.shape[1]
                break
        if H is None:
            return sample

        th, tw = self.size
        if H == th and W == tw:
            return sample
        if H < th or W < tw:
            # pad if smaller
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            # pad at bottom/right with zeros
            def _pad(x):
                if isinstance(x, torch.Tensor):
                    c = x.shape[0]
                    t = torch.zeros((c, H + pad_h, W + pad_w), dtype=x.dtype)
                    t[:, :H, :W] = x
                    return t
                arr = np.asarray(x)
                if arr.ndim == 2:
                    out = np.zeros((H + pad_h, W + pad_w), dtype=arr.dtype)
                    out[:H, :W] = arr
                    return out
                else:
                    # HWC
                    out = np.zeros((H + pad_h, W + pad_w, arr.shape[2]), dtype=arr.dtype)
                    out[:H, :W, :] = arr
                    return out
            for k in self.keys:
                if k in sample and sample[k] is not None:
                    sample[k] = _pad(sample[k])
            H, W = H + pad_h, W + pad_w

        i = random.randint(0, H - th)
        j = random.randint(0, W - tw)
        for k in self.keys:
            if k not in sample or sample[k] is None:
                continue
            x = sample[k]
            if isinstance(x, torch.Tensor):
                sample[k] = x[..., i:i + th, j:j + tw]
            else:
                arr = np.asarray(x)
                if arr.ndim == 2:
                    sample[k] = arr[i:i + th, j:j + tw]
                else:
                    sample[k] = arr[i:i + th, j:j + tw, ...]
        return sample


class JointRandomHorizontalFlip:
    """Flip thermal, aux and mask jointly with probability p."""

    def __init__(self, p: float = 0.5, keys: Tuple[str, ...] = ("image", "aux", "mask")):
        self.p = p
        self.keys = keys

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            for k in self.keys:
                if k not in sample or sample[k] is None:
                    continue
                x = sample[k]
                if isinstance(x, torch.Tensor):
                    sample[k] = torch.flip(x, dims=[-1])
                else:
                    arr = np.asarray(x)
                    if arr.ndim == 2:
                        sample[k] = arr[:, ::-1]
                    else:
                        sample[k] = arr[:, ::-1, ...]
        return sample


class AddGaussianNoise:
    """Add gaussian noise to thermal channel (tensor expected in [0,1])."""

    def __init__(self, mean: float = 0.0, std: float = 0.01, key: str = "image"):
        self.mean = mean
        self.std = std
        self.key = key

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.key in sample and sample[self.key] is not None:
            x = sample[self.key]
            if not isinstance(x, torch.Tensor):
                x = _ensure_tensor(x)
            noise = torch.randn_like(x) * self.std + self.mean
            x = x + noise
            x = x.clamp(0.0, 1.0)
            sample[self.key] = x
        return sample


class PhotometricJitter:
    """Apply ColorJitter-like photometric changes only to aux (RGB) images.

    args are same as torchvision.transforms.ColorJitter
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, key: str = "aux"):
        self.key = key
        self.transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.key in sample and sample[self.key] is not None:
            x = sample[self.key]
            if isinstance(x, torch.Tensor):
                # convert to PIL for torchvision jitter (it's fine)
                x_pil = F.to_pil_image(x)
                x_pil = self.transform(x_pil)
                sample[self.key] = F.to_tensor(x_pil)
            else:
                # assume HWC numpy array
                x_pil = Image.fromarray((x * 255).astype("uint8")) if x.max() <= 1.0 else Image.fromarray(x.astype("uint8"))
                x_pil = self.transform(x_pil)
                sample[self.key] = F.to_tensor(x_pil)
        return sample


class RandomCutout:
    """Apply random cutout (erase block) on thermal/aux channels."""

    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), key: str = "image"):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.key = key

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            return sample
        if self.key not in sample or sample[self.key] is None:
            return sample
        x = sample[self.key]
        if not isinstance(x, torch.Tensor):
            x = _ensure_tensor(x)
        C, H, W = x.shape
        area = H * W

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if h < H and w < W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                x[:, top:top + h, left:left + w] = 0.0
                sample[self.key] = x
                return sample
        # fallback: small square
        sz = int(min(H, W) * 0.1)
        top = random.randint(0, H - sz)
        left = random.randint(0, W - sz)
        x[:, top:top + sz, left:left + sz] = 0.0
        sample[self.key] = x
        return sample


class ToNormalize:
    """Normalize thermal and aux tensors with provided mean/std.

    mean_thermal: scalar
    std_thermal: scalar
    mean_aux: tuple of 3
    std_aux: tuple of 3
    """

    def __init__(self,
                 mean_thermal: Optional[float] = None,
                 std_thermal: Optional[float] = None,
                 mean_aux: Optional[Tuple[float, float, float]] = None,
                 std_aux: Optional[Tuple[float, float, float]] = None,
                 thermal_key: str = "image",
                 aux_key: str = "aux"):
        self.mean_thermal = mean_thermal
        self.std_thermal = std_thermal
        self.mean_aux = mean_aux
        self.std_aux = std_aux
        self.thermal_key = thermal_key
        self.aux_key = aux_key

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.thermal_key in sample and sample[self.thermal_key] is not None:
            t = sample[self.thermal_key]
            if self.mean_thermal is not None and self.std_thermal is not None:
                t = (t - self.mean_thermal) / (self.std_thermal + 1e-9)
            sample[self.thermal_key] = t
        if self.aux_key in sample and sample[self.aux_key] is not None:
            a = sample[self.aux_key]
            if self.mean_aux is not None and self.std_aux is not None:
                mean = torch.tensor(self.mean_aux).view(-1, 1, 1)
                std = torch.tensor(self.std_aux).view(-1, 1, 1)
                a = (a - mean) / (std + 1e-9)
            sample[self.aux_key] = a
        return sample
