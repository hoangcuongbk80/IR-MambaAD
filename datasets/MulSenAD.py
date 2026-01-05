"""
root/
  thermal/
    train/
    val/
    test/
  rgb/
    train/
    val/
    test/
  masks/ (optional)
    train/
    val/
    test/

The thermal and rgb folders should contain matching filenames (same base name, possibly different ext).
This class pairs corresponding files by stem (filename without extension).
"""

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Reuse helpers from InfraredAD by relative import if module is in same package.
# If running standalone, consider copying helpers; here we import by package-style.
try:
    from .InfraredAD import _read_image, _to_gray_tensor, _read_mask
except Exception:
    # fallback: attempt relative import assuming modules are in same folder on sys.path
    from InfraredAD import _read_image, _to_gray_tensor, _read_mask


def _to_rgb_tensor(img_np: np.ndarray) -> torch.Tensor:
    """
    Convert HxWxC numpy array to torch tensor (3,H,W), normalized to [0,1] float32.
    If image is single channel, duplicate to 3 channels.
    """
    arr = img_np.astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 4:
        # drop alpha if present
        arr = arr[:, :, :3]
    # scale to [0,1] by channel-wise min/max (assumes 8-bit)
    if np.issubdtype(arr.dtype, np.integer):
        maxv = float(np.iinfo(img_np.dtype).max)
        arr = arr / maxv
    else:
        arr = (arr - arr.min()) / max((arr.max() - arr.min()), 1e-6)
    # HWC -> CHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor


class MulSenAD(Dataset):
    """
    Dataset that returns paired samples: thermal image + RGB (or other modality).
    Each item: dict with keys {'thermal', 'rgb', 'mask', 'label', 'filename'} or tuple depending on return_filename.
    """

    def __init__(self,
                 root: Union[str, Path],
                 thermal_dir: str = "thermal",
                 aux_dir: str = "rgb",
                 mask_dir: Optional[str] = "masks",
                 split: str = "train",
                 radiometric: bool = True,
                 transform: Optional[Callable] = None,
                 return_filename: bool = False):
        self.root = Path(root)
        self.thermal_dir = thermal_dir
        self.aux_dir = aux_dir
        self.mask_dir = mask_dir
        self.split = split
        self.radiometric = radiometric
        self.transform = transform
        self.return_filename = return_filename

        # Resolve folders
        t_candidates = [self.root / thermal_dir / split, self.root / split / thermal_dir, self.root / thermal_dir]
        a_candidates = [self.root / aux_dir / split, self.root / split / aux_dir, self.root / aux_dir]
        self.thermal_path = next((p for p in t_candidates if p.exists()), None)
        self.aux_path = next((p for p in a_candidates if p.exists()), None)
        if self.thermal_path is None or self.aux_path is None:
            raise FileNotFoundError(f"Could not locate thermal or aux folders under {root} for split {split}.")

        # list files and pair by stem
        def all_files(p: Path) -> List[Path]:
            exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
            files = []
            for e in exts:
                files.extend(sorted(p.rglob(e)))
            return files

        thermal_files = all_files(self.thermal_path)
        aux_files = all_files(self.aux_path)

        aux_map = {}
        for f in aux_files:
            aux_map.setdefault(f.stem, []).append(f)

        paired = []
        for t in thermal_files:
            if t.stem in aux_map:
                # choose first matching aux file (if multiple with different ext)
                paired.append((t, aux_map[t.stem][0]))
        if len(paired) == 0:
            raise RuntimeError("No paired files found between thermal and aux folders (matching by filename stem).")
        self.pairs = paired

        # masks
        self.masks_path = None
        if mask_dir is not None:
            candidates_m = [self.root / mask_dir / split, self.root / split / mask_dir, self.root / mask_dir]
            self.masks_path = next((p for p in candidates_m if p.exists()), None)
            if self.masks_path is not None:
                # prepare map by stem
                self.mask_map = {p.stem: p for p in (self.masks_path.rglob("*") if self.masks_path.exists() else [])}
            else:
                self.mask_map = {}
        else:
            self.mask_map = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        t_path, a_path = self.pairs[idx]
        t_np = _read_image(t_path)
        a_np = _read_image(a_path)

        t_t = _to_gray_tensor(t_np, radiometric=self.radiometric)
        a_t = _to_rgb_tensor(a_np)

        mask_t = None
        if t_path.stem in self.mask_map:
            mask_t = _read_mask(self.mask_map[t_path.stem])
            # resize if needed
            if mask_t.shape[1:] != t_t.shape[1:]:
                mask_pil = Image.fromarray((mask_t.squeeze(0).numpy().astype(np.uint8) * 255))
                mask_pil = mask_pil.resize((t_t.shape[2], t_t.shape[1]), resample=Image.NEAREST)
                mask_t = torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float() / 255.0

        label = 1 if (mask_t is not None and mask_t.any()) else 0

        sample = {"thermal": t_t, "aux": a_t, "mask": mask_t, "label": label, "filename": t_path.name}

        if self.transform is not None:
            # transform should support dict input or be a callable applied to sample
            out = self.transform(sample)
            if isinstance(out, dict):
                sample = out
            else:
                # if transform returns only image-like, update thermal only
                sample["thermal"] = out

        if self.return_filename:
            return sample["thermal"], sample["aux"], sample["mask"], sample["label"], sample["filename"]
        return sample["thermal"], sample["aux"], sample["mask"], sample["label"]
