"""
Often solar PV thermography datasets have:
- high-resolution thermal frames,
- per-panel bounding boxes / instance annotations (CSV/JSON) or binary masks for defects,
- optional per-image metadata (datetime, insolation, temperature).

This class provides:
- reading high-res thermal frames (radiometric-aware),
- optional cropping to panel bounding boxes (if present),
- support for per-panel labels and masks.

Arguments:
- annotations: optional path to CSV or JSON listing image filename, optional bbox (x,y,w,h), optional mask path, label.
CSV expected columns: filename, x, y, w, h, mask (optional), label (0/1)
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import csv
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from .InfraredAD import _read_image, _to_gray_tensor, _read_mask
except Exception:
    from InfraredAD import _read_image, _to_gray_tensor, _read_mask


class ThermoSolarPV(Dataset):
    """
    Dataset for PV thermal inspection.

    Parameters
    ----------
    root: Path or str
        Dataset root containing image folder.
    images_dir: str
        Relative images folder under root (default "images").
    split: str
        "train", "val", or "test".
    ann_file: Optional[str]
        Path to annotation CSV or JSON that contains per-image or per-panel annotations.
        If None, class will scan images folder and return whole-image samples.
    crop_to_panel: bool
        If True and annotations contain bounding boxes, returns the cropped panel region instead of full image.
    radiometric: bool
        Radiometric 16-bit handling.
    transform: callable
        Optional transform applied to returned sample dict.
    """

    def __init__(self,
                 root: Union[str, Path],
                 images_dir: str = "images",
                 split: str = "train",
                 ann_file: Optional[Union[str, Path]] = None,
                 crop_to_panel: bool = False,
                 radiometric: bool = True,
                 transform: Optional[Callable] = None):
        self.root = Path(root)
        self.split = split
        self.images_path = (self.root / images_dir / split) if (self.root / images_dir / split).exists() else (self.root / split / images_dir)
        if self.images_path is None or not self.images_path.exists():
            raise FileNotFoundError(f"Can't locate images directory for split={split} under {root}.")

        self.ann_file = Path(ann_file) if ann_file is not None else None
        self.crop_to_panel = crop_to_panel
        self.radiometric = radiometric
        self.transform = transform

        # Load image list
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
        files = []
        for e in exts:
            files.extend(sorted(self.images_path.rglob(e)))
        self.image_files = files

        # parse annotations if provided
        self.annotations = {}  # map filename -> list of panels (each panel: dict with bbox/mask/label)
        if self.ann_file is not None and self.ann_file.exists():
            if self.ann_file.suffix.lower() in [".json"]:
                with open(self.ann_file, "r") as f:
                    data = json.load(f)
                # expected format: dict of filename -> list of panel dicts or list of dicts with 'filename' field
                if isinstance(data, dict):
                    self.annotations = data
                elif isinstance(data, list):
                    for item in data:
                        fn = item.get("filename") or item.get("file") or item.get("image")
                        if fn is None:
                            continue
                        self.annotations.setdefault(fn, []).append(item)
            elif self.ann_file.suffix.lower() in [".csv"]:
                with open(self.ann_file, newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        fn = row.get("filename")
                        if fn is None:
                            continue
                        # parse bbox if present
                        panel = {}
                        try:
                            panel["x"] = int(float(row.get("x", 0)))
                            panel["y"] = int(float(row.get("y", 0)))
                            panel["w"] = int(float(row.get("w", 0)))
                            panel["h"] = int(float(row.get("h", 0)))
                        except Exception:
                            panel = {}
                        if "mask" in row and row["mask"]:
                            panel["mask"] = row["mask"]
                        panel["label"] = int(row.get("label", 0))
                        self.annotations.setdefault(fn, []).append(panel)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_np = _read_image(img_path)
        full_img_t = _to_gray_tensor(img_np, radiometric=self.radiometric)

        filename = img_path.name
        panels = self.annotations.get(filename, None)
        if panels is None or not self.crop_to_panel:
            sample = {"image": full_img_t, "filename": filename, "panels": panels}
            if self.transform:
                sample = self.transform(sample)
            return sample

        # else return list of cropped panels (if multiple, return as list)
        crops = []
        for p in panels:
            if "x" in p and "y" in p and "w" in p and "h" in p:
                x, y, w, h = int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])
                # crop on numpy then convert with same scaling to preserve radiometry
                crop_np = img_np[y:y + h, x:x + w]
                crop_t = _to_gray_tensor(crop_np, radiometric=self.radiometric)
                mask_t = None
                if "mask" in p and p["mask"]:
                    mask_path = self.root / p["mask"]
                    if mask_path.exists():
                        mask_t = _read_mask(mask_path)
                        # ensure cropped mask shape matches crop
                        if mask_t.shape[1:] != crop_t.shape[1:]:
                            mask_pil = Image.fromarray((mask_t.squeeze(0).numpy().astype(np.uint8) * 255))
                            mask_pil = mask_pil.crop((x, y, x + w, y + h))
                            mask_pil = mask_pil.resize((crop_t.shape[2], crop_t.shape[1]), resample=Image.NEAREST)
                            mask_t = torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float() / 255.0
                label = int(p.get("label", 0))
                panel_sample = {"image": crop_t, "mask": mask_t, "label": label, "bbox": (x, y, w, h)}
                if self.transform:
                    panel_sample = self.transform(panel_sample)
                crops.append(panel_sample)
            else:
                # if no bbox, return full image for that panel
                panel_sample = {"image": full_img_t, "mask": None, "label": int(p.get("label", 0)), "bbox": None}
                if self.transform:
                    panel_sample = self.transform(panel_sample)
                crops.append(panel_sample)

        # if only one panel, return single item, else list
        if len(crops) == 1:
            return crops[0]
        return crops


# Example usage:
# ds = ThermoSolarPV("/data/solar_pv", split="train", ann_file="/data/solar_pv/ann.csv", crop_to_panel=True)
# sample = ds[0]
