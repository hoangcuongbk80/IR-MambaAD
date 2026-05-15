#!/usr/bin/env python3
"""

Usage:
    python test.py --data-root /path/to/dataset --checkpoint ./exp/best_checkpoint.pth --out-dir ./results
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from datasets.InfraredAD import InfraredAD
from datasets.transforms import ToTensorIfNeeded, ToNormalize
from datasets.dataloaders import create_loader
from model.model import IRMambaAD
from model.losses import reconstruction_map_loss

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import imageio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="./results")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-maps", action="store_true", help="Save anomaly maps (PNG) into out-dir/maps/")
    p.add_argument("--map-sigma", type=float, default=1.5, help="Optional gaussian smoothing sigma for visualization")
    return p.parse_args()


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device)
    return ck


def build_model_from_ck(ck: dict, device: str):
    args = ck.get("args", {})
    # create model using stored args if present, else fallback to defaults
    model = IRMambaAD(input_channels=args.get("input_channels", 1),
                      mwfm_base_ch=args.get("mwfm_base_ch", 32),
                      mwfm_out_ch=args.get("mwfm_out_ch", 1),
                      encoder_out_dim=args.get("encoder_out_dim", 256),
                      num_decoder_stages=args.get("num_decoder_stages", 3),
                      token_dim=args.get("token_dim", 256),
                      state_dim=args.get("state_dim", 512),
                      pretrained_encoder=False)
    model.load_state_dict(ck["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(model: nn.Module, dataloader, device: str, out_dir: Optional[str] = None, save_maps: bool = False):
    image_scores = []
    image_labels = []
    pixel_scores = []
    pixel_labels = []
    filenames_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch.get("image", None) or batch.get("f0", None)
            masks = batch.get("mask", None)
            filenames = batch.get("filename", None) or batch.get("f4", None)
            labels = batch.get("label", None)

            # stack lists if needed
            if isinstance(images, list):
                images = torch.stack(images, dim=0)
            images = images.to(device)
            recon = model(images)
            _, mse_map = reconstruction_map_loss(recon, images, reduction="mean")
            # image score = per-sample max
            scores = mse_map.view(mse_map.shape[0], -1).max(dim=1)[0].detach().cpu().numpy().tolist()
            image_scores.extend(scores)

            # accumulate labels
            if isinstance(labels, torch.Tensor):
                lab = labels.detach().cpu().numpy().tolist()
            elif isinstance(labels, list):
                lab = [int(x) for x in labels]
            else:
                lab = [int(labels)] * len(scores) if labels is not None else [0] * len(scores)
            image_labels.extend(lab)

            # pixel-level
            if masks is not None:
                if isinstance(masks, list):
                    masks = torch.stack(masks, dim=0)
                if masks.shape[-2:] != mse_map.shape[-2:]:
                    masks = nn.functional.interpolate(masks, size=mse_map.shape[-2:], mode="nearest")
                msks = masks.detach().cpu().numpy().reshape(-1).tolist()
                maps = mse_map.detach().cpu().numpy().reshape(-1).tolist()
                pixel_labels.extend(msks)
                pixel_scores.extend(maps)

            # optionally save maps png
            if save_maps:
                if filenames is None:
                    filenames = [f"img_{i}.png" for i in range(len(scores))]
                for i, fn in enumerate(filenames):
                    map_i = mse_map[i].detach().cpu().numpy()
                    # normalize for visualization 0-255
                    mn, mx = float(map_i.min()), float(map_i.max())
                    vis = (map_i - mn) / (mx - mn + 1e-8)
                    vis_u8 = (vis * 255).astype("uint8")
                    out_name = Path(out_dir) / "maps" / (Path(fn).stem + "_anom.png")
                    os.makedirs(out_name.parent, exist_ok=True)
                    imageio.imwrite(str(out_name), vis_u8)

    results = {}
    if SKLEARN_AVAILABLE:
        if len(set(image_labels)) > 1:
            results["image_aucroc"] = roc_auc_score(image_labels, image_scores)
            results["image_ap"] = average_precision_score(image_labels, image_scores)
        else:
            results["image_aucroc"] = None
            results["image_ap"] = None

        if len(pixel_labels) > 0 and len(set(pixel_labels)) > 1:
            results["pixel_aucroc"] = roc_auc_score(pixel_labels, pixel_scores)
            results["pixel_ap"] = average_precision_score(pixel_labels, pixel_scores)
        else:
            results["pixel_aucroc"] = None
            results["pixel_ap"] = None
    else:
        results["image_aucroc"] = None
        results["image_ap"] = None
        results["pixel_aucroc"] = None
        results["pixel_ap"] = None

    return results


def main():
    args = parse_args()
    device = args.device
    ck = load_checkpoint(args.checkpoint, device)
    model = build_model_from_ck(ck, device)

    # dataset
    test_ds = InfraredAD(root=args.data_root, split="test", radiometric=True, return_filename=True)
    test_ds.transform = ToTensorIfNeeded()
    test_loader = create_loader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.out_dir, exist_ok=True)
    results = evaluate_checkpoint(model, test_loader, device=device, out_dir=args.out_dir, save_maps=args.save_maps)

    # save results
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json = __import__("json")
        json.dump(results, f, indent=2)

    print("Evaluation results:", results)


if __name__ == "__main__":
    main()
