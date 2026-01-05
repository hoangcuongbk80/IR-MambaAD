#!/usr/bin/env python3
"""
Usage examples:
    python train.py --data-root /path/to/dataset --exp-dir ./exp/run1 --mode recon
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import random_split

# Local imports
from datasets.dataloaders import create_loader
from datasets.InfraredAD import InfraredAD
from datasets.transforms import DictCompose, ToTensorIfNeeded, JointRandomCrop, JointRandomHorizontalFlip, AddGaussianNoise, RandomCutout, ToNormalize
from model.model import IRMambaAD
from model.losses import reconstruction_map_loss

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Dataset root folder (passed to InfraredAD)")
    p.add_argument("--exp-dir", type=str, default="./exp", help="Experiment output dir (checkpoints, logs)")
    p.add_argument("--mode", type=str, default="recon", choices=["recon", "pretrain"], help="Train mode")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batch-size-eval", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--save-interval", type=int, default=5, help="Epoch interval to save checkpoints")
    p.add_argument("--mwfm-base-ch", type=int, default=32)
    p.add_argument("--mwfm-out-ch", type=int, default=1)
    p.add_argument("--encoder-out-dim", type=int, default=256)
    p.add_argument("--num-decoder-stages", type=int, default=3)
    p.add_argument("--token-dim", type=int, default=256)
    p.add_argument("--state-dim", type=int, default=512)
    p.add_argument("--pretrained-encoder", action="store_true", help="Load ImageNet pretrained encoder weights if available")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(train: bool = True, crop_size: tuple = (256, 256), mean_thermal: Optional[float] = None, std_thermal: Optional[float] = None):
    transforms = []
    transforms.append(ToTensorIfNeeded(thermal_key="image", aux_key="aux", mask_key="mask"))
    if train:
        transforms.append(JointRandomCrop(crop_size))
        transforms.append(JointRandomHorizontalFlip(0.5))
        transforms.append(AddGaussianNoise(0.0, 0.01, key="image"))
        transforms.append(RandomCutout(p=0.3, key="image"))
    # Normalize — if mean/std unknown, pass None for identity
    transforms.append(ToNormalize(mean_thermal, std_thermal, mean_aux=None, std_aux=None))
    return DictCompose(transforms)


def build_datasets(args):
    # Create InfraredAD datasets for train/val/test
    train_ds = InfraredAD(root=args.data_root, split="train", radiometric=True, return_filename=True)
    val_ds = InfraredAD(root=args.data_root, split="val", radiometric=True, return_filename=True) if (Path(args.data_root) / "val").exists() else None
    test_ds = InfraredAD(root=args.data_root, split="test", radiometric=True, return_filename=True) if (Path(args.data_root) / "test").exists() else None
    return train_ds, val_ds, test_ds


def build_model(args, device):
    model = IRMambaAD(input_channels=1,
                      mwfm_base_ch=args.mwfm_base_ch,
                      mwfm_out_ch=args.mwfm_out_ch,
                      encoder_out_dim=args.encoder_out_dim,
                      num_decoder_stages=args.num_decoder_stages,
                      token_dim=args.token_dim,
                      state_dim=args.state_dim,
                      pretrained_encoder=args.pretrained_encoder)
    model = model.to(device)
    return model


def save_checkpoint(state: dict, path: str):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def evaluate_model(model: nn.Module, dataloader, device: str):
    model.eval()
    image_scores = []
    image_labels = []
    pixel_scores = []  # flattened masks/pixel scores for pixel-level metrics
    pixel_labels = []
    with torch.no_grad():
        for batch in dataloader:
            # collate returns dict per dataloaders.generic_collate
            # expected keys: 'image' (B,1,H,W), 'mask' (B,1,H,W) or None, 'label' list or tensor
            images = batch.get("image", None) or batch.get("f0", None)
            masks = batch.get("mask", None) or batch.get("f2", None)  # heuristics
            labels = batch.get("label", None)
            filenames = batch.get("filename", None)
            if isinstance(images, list):
                # fallback if collate left lists
                images = torch.stack(images, dim=0)
            images = images.to(device)
            recon = model(images)
            loss_s, mse_map = reconstruction_map_loss(recon, images, reduction="mean")
            # per-sample image score = max over map
            per_sample_scores = mse_map.view(mse_map.shape[0], -1).max(dim=1)[0].detach().cpu().numpy()
            if labels is None:
                # attempt labels extraction from batch (list)
                labels = batch.get("label", [0] * images.shape[0])
            # ensure labels to array
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy().tolist()
            if isinstance(labels, list):
                image_labels.extend([int(l) for l in labels])
            else:
                image_labels.extend([int(labels)] * len(per_sample_scores))
            image_scores.extend(per_sample_scores.tolist())

            # pixel-level accumulation
            if masks is not None:
                if isinstance(masks, list):
                    masks = torch.stack(masks, dim=0)
                # ensure shapes align
                if masks.shape[-2:] != mse_map.shape[-2:]:
                    masks = nn.functional.interpolate(masks, size=mse_map.shape[-2:], mode="nearest")
                msks = masks.view(masks.shape[0], -1).detach().cpu().numpy()
                maps = mse_map.view(mse_map.shape[0], -1).detach().cpu().numpy()
                pixel_labels.extend(msks.reshape(-1).tolist())
                pixel_scores.extend(maps.reshape(-1).tolist())

    metrics = {}
    # image-level metrics
    if SKLEARN_AVAILABLE and len(set(image_labels)) > 1:
        metrics["image_aucroc"] = roc_auc_score(image_labels, image_scores)
        metrics["image_ap"] = average_precision_score(image_labels, image_scores)
    else:
        metrics["image_aucroc"] = None
        metrics["image_ap"] = None

    # pixel-level metrics
    if SKLEARN_AVAILABLE and len(set(pixel_labels)) > 1 and len(pixel_labels) > 0:
        metrics["pixel_aucroc"] = roc_auc_score(pixel_labels, pixel_scores)
        metrics["pixel_ap"] = average_precision_score(pixel_labels, pixel_scores)
    else:
        metrics["pixel_aucroc"] = None
        metrics["pixel_ap"] = None

    return metrics


def train_recon(args):
    device = args.device
    print(f"Using device: {device}")
    train_ds, val_ds, test_ds = build_datasets(args)

    # Basic stats: compute means if desired. For now, leave normalization None to preserve radiometry
    train_transform = make_transforms(train=True, crop_size=(256, 256), mean_thermal=None, std_thermal=None)
    val_transform = make_transforms(train=False, crop_size=(256, 256), mean_thermal=None, std_thermal=None)

    # attach transforms
    train_ds.transform = train_transform
    if val_ds is not None:
        val_ds.transform = val_transform

    train_loader = create_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_loader(val_ds, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers) if val_ds is not None else None

    model = build_model(args, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(enabled=(device.startswith("cuda")))
    criterion = reconstruction_map_loss

    start_epoch = 0
    best_metric = -1.0

    os.makedirs(args.exp_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.exp_dir, "best_checkpoint.pth")

    if args.resume:
        if os.path.exists(args.resume):
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck.get("optim_state", optimizer.state_dict()))
            start_epoch = ck.get("epoch", 0) + 1
            best_metric = ck.get("best_metric", best_metric)
            print(f"Resumed checkpoint {args.resume} at epoch {start_epoch}")

    model.train()
    set_seed(args.seed)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        n_samples = 0
        for batch in train_loader:
            images = batch.get("image", None) or batch.get("f0", None)
            # images might be list; convert
            if isinstance(images, list):
                images = torch.stack(images, dim=0)
            images = images.to(device)
            optimizer.zero_grad()
            with amp.autocast(enabled=(device.startswith("cuda"))):
                recon = model(images)
                loss_scalar, _ = criterion(recon, images, reduction="mean")
            scaler.scale(loss_scalar).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss_scalar.item()) * images.shape[0]
            n_samples += images.shape[0]

        epoch_loss = running_loss / max(1, n_samples)
        epoch_time = time.time() - epoch_start

        # validation
        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device=device)
            val_metric_for_selection = val_metrics.get("image_aucroc") or ( -val_metrics.get("image_ap", 0) if val_metrics.get("image_ap") is not None else None)
        else:
            val_metric_for_selection = -epoch_loss

        # save checkpoint if best
        if val_metric_for_selection is not None and (best_metric is None or val_metric_for_selection > best_metric):
            best_metric = val_metric_for_selection
            ck = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "args": vars(args)
            }
            save_checkpoint(ck, checkpoint_path)
            print(f"[Epoch {epoch}] Saved new best checkpoint to {checkpoint_path}")

        # periodic save
        if (epoch + 1) % args.save_interval == 0:
            path = os.path.join(args.exp_dir, f"ck_epoch_{epoch+1}.pth")
            ck = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "args": vars(args)
            }
            save_checkpoint(ck, path)

        # logging
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss: {epoch_loss:.6f} | val_image_auc: {val_metrics.get('image_aucroc'):.4f} | time: {epoch_time:.1f}s")

    print("Training finished.")
    # final save
    save_checkpoint({"epoch": args.epochs-1, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "best_metric": best_metric, "args": vars(args)}, os.path.join(args.exp_dir, "final_checkpoint.pth"))


def train_pretrain_stub(args):
    """
    Placeholder for DINO-style self-supervised pretraining of MWFM+Encoder.
    Implementing a full DINO training pipeline is non-trivial and beyond the scope of this
    starter script; leave as a hook where you can integrate an existing DINO implementation.
    """
    print("Pretrain mode selected. This script includes only a stub for DINO-style pretraining.")
    print("Integrate your preferred self-supervised training loop here (DINO, MoCo-v3, BYOL, etc.).")
    return


def main():
    args = parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.mode == "recon":
        train_recon(args)
    else:
        train_pretrain_stub(args)


if __name__ == "__main__":
    main()
