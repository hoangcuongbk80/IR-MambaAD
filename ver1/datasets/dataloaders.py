"""

Helpers to create DataLoaders for train/val/test with sensible defaults and a collate function
that handles dict-style or tuple-style dataset samples.

Functions:
- generic_collate(batch): stacks tensors where possible, packs filenames and labels as lists.
- create_loader(dataset, batch_size, shuffle, num_workers, pin_memory)
- create_split_loaders(train_ds, val_ds, test_ds, ...)

Usage:
    from datasets.dataloaders import create_loader
    train_loader = create_loader(train_ds, batch_size=16, shuffle=True)
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader

def generic_collate(batch: List[Any]) -> Dict[str, Any]:
    """
    Collate that supports:
    - dict samples: keys -> either stackable tensors or lists
    - tuple samples: assumed (thermal, aux, mask, label, filename) or similar
    Returns a dict with keys 'thermal'/'image', 'aux', 'mask', 'label', 'filename' if present.
    """
    if len(batch) == 0:
        return {}
    first = batch[0]
    if isinstance(first, dict):
        keys = set()
        for b in batch:
            keys.update(b.keys())
        out = {}
        for k in keys:
            vals = [b.get(k, None) for b in batch]
            # if all are tensors, stack, else keep list
            if all(isinstance(v, torch.Tensor) for v in vals if v is not None):
                try:
                    out[k] = torch.stack([v for v in vals], dim=0)
                except Exception:
                    out[k] = vals
            else:
                out[k] = vals
        return out
    else:
        # tuple style
        # build outputs by position
        max_len = max(len(b) for b in batch)
        out = {}
        for pos in range(max_len):
            vals = [b[pos] if len(b) > pos else None for b in batch]
            if all(isinstance(v, torch.Tensor) for v in vals if v is not None):
                try:
                    out[f"f{pos}"] = torch.stack([v for v in vals], dim=0)
                except Exception:
                    out[f"f{pos}"] = vals
            else:
                out[f"f{pos}"] = vals
        return out


def create_loader(dataset,
                  batch_size: int = 8,
                  shuffle: bool = False,
                  num_workers: int = 4,
                  pin_memory: bool = True,
                  collate_fn = generic_collate) -> DataLoader:
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)


def create_split_loaders(train_dataset=None,
                         val_dataset=None,
                         test_dataset=None,
                         batch_size_train: int = 8,
                         batch_size_eval: int = 16,
                         num_workers: int = 4) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    train_loader = create_loader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers) if train_dataset is not None else None
    val_loader = create_loader(val_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers) if val_dataset is not None else None
    test_loader = create_loader(test_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers) if test_dataset is not None else None
    return train_loader, val_loader, test_loader
