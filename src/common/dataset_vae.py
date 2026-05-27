#!/usr/bin/env python3
"""Unified multi-modal VAE dataset for MRIxFields.

Loads T1W + T2W + T2FLAIR from all field strengths (0.1T–7T)
for training VAEs that are shared across modalities and fields.

Expected directory layout:
    <data_root>/Training_retrospective/<modality>/<field>/*.nii.gz

Each sample is a 3D patch extracted via random crop (train) or center crop (val).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from common.io import (
    DOMAINS,
    MODALITIES,
    normalize_volume,
    resample_volume,
    list_nifti_files,
)

__all__ = ["MRIxFieldsMultimodalDataset"]


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


class MRIxFieldsMultimodalDataset(Dataset):
    """Multi-modal, multi-field patch dataset for VAE training.

    Parameters:
        data_root: path to challenge data root.
        splits: list of split keys, e.g. ["retro_train"].
        modalities: list of modalities to include;
                    default = ["T1W", "T2W", "T2FLAIR"].
        fields: list of field strengths to include;
                default = ["0.1T", "1.5T", "3T", "5T", "7T"].
        patch_size: spatial patch size (H, W, D).
        target_spacing: if given, resample to this isotropic spacing (mm).
        percentile_lower, percentile_upper: robust normalization bounds.
        is_training: whether to use random crop (True) or center crop (False).
        max_samples: optional cap on total number of samples.
    """

    def __init__(
        self,
        data_root: Path,
        splits: Sequence[str] = ("retro_train",),
        modalities: Sequence[str] = tuple(MODALITIES),
        fields: Sequence[str] = tuple(DOMAINS),
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        target_spacing: Optional[Tuple[float, float, float]] = None,
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        is_training: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.patch_size = tuple(patch_size)
        self.target_spacing = target_spacing
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.is_training = is_training

        # Collect all volume paths
        self.samples: List[Tuple[Path, str, str]] = []
        # (path, modality, field)
        for split in splits:
            for modality in modalities:
                for field in fields:
                    files = list_nifti_files(self.data_root, split, modality, field)
                    for p in files:
                        self.samples.append((p, modality, field))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError(
                f"No volumes found for splits={splits}, "
                f"modalities={modalities}, fields={fields}"
            )

        # Mappings for downstream use (e.g. conditioning)
        self.mod_to_idx = {m: i for i, m in enumerate(MODALITIES)}
        self.field_to_idx = {f: i for i, f in enumerate(DOMAINS)}

        print(
            f"MRIxFieldsMultimodalDataset: {len(self.samples)} patches "
            f"(train={is_training}, patch={patch_size})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, modality, field = self.samples[idx]

        # Load raw volume
        img = nib.load(str(path))
        vol = img.get_fdata(dtype=np.float32)
        spacing = np.abs(np.diag(img.affine)[:3])

        # Optional resampling
        if self.target_spacing is not None:
            vol = resample_volume(vol, spacing, self.target_spacing)

        # Normalization → [-1, 1]
        vol = normalize_volume(vol, self.percentile_lower, self.percentile_upper)

        # Crop / pad to patch_size
        patch = self._extract_patch(vol)

        # To tensor (1, H, W, D)
        x = torch.from_numpy(patch).unsqueeze(0)

        return {
            "x": x,
            "modality": modality,
            "field": field,
            "mod_idx": torch.tensor(self.mod_to_idx[modality], dtype=torch.long),
            "field_idx": torch.tensor(self.field_to_idx[field], dtype=torch.long),
            "path": str(path),
        }

    # ----------------------------------------------------------------------- #
    # Crop helpers                                                            #
    # ----------------------------------------------------------------------- #

    def _extract_patch(self, vol: np.ndarray) -> np.ndarray:
        ph, pw, pd = self.patch_size
        h, w, d = vol.shape[:3]

        # Pad if volume is smaller than patch
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_d = max(0, pd - d)
        if pad_h or pad_w or pad_d:
            vol = np.pad(
                vol,
                [
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                    (pad_d // 2, pad_d - pad_d // 2),
                ],
                mode="reflect",
            )
            h, w, d = vol.shape[:3]

        if self.is_training:
            # Random crop
            sh = np.random.randint(0, h - ph + 1)
            sw = np.random.randint(0, w - pw + 1)
            sd = np.random.randint(0, d - pd + 1)
        else:
            # Center crop
            sh = (h - ph) // 2
            sw = (w - pw) // 2
            sd = (d - pd) // 2

        return vol[sh : sh + ph, sw : sw + pw, sd : sd + pd].astype(np.float32)


# --------------------------------------------------------------------------- #
# Collate (optional — default collate works for dicts of tensors)            #
# --------------------------------------------------------------------------- #


def vae_multimodal_collate(batch: List[dict]) -> dict:
    """Standard dict-based collate for DataLoader."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ("modality", "field", "path"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
