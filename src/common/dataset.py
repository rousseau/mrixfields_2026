#!/usr/bin/env python3
"""Unified datasets for MRIxFields.

Consolidated from:
  - src/cfm/train_cfm_3d.py (NIfTILatentDataset)
  - src/vae3d/train_vae_3d.py (NIfTIVolumeDataset)
  - src/vae3d/train_vqvae.py (MRIxFieldsHybridDataset)
  - src/cfm/train_mmfm_3d.py (MultiModalNIfTILatentDataset)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from common.io import (
    FILE_RE,
    MODALITIES,
    DOMAINS,
    SPLIT_MAP,
    center_crop_or_pad_np,
    load_nifti_volume,
    normalize_volume,
    random_crop_or_pad_np,
    resample_volume,
)


# --------------------------------------------------------------------------- #
# Base dataset                                                                #
# --------------------------------------------------------------------------- #


class MRIxFieldsBaseDataset(Dataset):
    """Base class with common logic for all MRIxFields datasets.

    Handles file listing, preprocessing pipeline, and modality/domain indexing.
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modalities: Sequence[str],
        fields: Sequence[str],
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
        max_per_class: Optional[int] = None,
        random_crop_prob: float = 0.0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.modalities = list(modalities)
        self.fields = list(fields)
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.target_spacing = target_spacing
        self.volume_size = volume_size
        self.random_crop_prob = random_crop_prob

        self.mod_to_idx = {m: i for i, m in enumerate(self.modalities)}
        self.field_to_idx = {f: i for i, f in enumerate(self.fields)}

        self.samples: List[Tuple[Path, int, int]] = []
        self._build_samples(max_per_class)

        if not self.samples:
            raise FileNotFoundError(
                f"Aucun volume NIfTI trouvé dans {self.data_root}/"
                f"{SPLIT_MAP.get(split, split)} pour {modalities}×{fields}"
            )

    def _build_samples(self, max_per_class: Optional[int]):
        """Populate self.samples with (path, mod_idx, field_idx)."""
        split_dir = SPLIT_MAP.get(self.split, self.split)
        for modality in self.modalities:
            for field in self.fields:
                d = self.data_root / split_dir / modality / field
                files = sorted(d.glob("*.nii.gz"))
                if max_per_class is not None:
                    files = files[:max_per_class]
                mod_idx = self.mod_to_idx[modality]
                field_idx = self.field_to_idx[field]
                for p in files:
                    if FILE_RE.match(p.name) is None:
                        continue
                    self.samples.append((p, mod_idx, field_idx))

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """Load a single volume and apply full preprocessing."""
        vol, _ = load_nifti_volume(
            path,
            target_spacing=self.target_spacing,
            normalize=True,
            lo_pct=self.percentile_lower,
            hi_pct=self.percentile_upper,
        )
        # Optional random crop for training diversity
        if self.volume_size is not None:
            if self.random_crop_prob > 0 and random.random() < self.random_crop_prob:
                vol = random_crop_or_pad_np(vol, self.volume_size)
            else:
                vol = center_crop_or_pad_np(vol, self.volume_size)
        return torch.from_numpy(vol).unsqueeze(0)  # (1, H, W, D)

    def __len__(self) -> int:
        return len(self.samples)


# --------------------------------------------------------------------------- #
# Single-stream dataset (for VAE autoencoder pre-training)                    #
# --------------------------------------------------------------------------- #


class NIfTIVolumeDataset(MRIxFieldsBaseDataset):
    """Dataset for VAE pre-training.

    Returns a single preprocessed volume tensor (1, H, W, D) in [-1, 1].
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modality: str,
        domains: Sequence[str],
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
        max_per_class: Optional[int] = None,
        random_crop_prob: float = 0.0,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            modalities=[modality],
            fields=domains,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
            target_spacing=target_spacing,
            volume_size=volume_size,
            max_per_class=max_per_class,
            random_crop_prob=random_crop_prob,
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, _, _ = self.samples[idx]
        return self._load_tensor(path)


# --------------------------------------------------------------------------- #
# Latent dataset (for CFM training)                                           #
# --------------------------------------------------------------------------- #


class NIfTILatentDataset(MRIxFieldsBaseDataset):
    """Dataset for CFM 3D latent-space training.

    Returns (volume_tensor, domain_idx) where volume_tensor is (1, H, W, D).
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modality: str,
        domains: Sequence[str],
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
        max_per_domain: Optional[int] = None,
        random_crop_prob: float = 0.0,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            modalities=[modality],
            fields=domains,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
            target_spacing=target_spacing,
            volume_size=volume_size,
            max_per_class=max_per_domain,
            random_crop_prob=random_crop_prob,
        )
        # Override: use domain index (field) as the label
        self.samples = [(p, self.field_to_idx[self.fields[f_idx]]) for p, _, f_idx in self.samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, domain_idx = self.samples[idx]
        return self._load_tensor(path), domain_idx


# --------------------------------------------------------------------------- #
# Multi-modal dataset (for MMFM / multi-modal CFM)                            #
# --------------------------------------------------------------------------- #


class MultiModalNIfTILatentDataset(MRIxFieldsBaseDataset):
    """Multi-modal / multi-field dataset for MMFM v1 or multi-modal CFM.

    Returns (volume_tensor, mod_idx, field_idx, class_idx).
    class_idx = mod_idx * n_fields + field_idx (flat class index).
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modalities: Sequence[str],
        fields: Sequence[str],
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
        max_per_class: Optional[int] = None,
        random_crop_prob: float = 0.0,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            modalities=modalities,
            fields=fields,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
            target_spacing=target_spacing,
            volume_size=volume_size,
            max_per_class=max_per_class,
            random_crop_prob=random_crop_prob,
        )
        # Rebuild samples with flat class index
        self.samples = []
        split_dir = SPLIT_MAP.get(self.split, self.split)
        for modality in self.modalities:
            for field in self.fields:
                d = self.data_root / split_dir / modality / field
                files = sorted(d.glob("*.nii.gz"))
                if max_per_class is not None:
                    files = files[:max_per_class]
                mod_idx = self.mod_to_idx[modality]
                field_idx = self.field_to_idx[field]
                class_idx = mod_idx * len(self.fields) + field_idx
                for p in files:
                    if FILE_RE.match(p.name) is None:
                        continue
                    self.samples.append((p, mod_idx, field_idx, class_idx))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        path, mod_idx, field_idx, class_idx = self.samples[idx]
        x = self._load_tensor(path)
        return (
            x,
            torch.tensor(mod_idx, dtype=torch.long),
            torch.tensor(field_idx, dtype=torch.long),
            torch.tensor(class_idx, dtype=torch.long),
        )

    def check_coherence(self) -> Dict[str, any]:
        """Verify indexing and cardinality for each modality and field."""
        stats = {}
        for m_idx, m_name in enumerate(self.modalities):
            m_stats = {}
            for f_idx, f_name in enumerate(self.fields):
                count = sum(1 for s in self.samples if s[1] == m_idx and s[2] == f_idx)
                m_stats[f_name] = count
            stats[m_name] = m_stats
        return stats



# --------------------------------------------------------------------------- #
# Paired dataset (for VQ-VAE / disentanglement)                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampleMeta:
    path: Path
    split: str
    modality: str
    field: str
    subject_id: str


class MRIxFieldsPairedDataset(Dataset):
    """Dataset with paired cross-modal samples for VQ-VAE / disentanglement.

    Returns dict with x_src, x_tgt, src_mod, src_field, tgt_mod, tgt_field, is_paired.
    """

    def __init__(
        self,
        data_root: Path,
        splits: Sequence[str],
        modalities: Sequence[str],
        fields: Sequence[str],
        volume_size: Tuple[int, int, int],
        paired_prob: float = 0.5,
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.volume_size = volume_size
        self.paired_prob = paired_prob
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.target_spacing = target_spacing

        self.samples: List[SampleMeta] = []
        self.by_key: Dict[Tuple[str, str, str], Dict[str, SampleMeta]] = {}

        for split in splits:
            split_dir = SPLIT_MAP.get(split, split)
            for modality in modalities:
                for field in fields:
                    d = self.data_root / split_dir / modality / field
                    for p in sorted(d.glob("*.nii.gz")):
                        m = FILE_RE.match(p.name)
                        if m is None:
                            continue
                        subj = m.group(3)
                        meta = SampleMeta(
                            path=p,
                            split=split,
                            modality=modality,
                            field=field,
                            subject_id=subj,
                        )
                        self.samples.append(meta)
                        key = (split, field, subj)
                        if key not in self.by_key:
                            self.by_key[key] = {}
                        self.by_key[key][modality] = meta

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError("Aucun fichier NIfTI détecté.")

        self.mod_to_idx = {m: i for i, m in enumerate(modalities)}
        self.field_to_idx = {f: i for i, f in enumerate(fields)}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, meta: SampleMeta) -> torch.Tensor:
        img = nib.load(str(meta.path))
        vol = img.get_fdata(dtype=np.float32)
        if self.target_spacing is not None:
            spacing = np.abs(np.diag(img.affine)[:3])
            vol = resample_volume(vol, spacing, self.target_spacing)
        vol = normalize_volume(vol, self.percentile_lower, self.percentile_upper)
        vol = center_crop_or_pad_np(vol, self.volume_size)
        return torch.from_numpy(vol).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src = self.samples[idx]
        x_src = self._load_tensor(src)
        src_mod_idx = self.mod_to_idx[src.modality]
        src_field_idx = self.field_to_idx[src.field]

        candidates = self.by_key[(src.split, src.field, src.subject_id)]
        other_mods = [m for m in candidates.keys() if m != src.modality]

        is_paired = bool(other_mods) and (random.random() < self.paired_prob)

        if is_paired:
            tgt_mod = random.choice(other_mods)
            tgt = candidates[tgt_mod]
            x_tgt = self._load_tensor(tgt)
            tgt_mod_idx = self.mod_to_idx[tgt.modality]
            tgt_field_idx = self.field_to_idx[tgt.field]
        else:
            x_tgt = torch.zeros_like(x_src)
            tgt_mod_idx = -1
            tgt_field_idx = -1

        return {
            "x_src": x_src,
            "x_tgt": x_tgt,
            "src_mod": torch.tensor(src_mod_idx, dtype=torch.long),
            "src_field": torch.tensor(src_field_idx, dtype=torch.long),
            "tgt_mod": torch.tensor(tgt_mod_idx, dtype=torch.long),
            "tgt_field": torch.tensor(tgt_field_idx, dtype=torch.long),
            "is_paired": torch.tensor(1 if is_paired else 0, dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Latent cache dataset (MMFM multi-marginal, fast training)                   #
# --------------------------------------------------------------------------- #

import json as _json


class LatentCacheDataset(Dataset):
    """Dataset de latents MedVAE pré-encodés (volume ENTIER, pas de crop).

    Lit un cache produit par src/cfm/precompute_latents.py. Les latents sont
    chargés en RAM au démarrage (option) puis servis tels quels, avec une
    augmentation optionnelle par flip gauche/droite (axe latéral).

    Retourne (latent_tensor, mod_idx, field_idx, class_idx) où latent_tensor
    est (C, H', W', D') en float32.

    Args:
        cache_dir: dossier contenant index.json (…/latent_cache/<vae_id>/<split>).
        cache_root: racine relative aux chemins de l'index (…/latent_cache).
        preload_ram: si True, charge tous les latents en RAM.
        flip_lr_prob: probabilité de flip gauche/droite.
        flip_axis: axe spatial du flip dans le latent (0=H,1=W,2=D). Défaut 0.
    """

    def __init__(
        self,
        cache_dir: Path,
        cache_root: Path,
        preload_ram: bool = True,
        flip_lr_prob: float = 0.0,
        flip_axis: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_root = Path(cache_root)
        self.preload_ram = preload_ram
        self.flip_lr_prob = flip_lr_prob
        self.flip_axis = flip_axis

        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index de cache introuvable : {index_path}")
        with open(index_path) as f:
            self.meta = _json.load(f)

        self.samples = self.meta["samples"]
        if not self.samples:
            raise RuntimeError(f"Cache vide : {index_path}")
        self.latent_shape = tuple(self.meta.get("latent_shape") or ())

        self._ram: Dict[str, torch.Tensor] = {}
        if self.preload_ram:
            for s in self.samples:
                self._ram[s["path"]] = self._load_disk(s["path"])

    def _load_disk(self, rel_path: str) -> torch.Tensor:
        z = torch.load(self.cache_root / rel_path, map_location="cpu")
        return z.to(torch.float32)

    def _get_latent(self, rel_path: str) -> torch.Tensor:
        if self.preload_ram:
            return self._ram[rel_path].clone()
        return self._load_disk(rel_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        z = self._get_latent(s["path"])  # (C, H', W', D')
        if self.flip_lr_prob > 0 and random.random() < self.flip_lr_prob:
            # axe spatial +1 car dim0 = canaux
            z = torch.flip(z, dims=[self.flip_axis + 1])
        return (
            z,
            torch.tensor(s["mod_idx"], dtype=torch.long),
            torch.tensor(s["field_idx"], dtype=torch.long),
            torch.tensor(s["class_idx"], dtype=torch.long),
        )
