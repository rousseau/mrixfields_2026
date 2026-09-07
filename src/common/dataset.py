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
        field_norm_stats: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
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
        # {modality: {field: {"lo": float, "hi": float}}} — see
        # compute_field_norm_stats.py. When set, normalization uses these
        # FIXED per-field statistics instead of this volume's own percentiles,
        # preserving genuine inter-field intensity-scale differences.
        self.field_norm_stats = field_norm_stats

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

    def _load_tensor(
        self, path: Path, modality: Optional[str] = None, field: Optional[str] = None
    ) -> torch.Tensor:
        """Load a single volume and apply full preprocessing."""
        fixed_lo = fixed_hi = None
        if self.field_norm_stats is not None and modality is not None and field is not None:
            entry = self.field_norm_stats.get(modality, {}).get(field)
            if entry is not None:
                fixed_lo, fixed_hi = entry["lo"], entry["hi"]
        vol, _ = load_nifti_volume(
            path,
            target_spacing=self.target_spacing,
            normalize=True,
            lo_pct=self.percentile_lower,
            hi_pct=self.percentile_upper,
            fixed_lo=fixed_lo,
            fixed_hi=fixed_hi,
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
        field_norm_stats: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
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
            field_norm_stats=field_norm_stats,
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
        x = self._load_tensor(path, self.modalities[mod_idx], self.fields[field_idx])
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

import hashlib as _hashlib
import json as _json


def flat_latent_cache_id(
    cfg: dict,
    target_spacing,
    volume_size,
    p_lo: float,
    p_hi: float,
    field_norm_stats_path: Optional[str] = None,
    encode_tile=None,
    encode_tile_margin: Optional[int] = None,
    amp_dtype: Optional[str] = None,
) -> str:
    """Cache identifier for FlatLatentCacheDataset — shared by
    precompute_mmfm_latents.py (writer) and train_mmfm_3d.py (reader) so both
    always agree on the same cache path for a given config. Any change to the
    VAE checkpoint, spacing, crop size, or normalization percentiles yields a
    different id, so a stale/mismatched cache is never silently reused.

    field_norm_stats_path: path to a compute_field_norm_stats.py JSON, if
    fixed per-field normalization is used instead of per-volume percentiles
    (see normalize_volume_fixed). Included in the hash so switching between
    per-volume and per-field normalization always yields a fresh cache_id
    rather than silently reusing latents encoded under the other scheme.

    encode_tile / encode_tile_margin / amp_dtype — AJOUTÉS LE 2026-09-07, et la
    collision qu'ils réparent avait DÉJÀ servi.

    Le SCHÉMA D'ENCODAGE ne faisait pas partie de la clé, alors que les deux
    scripts d'écriture n'encodent pas de la même façon :
      - `precompute_unet_latents.py:162` appelle `tiled_encode` (tuiles jointives
        96x112x96 + marge de contexte 16) ;
      - `precompute_mmfm_latents.py:149` appelle `vae.encode` DIRECTEMENT, donc la
        fenêtre glissante gaussienne interne de MedVAE (`roi_size_calc`,
        `gpu_dim=160`) — une troisième géométrie, à fenêtres recouvrantes.
    Les deux sortent en `flat_dim=129024`, donc aucune vérification de forme ne
    bronche. Et les deux produisaient le MÊME identifiant.

    Conséquence déjà consommée : `medvae_finetune_1989e9d1` (adoption LPIPS,
    2026-09-02) et `medvae_finetune_ff550d64` (cache sans normalisation,
    2026-09-06) ont été écrits par `precompute_mmfm_latents.py`, donc NON TUILÉS,
    puis comparés à `medvae_finetune_c4d1e200` qui l'est. Les deux verdicts
    négatifs de ces campagnes portaient donc deux variables et pas une. L'écart
    entre les deux schémas est mesuré : 0.1140 contre 0.1168 de nRMSE sur la
    tranche notée (`results/mmfm/representation_20260906/manifest.md`).

    NOTE DE COMPATIBILITÉ : les répertoires de cache existants portent des noms
    produits par l'ANCIENNE clé, que cette fonction ne peut plus reproduire. Ils
    restent utilisables — les configs de production fixent `latent_cache_dir` en
    dur — et c'est `index.json` qui fait foi sur leur provenance (il porte
    désormais `encode_tile`/`encode_tile_margin`/`amp_dtype`/`encode_scheme`).
    Le garde-fou qui protège réellement est
    `infer_mmfm_unified._check_cache_consistency`, pas le nom du répertoire.
    """
    vae_cfg = cfg.get("vae", {})
    key = "|".join([
        str(vae_cfg.get("vae_type", "vae")),
        str(vae_cfg.get("checkpoint", "")),
        str(target_spacing),
        str(volume_size),
        f"{p_lo}",
        f"{p_hi}",
        f"field_norm={field_norm_stats_path or ''}",
        f"tile={tuple(encode_tile) if encode_tile else None}",
        f"margin={encode_tile_margin}",
        f"amp={amp_dtype or ''}",
    ])
    h = _hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"{vae_cfg.get('vae_type', 'vae')}_{h}"


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


class FlatLatentCacheDataset(Dataset):
    """Dataset de vecteurs latents MMFM pré-encodés ET pré-aplatis.

    Lit un cache produit par src/cfm/precompute_mmfm_latents.py, qui reproduit
    exactement le pipeline resample->normalize->center_crop_or_pad->encode->
    to_vector() de MultiModalNIfTILatentDataset. Contrairement à
    LatentCacheDataset (volume entier, format spatial, nécessite to_vector()
    à l'usage), les tenseurs ici sont déjà aplatis (flat_dim,) et prêts à
    l'emploi direct par VectorMMFM — aucun encode/resample/crop restant dans
    la boucle d'entraînement.

    Retourne (latent_vec, mod_idx, field_idx, class_idx).

    Augmentation par flip : le vecteur étant un latent SPATIAL aplati, le flip
    ne peut se faire qu'en le remettant en forme, d'où `latent_shape`. Il est
    donc OBLIGATOIRE dès que `flip_lr_prob > 0` — sans lui on ne saurait pas
    quel axe retourner. Ce paramètre a longtemps manqué : `flip_lr_prob` était
    déclaré dans les configs vectorisée et INR mais la classe ne l'acceptait
    pas, si bien qu'il était silencieusement ignoré et que seul l'UNet
    bénéficiait de l'augmentation (écart mesuré : 26 % de L2 relatif sur un
    latent retourné). Une exception vaut mieux qu'un paramètre avalé.

    ATTENTION — ne convient PAS à un latent non spatial. Le `z` d'un INR est un
    vecteur de modulation GLOBAL, sans structure spatiale : le retourner n'a
    aucun sens géométrique (il faudrait réajuster `z` sur le volume retourné).
    Voir `cfm/arch_inr.py`, qui refuse explicitement cette augmentation.
    """

    def __init__(self, cache_dir: Path, cache_root: Path, preload_ram: bool = True,
                 flip_lr_prob: float = 0.0, flip_axis: int = 0,
                 latent_shape: Optional[Tuple[int, ...]] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_root = Path(cache_root)
        self.preload_ram = preload_ram
        self.flip_lr_prob = float(flip_lr_prob)
        self.flip_axis = int(flip_axis)
        self.latent_shape = tuple(latent_shape) if latent_shape else None
        if self.flip_lr_prob > 0:
            if self.latent_shape is None or len(self.latent_shape) != 4:
                raise ValueError(
                    "flip_lr_prob > 0 exige latent_shape=(C, H, W, D) : un vecteur "
                    "aplati ne peut être retourné sans savoir comment le remettre "
                    "en forme. Reçu latent_shape=" + repr(self.latent_shape)
                )
            if not 0 <= self.flip_axis <= 2:
                raise ValueError(f"flip_axis doit être dans [0, 2], reçu {self.flip_axis}")

        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index de cache introuvable : {index_path}")
        with open(index_path) as f:
            self.meta = _json.load(f)

        self.samples = self.meta["samples"]
        if not self.samples:
            raise RuntimeError(f"Cache vide : {index_path}")
        self.flat_dim = self.meta.get("flat_dim")
        if self.flip_lr_prob > 0:
            expected = int(np.prod(self.latent_shape))
            if self.flat_dim is not None and self.flat_dim != expected:
                raise ValueError(
                    f"latent_shape={self.latent_shape} donne {expected} éléments, "
                    f"mais le cache annonce flat_dim={self.flat_dim} — le flip "
                    "remettrait le vecteur dans une forme qui n'est pas la sienne."
                )
        n_fields = len(self.meta["fields"])
        for s in self.samples:
            s["class_idx"] = s["mod_idx"] * n_fields + s["field_idx"]

        self._ram: Dict[str, torch.Tensor] = {}
        if self.preload_ram:
            for s in self.samples:
                self._ram[s["path"]] = self._load_disk(s["path"])

    def _load_disk(self, rel_path: str) -> torch.Tensor:
        return torch.load(self.cache_root / rel_path, map_location="cpu").to(torch.float32)

    def _get_latent(self, rel_path: str) -> torch.Tensor:
        # `.clone()` quand le tenseur vient de la RAM partagée : le flip crée un
        # nouveau tenseur, mais sans copie une modification en place ultérieure
        # corromprait le cache pour tous les échantillons suivants.
        if self.preload_ram:
            return self._ram[rel_path].clone()
        return self._load_disk(rel_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        z = self._get_latent(s["path"])
        if self.flip_lr_prob > 0 and random.random() < self.flip_lr_prob:
            # Le vecteur est un latent spatial aplati : on le remet en forme,
            # on retourne l'axe demandé, puis on ré-aplatit. `+1` car dim 0 =
            # canaux, exactement comme LatentCacheDataset — les deux
            # architectures voient ainsi la MÊME augmentation.
            z = torch.flip(z.view(self.latent_shape), dims=[self.flip_axis + 1]).reshape(-1)
        return (
            z,
            torch.tensor(s["mod_idx"], dtype=torch.long),
            torch.tensor(s["field_idx"], dtype=torch.long),
            torch.tensor(s["class_idx"], dtype=torch.long),
        )
