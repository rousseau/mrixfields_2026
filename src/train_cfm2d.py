#!/usr/bin/env python3
"""
OT-CFM 2D training + inference — MRIxFields 2026

Architecture:
  - UNetModel conditionné sur le domaine cible (class label) via label embedding
  - Input du modèle = cat(x_t, x_src)  →  (B, 2, H, W)
  - Output = champ vectoriel de vitesse  →  (B, 1, H, W)
  - Entraîné avec ExactOT conditional flow matching (OT-CFM)
  - Paires non-appairées (retro_train) : src et tgt sont de domaines différents

Usage:
  # Entraînement
  python src/train_cfm2d.py --config configs/cfm_T1W.yaml

  # Inférence
  python src/train_cfm2d.py --mode infer \\
    --config configs/cfm_T1W.yaml \\
    --checkpoint outputs/cfm2d/runs/cfm2d_T1W/weights/checkpoint_5000.pth \\
    --input_dir /data/T1W/0.1T/ \\
    --output_dir /data/predictions/T1W/0.1T_to_7T/ \\
    --source_field 0.1T --target_field 7T
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModel

# Transforms depuis le challenge (installé comme mrixfields)
from mrixfields.data.transforms import CenterCropOrPad, Compose, ScaleToMinusOneOne, ToTensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAINS: List[str] = ["0.1T", "1.5T", "3T", "5T", "7T"]
DOMAIN_TO_IDX: Dict[str, int] = {d: i for i, d in enumerate(DOMAINS)}
NUM_DOMAINS: int = len(DOMAINS)


# ===========================================================================
# Env / Path resolution
# ===========================================================================

def _load_env(env_arg: Optional[str]) -> Optional[dict]:
    """Charge configs/env/{env}.yaml et expande les variables shell ($WORK, etc.)."""
    if env_arg is None:
        return None
    env_path = env_arg if env_arg.endswith(".yaml") else f"configs/env/{env_arg}.yaml"
    if not os.path.isabs(env_path):
        if os.path.exists(env_path):
            env_path = os.path.abspath(env_path)
        else:
            # Fallback : relatif à la racine du projet (parent de src/)
            project_root = Path(__file__).parent.parent
            candidate = project_root / env_path
            if candidate.exists():
                env_path = str(candidate)
    with open(env_path) as f:
        raw = yaml.safe_load(f)
    return {k: os.path.expandvars(str(v)) for k, v in raw.items()}


def _resolve_paths(cfg: dict, env: Optional[dict]) -> dict:
    """Résout preprocessed_dir et output_dir depuis l'env + sous-chemins de la config.

    Si env est None (pas de --env), utilise les chemins absolus directs de la config
    (backward compat).
    """
    if env is None:
        return cfg
    output_root = Path(env["output_root"])
    data = cfg.setdefault("data", {})
    if "preprocessed_subdir" in data:
        data["preprocessed_dir"] = str(output_root / data["preprocessed_subdir"])
    if "output_subdir" in data:
        data["output_dir"] = str(output_root / data["output_subdir"])
    # Expose data_root pour le mode on-the-fly NIfTI
    if "data_root" in env:
        data.setdefault("data_root", env["data_root"])
    return cfg


# ==========================================================================
# Dataset
# ==========================================================================

def _make_transform(img_size: int) -> Compose:
    """Padding/crop centre → tenseur → [-1, 1]."""
    return Compose([
        CenterCropOrPad((img_size, img_size)),
        ToTensor(),
        ScaleToMinusOneOne(),
    ])


class CachedDomainDataset(Dataset):
    """Dataset par domaine : charge les .npz pré-extraits et retourne (image, domain_idx)."""

    def __init__(
        self,
        preprocessed_dir: Path,
        split: str,
        modality: str,
        field: str,
        img_size: int = 512,
    ):
        self.domain_idx = DOMAIN_TO_IDX[field]
        self.transform = _make_transform(img_size)
        d = Path(preprocessed_dir) / split / modality / field
        self.files = sorted(d.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"Aucun .npz dans {d}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        npz = np.load(self.files[idx])
        image = npz["image"]  # (H, W) float32 in [0, 1]
        image = self.transform(image)  # → (1, img_size, img_size) in [-1, 1]
        return image, self.domain_idx


# Mapping abréviation → répertoire réel du challenge
_SPLIT_ABBR_TO_DIR = {
    "retro_train": "Training_retrospective",
    "pro_train":   "Training_prospective",
    "pro_val":     "Validating_prospective",
    "pro_test":    "Testing_prospective",
}


class NIfTIDomainDataset(Dataset):
    """Dataset on-the-fly : lit des volumes NIfTI et extrait des coupes 2D axiales.

    Remplace CachedDomainDataset quand les fichiers .npz ne sont pas disponibles.
    Structure attendue : {data_root}/{split_dir}/{modality}/{field}/*.nii.gz
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modality: str,
        field: str,
        img_size: int = 512,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
    ):
        self.domain_idx = DOMAIN_TO_IDX[field]
        self.transform = _make_transform(img_size)
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std

        split_dir = _SPLIT_ABBR_TO_DIR.get(split, split)
        domain_dir = Path(data_root) / split_dir / modality / field
        nifti_files = sorted(domain_dir.glob("*.nii.gz"))
        if not nifti_files:
            raise FileNotFoundError(f"Aucun fichier .nii.gz dans {domain_dir}")

        self.samples: List[Tuple[Path, int]] = []
        for nifti_path in nifti_files:
            vol = nib.load(str(nifti_path)).get_fdata(dtype=np.float32)
            vmin, vmax = vol.min(), vol.max()
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
            n_slices = vol.shape[slice_axis]
            for i in range(n_slices):
                slc = self._get_slice(vol, i)
                if slc.std() > min_slice_std:
                    self.samples.append((nifti_path, i))

    def _get_slice(self, volume: np.ndarray, idx: int) -> np.ndarray:
        slicing = [slice(None)] * volume.ndim
        slicing[self.slice_axis] = idx
        return volume[tuple(slicing)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        nifti_path, slice_idx = self.samples[idx]
        vol = nib.load(str(nifti_path)).get_fdata(dtype=np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        slc = self._get_slice(vol, slice_idx)
        image = self.transform(slc)
        return image, self.domain_idx


def _make_infinite(loader: DataLoader):
    """Itérateur infini sur un DataLoader."""
    while True:
        yield from loader


# ==========================================================================
# Model
# ==========================================================================

def build_unet(cfg: dict, use_checkpoint: Optional[bool] = None) -> UNetModel:
    """Crée le UNetModel conditionné sur le domaine cible.

    in_channels=2  : cat(x_t, x_src)
    out_channels=1 : champ vectoriel (1 canal pour image mono-canal)
    num_classes=5  : domaines 0.1T / 1.5T / 3T / 5T / 7T
    """
    img_size = cfg["model"]["img_size"]
    model_channels = cfg["model"]["model_channels"]
    num_res_blocks = cfg["model"]["num_res_blocks"]
    if use_checkpoint is None:
        use_checkpoint = cfg["model"].get("use_gradient_checkpointing", True)

    # Architecture standard pour les résolutions usuelles
    if img_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        # Attention aux niveaux de downsampling 16× et 32× (spatial: 32×32 et 16×16)
        attention_resolutions = (16, 32)
    elif img_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
        attention_resolutions = (8, 16)
    else:
        raise ValueError(f"img_size non supporté : {img_size}")

    return UNetModel(
        image_size=img_size,
        in_channels=2,               # cat(x_t, x_src)
        model_channels=model_channels,
        out_channels=1,              # champ vectoriel pour image 1-canal
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=0.0,
        channel_mult=channel_mult,
        dims=2,
        num_classes=NUM_DOMAINS,     # conditioning sur domaine cible
        use_checkpoint=use_checkpoint,
        num_heads=4,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
    )


# ==========================================================================
# Training
# ==========================================================================

def train(cfg_path: str, env_path: Optional[str] = None, resume: Optional[str] = None) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, _load_env(env_path))
    # --resume CLI a priorité sur le champ "resume" de la config
    if resume is not None:
        cfg["resume"] = resume

    # --- Chemins ---
    preprocessed_dir_str = cfg["data"].get("preprocessed_dir")
    data_root_str = cfg["data"].get("data_root")
    output_dir = Path(cfg["data"]["output_dir"])
    modality = cfg["data"]["modality"]
    split = cfg["data"].get("split", "retro_train")
    domains_to_use = cfg["data"].get("domains", DOMAINS)

    # Sélection du mode de chargement :
    # - CachedDomainDataset  si preprocessed_dir contient des fichiers .npz
    # - NIfTIDomainDataset   sinon, lecture directe depuis les NIfTI (data_root requis)
    use_cache = (
        preprocessed_dir_str is not None
        and Path(preprocessed_dir_str).is_dir()
        and next(Path(preprocessed_dir_str).rglob("*.npz"), None) is not None
    )
    if not use_cache and data_root_str is None:
        raise RuntimeError(
            "Aucune source de données disponible : définissez preprocessed_dir "
            "(fichiers .npz) ou data_root (NIfTI on-the-fly) dans la config / env."
        )
    if use_cache:
        preprocessed_dir = Path(preprocessed_dir_str)
        print(f"Mode : cache .npz ({preprocessed_dir})")
    else:
        data_root = Path(data_root_str)
        print(f"Mode : NIfTI on-the-fly ({data_root})")

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Datasets & DataLoaders ---
    img_size = cfg["model"]["img_size"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 4)

    print(f"\nChargement datasets ({modality}, split={split}) :")
    loaders: Dict[str, any] = {}
    for field in domains_to_use:
        if use_cache:
            ds = CachedDomainDataset(preprocessed_dir, split, modality, field, img_size)
        else:
            ds = NIfTIDomainDataset(data_root, split, modality, field, img_size)
        print(f"  {field}: {len(ds)} slices")
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        loaders[field] = _make_infinite(loader)

    # --- Modèle ---
    model = build_unet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModèle: {n_params / 1e6:.1f}M paramètres")

    # --- Optimiseur ---
    lr = cfg["train"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # LR : plateau jusqu'à total_iters//2 puis décroissance linéaire jusqu'à 0
    total_iters = cfg["train"]["total_iters"]

    def _lr_lambda(step: int) -> float:
        decay_start = total_iters // 2
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / (total_iters - decay_start))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # --- Flow Matcher ---
    sigma = cfg["train"].get("sigma", 0.0)
    try:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        print("Flow matcher: ExactOT-CFM")
    except Exception:
        FM = ConditionalFlowMatcher(sigma=sigma)
        print("Flow matcher: CFM (indépendant)")

    # --- Reprendre depuis un checkpoint ---
    start_iter = 0
    resume = cfg.get("resume")
    if resume and Path(resume).exists():
        print(f"Reprise depuis : {resume}")
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1
        print(f"  → reprise à l'itération {start_iter}")

    save_every = cfg["train"].get("save_every", 5000)
    print_every = cfg["train"].get("print_every", 500)
    domain_list = list(domains_to_use)

    print(f"\nDébut entraînement: {total_iters} iters | batch={batch_size} | lr={lr} | sigma={sigma}")
    print(f"  save_every={save_every}, print_every={print_every}")
    print(f"  weights → {weights_dir}\n")

    model.train()
    t0 = time.time()
    loss_accum = 0.0

    for step in range(start_iter, total_iters):
        # ---- Tirer deux domaines distincts ----
        src_field, tgt_field = random.sample(domain_list, 2)
        tgt_idx = DOMAIN_TO_IDX[tgt_field]

        # ---- Batchs ----
        x0, _ = next(loaders[src_field])   # (B, 1, H, W) in [-1, 1]
        x1, _ = next(loaders[tgt_field])   # (B, 1, H, W) in [-1, 1]
        x0, x1 = x0.to(device), x1.to(device)

        # ---- OT-CFM ----
        t_batch, x_t, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # t_batch : (B,)  x_t : (B,1,H,W)  ut = x1 - x0 : (B,1,H,W)

        # ---- Forward ----
        x_in = torch.cat([x_t, x0], dim=1)  # (B, 2, H, W) — conditionné sur image source
        t_vec = t_batch.to(device)
        y = torch.full((x0.shape[0],), tgt_idx, dtype=torch.long, device=device)
        vt = model(t_vec, x_in, y=y)         # (B, 1, H, W) — champ vectoriel prédit

        # ---- Loss : MSE entre champ prédit et cible ----
        loss = F.mse_loss(vt, ut)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_accum += loss.item()

        # ---- Logging ----
        if (step + 1) % print_every == 0:
            avg_loss = loss_accum / print_every
            lr_cur = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(
                f"[{step+1:6d}/{total_iters}]  loss={avg_loss:.4f}"
                f"  lr={lr_cur:.2e}"
                f"  {src_field}→{tgt_field}"
                f"  t={elapsed/60:.1f}min"
            )
            loss_accum = 0.0

        # ---- Checkpoint intermédiaire ----
        if (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step+1}.pth"
            torch.save({
                "iter": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg_path": str(cfg_path),
            }, ckpt_path)
            print(f"  → Checkpoint : {ckpt_path}")

    # ---- Checkpoint final ----
    final_path = weights_dir / "model_final.pth"
    torch.save({
        "iter": total_iters - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg_path": str(cfg_path),
    }, final_path)
    print(f"\nEntraînement terminé. Modèle final : {final_path}")


# ==========================================================================
# Inference
# ==========================================================================

def _get_pad_params(h: int, w: int, th: int, tw: int):
    """Calcule les paramètres de padding/crop pour passer de (h,w) à (th,tw)."""
    dh, dw = th - h, tw - w
    ph1 = max(0, dh) // 2
    ph2 = max(0, dh) - ph1
    pw1 = max(0, dw) // 2
    pw2 = max(0, dw) - pw1
    # Si l'image est plus grande que la cible, on crop
    ch1 = max(0, -dh) // 2
    cw1 = max(0, -dw) // 2
    return ph1, ph2, pw1, pw2, ch1, cw1


def _pad_slice(sl: np.ndarray, th: int, tw: int):
    """Pad/crop centre une coupe 2D vers (th, tw). Retourne (image_padded, params)."""
    h, w = sl.shape
    ph1, ph2, pw1, pw2, ch1, cw1 = _get_pad_params(h, w, th, tw)
    sl_padded = np.pad(sl, [(ph1, ph2), (pw1, pw2)], mode="constant")
    sl_padded = sl_padded[ch1: ch1 + th, cw1: cw1 + tw]
    return sl_padded, (h, w, ph1, pw1)


def _unpad_slice(arr: np.ndarray, orig_h: int, orig_w: int, ph1: int, pw1: int) -> np.ndarray:
    """Inverse du padding : extrait la région valide et retourne shape (orig_h, orig_w)."""
    th, tw = arr.shape
    y0 = ph1
    x0 = pw1
    eff_h = min(orig_h, th - y0)
    eff_w = min(orig_w, tw - x0)
    out = np.zeros((orig_h, orig_w), dtype=arr.dtype)
    out[:eff_h, :eff_w] = arr[y0: y0 + eff_h, x0: x0 + eff_w]
    return out


def _euler_integrate(
    model: UNetModel,
    x_src: torch.Tensor,
    tgt_idx: int,
    device: torch.device,
    n_steps: int = 50,
) -> torch.Tensor:
    """Intégration Euler t=0→1 du champ vectoriel.

    Args:
        x_src : (1, 1, H, W) float32 in [-1, 1]  — coupe source
        tgt_idx : index du domaine cible
    Returns:
        (1, 1, H, W) float32 in [-1, 1]  — coupe générée
    """
    x = x_src.clone().to(device)
    x_src_dev = x_src.to(device)
    dt = 1.0 / n_steps
    y = torch.tensor([tgt_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(n_steps):
            t = torch.tensor([i * dt], dtype=torch.float32, device=device)
            x_in = torch.cat([x, x_src_dev], dim=1)  # (1, 2, H, W)
            vt = model(t, x_in, y=y)                 # (1, 1, H, W)
            x = x + dt * vt

    return x


def infer(
    cfg_path: str,
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    source_field: str,
    target_field: str,
    env_path: Optional[str] = None,
) -> None:
    """Génère des prédictions pour tous les volumes NIfTI d'un répertoire.

    Les volumes sont transformés coupe par coupe : source_field → target_field.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, _load_env(env_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    model = build_unet(cfg, use_checkpoint=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()
    print(f"Modèle chargé : {checkpoint}")

    img_size = cfg["model"]["img_size"]
    n_steps = cfg.get("inference", {}).get("n_steps", 50)
    tgt_idx = DOMAIN_TO_IDX[target_field]

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_path.glob("*.nii.gz"))
    if not nifti_files:
        nifti_files = sorted(input_path.glob("*.nii"))
    print(f"Inférence {source_field}→{target_field} : {len(nifti_files)} volumes")

    for nii_path in nifti_files:
        print(f"  {nii_path.name} …", end=" ", flush=True)
        nii_img = nib.load(str(nii_path))
        vol = nii_img.get_fdata(dtype=np.float32)  # (H, W, D) in [0, 1]

        # Inférence coupe par coupe selon l'axe axial (axis=2)
        slice_axis = 2
        n_slices = vol.shape[slice_axis]
        out_vol = np.zeros_like(vol)

        for s_idx in range(n_slices):
            sl = np.take(vol, s_idx, axis=slice_axis)  # (H, W)

            # Normaliser à [0,1] (au cas où les valeurs dépassent)
            sl = np.clip(sl, 0.0, 1.0)

            # Pad/crop → img_size × img_size
            sl_padded, pad_params = _pad_slice(sl, img_size, img_size)
            orig_h, orig_w, ph1, pw1 = pad_params

            # Tensor [-1, 1]
            x_src = torch.from_numpy(sl_padded * 2.0 - 1.0).float()
            x_src = x_src.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # Euler integration
            x_out = _euler_integrate(model, x_src, tgt_idx, device, n_steps)

            # Post-traitement : [-1,1] → [0,1]
            sl_out = x_out.squeeze().cpu().numpy()
            sl_out = np.clip((sl_out + 1.0) / 2.0, 0.0, 1.0)

            # Remettre à la taille originale
            sl_out = _unpad_slice(sl_out, orig_h, orig_w, ph1, pw1)

            # Insérer dans le volume de sortie
            idx = [slice(None)] * 3
            idx[slice_axis] = s_idx
            out_vol[tuple(idx)] = sl_out

        # Sauvegarder
        out_nii = nib.Nifti1Image(out_vol, nii_img.affine, nii_img.header)
        out_path = output_path / nii_path.name
        nib.save(out_nii, str(out_path))
        print(f"OK → {out_path.name}")


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="OT-CFM 2D — MRIxFields 2026")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML de config")
    parser.add_argument(
        "--env", default=None,
        help="Env config : 'local', 'jeanzay', ou chemin vers configs/env/*.yaml",
    )

    # Arguments pour le mode infer
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--source_field", default="0.1T")
    parser.add_argument("--target_field", default="7T")

    # Reprise d'un entraînement interrompu
    parser.add_argument("--resume", default=None,
                        help="Chemin vers un checkpoint .pth pour reprendre l'entraînement")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.config, env_path=args.env, resume=args.resume)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint requis pour le mode infer")
        if not args.input_dir:
            parser.error("--input_dir requis pour le mode infer")
        if not args.output_dir:
            parser.error("--output_dir requis pour le mode infer")
        infer(
            cfg_path=args.config,
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            source_field=args.source_field,
            target_field=args.target_field,
            env_path=args.env,
        )


if __name__ == "__main__":
    main()
