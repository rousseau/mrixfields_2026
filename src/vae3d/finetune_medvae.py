#!/usr/bin/env python3
"""
Fine-tuning de MedVAE 3D sur l'intégralité de Training_retrospective.

Objectif : adapter les poids Stanford MIMI (pré-entraînés sur 1M images médicales)
au domaine MRIxFields — toutes modalités (T1W, T2W, T2FLAIR) et tous champs
(0.1T, 1.5T, 3T, 5T, 7T) — afin d'obtenir un espace latent unifié cross-field.

Différences clés par rapport au training précédent (train_vae.py) :
  - Dataset : Training_retrospective complet (~1939 volumes), pas 3 volumes prospectifs
  - Toutes modalités × tous champs (modèle généraliste, pas T1W uniquement)
  - Loss : L1 reconstruction (plus robuste que MSE sur les images médicales)
  - LR   : 1e-5 avec warmup cosine (fine-tuning, 10× < training scratch)
  - Checkpoints intermédiaires + model_best.pth sur validation loss
  - Auto-requeue SLURM (signal USR1 → sauvegarde + requeue)
  - Compatible DDP (torchrun)

Usage :
  # Single GPU — local
  python src/vae3d/finetune_medvae.py \\
      --config configs/medvae_finetune_all.yaml --env local

  # Multi-GPU — local (torchrun)
  torchrun --nproc_per_node=2 src/vae3d/finetune_medvae.py \\
      --config configs/medvae_finetune_all.yaml --env local

  # Jean Zay (via SLURM)
  sbatch src/slurm/finetune_medvae_jeanzay.slurm

  # Reprendre depuis un checkpoint
  python src/vae3d/finetune_medvae.py \\
      --config configs/medvae_finetune_all.yaml --env local \\
      --resume outputs/medvae/runs/medvae_finetune_all/weights/medvae_step_010000.pth

Format du checkpoint sauvegardé :
  {
    "step":          int,
    "model":         MVAE.state_dict(),   # clés "model.encoder.*", "model.decoder.*"
    "optimizer":     ...,
    "scheduler":     ...,
    "best_val_loss": float,
    "cfg_path":      str,
  }
  → directement chargeable par benchmark_vae.py (clé "model" → state dict MVAE).
"""

import argparse
import os
import re
import signal
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from scipy.ndimage import zoom as scipy_zoom
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SPLIT_MAP = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}

ALL_MODALITIES = ["T1W", "T2W", "T2FLAIR"]
ALL_FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]

# Pattern de nommage des fichiers NIfTI du challenge
FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")


# ---------------------------------------------------------------------------
# Utilitaires de prétraitement
# ---------------------------------------------------------------------------


def _resample_volume(
    vol: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Rééchantillonnage isotrope par zoom scipy (ordre 1)."""
    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def _normalize(
    vol: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> np.ndarray:
    """Normalisation robuste aux percentiles → [-1, 1]."""
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MRIxFieldsDataset(Dataset):
    """Dataset multi-modalité × multi-champ pour le fine-tuning de MedVAE.

    Chaque sample est un patch 3D (1, pH, pW, pD) extrait aléatoirement
    (train) ou centré (val) depuis un volume NIfTI complet.

    Structure attendue des données :
      <data_root>/<split_dir>/<modality>/<field>/X_<MOD>_<FIELD>_<ID>.nii.gz
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "retro_train",
        modalities: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        patch_size: Tuple[int, int, int] = (128, 128, 80),
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        is_training: bool = True,
    ):
        self.patch_size = patch_size
        self.lo_pct = percentile_lower
        self.hi_pct = percentile_upper
        self.target_spacing = target_spacing
        self.is_training = is_training

        modalities = modalities or ALL_MODALITIES
        fields = fields or ALL_FIELDS
        split_dir = SPLIT_MAP.get(split, split)

        self.volumes: List[Path] = []
        missing_dirs: List[str] = []

        for mod in modalities:
            for field in fields:
                d = Path(data_root) / split_dir / mod / field
                if not d.exists():
                    missing_dirs.append(str(d))
                    continue
                found = sorted(p for p in d.glob("*.nii.gz") if FILE_RE.match(p.name))
                self.volumes.extend(found)

        if missing_dirs:
            print(f"  [WARN] {len(missing_dirs)} répertoire(s) introuvable(s) :")
            for d in missing_dirs[:5]:
                print(f"    {d}")
            if len(missing_dirs) > 5:
                print(f"    ... ({len(missing_dirs) - 5} de plus)")

        if not self.volumes:
            raise FileNotFoundError(
                f"Aucun volume NIfTI sous {data_root}/{split_dir}/<mod>/<field>/\n"
                f"Modalités : {modalities}\nChamps : {fields}"
            )

        # Statistiques par (modalité, champ)
        from collections import Counter

        counts = Counter()
        for p in self.volumes:
            m = FILE_RE.match(p.name)
            if m:
                counts[(p.parent.parent.name, p.parent.name)] += 1

        print(
            f"  MRIxFieldsDataset [{split}] : {len(self.volumes)} volumes\n"
            + "".join(
                f"    {mod:8s} {field:5s} : {n}\n"
                for (mod, field), n in sorted(counts.items())
            ).rstrip()
        )

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.volumes[idx]
        img = nib.load(str(path))
        vol = img.get_fdata(dtype=np.float32)

        # Rééchantillonnage
        if self.target_spacing is not None:
            spacing = np.abs(np.diag(img.affine)[:3])
            if not np.allclose(spacing, self.target_spacing, atol=0.05):
                vol = _resample_volume(vol, spacing, self.target_spacing)

        # Normalisation percentile
        vol = _normalize(vol, self.lo_pct, self.hi_pct)

        # Crop / padding
        ph, pw, pd = self.patch_size
        h, w, d = vol.shape

        # Padding si volume plus petit que le patch
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_d = max(0, pd - d)
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            vol = np.pad(vol, [(0, pad_h), (0, pad_w), (0, pad_d)], mode="reflect")
            h, w, d = vol.shape

        if self.is_training:
            sh = np.random.randint(0, h - ph + 1)
            sw = np.random.randint(0, w - pw + 1)
            sd = np.random.randint(0, d - pd + 1)
        else:
            sh = (h - ph) // 2
            sw = (w - pw) // 2
            sd = (d - pd) // 2

        vol = vol[sh : sh + ph, sw : sw + pw, sd : sd + pd]
        return torch.from_numpy(vol).unsqueeze(0).float()  # (1, pH, pW, pD)


# ---------------------------------------------------------------------------
# Helpers config / env
# ---------------------------------------------------------------------------


def _load_env(env_arg: Optional[str]) -> Optional[dict]:
    """Charge le fichier YAML d'environnement (local | jeanzay | chemin direct)."""
    if env_arg is None:
        return None
    env_path = env_arg if env_arg.endswith(".yaml") else f"configs/env/{env_arg}.yaml"
    if not os.path.isabs(env_path):
        # Chercher depuis la racine du projet (2 niveaux au-dessus de ce script)
        candidate = Path(__file__).resolve().parent.parent.parent / env_path
        if candidate.exists():
            env_path = str(candidate)
    with open(env_path) as f:
        raw = yaml.safe_load(f)
    return {k: os.path.expandvars(str(v)) for k, v in raw.items()}


def _resolve_paths(cfg: dict, env: Optional[dict]) -> dict:
    """Injecte data_root et output_dir depuis l'environnement dans la config."""
    if env is None:
        return cfg
    data = cfg.setdefault("data", {})
    if "output_subdir" in data and "output_root" in env:
        data["output_dir"] = str(Path(env["output_root"]) / data["output_subdir"])
    data.setdefault("data_root", env.get("data_root", ""))
    return cfg


def is_main() -> bool:
    """True sur le processus principal (rang 0 ou single-GPU)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Forward MedVAE — gère les deux modes de retour de encode()
# ---------------------------------------------------------------------------


def medvae_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode → (reparamétrise si distribution) → decode.

    Retourne (recon, kl_loss).
    kl_loss = 0 si encode() retourne directement le sample z (cas le plus courant).
    Si encode() retourne (mean, logvar), applique la reparamétrique et calcule la KL.
    """
    z_out = model.encode(x)

    if isinstance(z_out, (tuple, list)) and len(z_out) >= 2:
        # encode() → (mean, logvar, ...)
        z_mean, z_logvar = z_out[0], z_out[1]
        z_logvar = z_logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * z_logvar)
        z = z_mean + std * torch.randn_like(std)
        kl = -0.5 * torch.mean(1.0 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    else:
        # encode() → sample z directement
        z = z_out
        kl = x.new_zeros(1).squeeze()

    recon = model.decode(z)

    # Ajustement spatial si l'architecture change la taille (au cas où)
    if recon.shape != x.shape:
        recon = F.interpolate(
            recon, size=x.shape[2:], mode="trilinear", align_corners=False
        )

    return recon, kl


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    lambda_l1: float,
    lambda_kl: float,
    max_batches: int = 50,
) -> float:
    """Calcule la loss de validation (L1 + λ·KL) sur au plus max_batches batches."""
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch.to(device)
        with torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            recon, kl = medvae_forward(model, x)
            loss = lambda_l1 * F.l1_loss(recon, x) + lambda_kl * kl
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


# ---------------------------------------------------------------------------
# Training principal
# ---------------------------------------------------------------------------


def train(
    cfg_path: str,
    env_arg: Optional[str] = None,
    resume_path: Optional[str] = None,
    data_root_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> None:

    # ── Config ──────────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, _load_env(env_arg))

    # Overrides CLI
    if data_root_override:
        cfg["data"]["data_root"] = data_root_override
    if output_dir_override:
        cfg["data"]["output_dir"] = output_dir_override

    data_root = Path(cfg["data"]["data_root"])
    output_dir = Path(cfg["data"]["output_dir"])
    split = cfg["data"].get("split", "retro_train")
    modalities = cfg["data"].get("modalities", ALL_MODALITIES)
    fields = cfg["data"].get("fields", ALL_FIELDS)
    patch_size = tuple(cfg["data"]["patch_size"])
    lo_pct = cfg["data"].get("percentile_lower", 0.5)
    hi_pct = cfg["data"].get("percentile_upper", 99.5)
    raw_ts = cfg["data"].get("target_spacing", [1.0, 1.0, 1.0])
    target_sp = tuple(float(v) for v in raw_ts) if raw_ts else None
    val_frac = cfg["data"].get("val_fraction", 0.1)

    medvae_name = cfg["medvae"]["model_name"]
    freeze_enc = cfg["medvae"].get("freeze_encoder", False)
    freeze_dec = cfg["medvae"].get("freeze_decoder", False)

    total_steps = cfg["train"]["total_steps"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 8)
    lr = float(cfg["train"]["lr"])
    lr_warmup = cfg["train"].get("lr_warmup_steps", 500)
    save_every = cfg["train"].get("save_every", 2000)
    print_every = cfg["train"].get("print_every", 100)
    use_amp = cfg["train"].get("use_amp", True)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    lambda_l1 = cfg["train"].get("lambda_l1", 1.0)
    lambda_kl = cfg["train"].get("lambda_kl", 0.0)

    # ── Distributed setup ───────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    if is_dist:
        dist.init_process_group(backend=cfg["train"].get("dist_backend", "nccl"))
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "weights").mkdir(exist_ok=True)
        print(f"\n{'=' * 70}")
        print(f" Fine-tuning MedVAE — {medvae_name}")
        print(f"{'=' * 70}")
        print(f"  Data root  : {data_root}")
        print(f"  Output     : {output_dir}")
        print(f"  Modalités  : {modalities}")
        print(f"  Champs     : {fields}")
        print(f"  Patch      : {patch_size}  (div. 4 requise)")
        print(f"  Steps      : {total_steps}  |  batch={batch_size}  |  lr={lr:.1e}")
        print(f"  Device     : {device}  |  world_size={world_size}")
        print(f"  AMP        : {use_amp}")
        if freeze_enc:
            print("  → Encoder gelé")
        if freeze_dec:
            print("  → Decoder gelé")
        print()

    # ── Dataset ─────────────────────────────────────────────────────────────
    full_ds = MRIxFieldsDataset(
        data_root=data_root,
        split=split,
        modalities=modalities,
        fields=fields,
        patch_size=patch_size,
        percentile_lower=lo_pct,
        percentile_upper=hi_pct,
        target_spacing=target_sp,
        is_training=True,
    )

    # Validation : Validating_prospective si disponible, sinon split du train
    try:
        val_ds = MRIxFieldsDataset(
            data_root=data_root,
            split="pro_val",
            modalities=modalities,
            fields=fields,
            patch_size=patch_size,
            percentile_lower=lo_pct,
            percentile_upper=hi_pct,
            target_spacing=target_sp,
            is_training=False,
        )
        if is_main():
            print("  → Validation : Validating_prospective")
    except FileNotFoundError:
        n_val = max(1, int(len(full_ds) * val_frac))
        indices = list(range(len(full_ds)))
        np.random.RandomState(42).shuffle(indices)
        val_ds = Subset(full_ds, indices[:n_val])
        full_ds = Subset(full_ds, indices[n_val:])
        if is_main():
            print(f"  → Validation : {n_val} volumes (split {val_frac:.0%} du train)")

    train_sampler = DistributedSampler(full_ds, shuffle=True) if is_dist else None
    train_loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, num_workers),
        pin_memory=True,
    )

    if is_main():
        n_train = (
            len(full_ds) if not isinstance(full_ds, Subset) else len(full_ds.indices)
        )
        print(
            f"  Train : {n_train} volumes → {len(train_loader)} batches/epoch\n"
            f"  Val   : {len(val_ds)} volumes\n"
        )

    # ── Modèle ──────────────────────────────────────────────────────────────
    # HF_HUB_OFFLINE = "1" : évite les appels réseau sur les nœuds compute JZ
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from medvae import MVAE
    except ImportError as e:
        raise RuntimeError(
            "Package 'medvae' non installé.\n"
            "  pip install medvae\n"
            "Sur Jean Zay, installer sur le nœud de login puis relancer."
        ) from e

    model = MVAE(model_name=medvae_name, modality="mri").to(device)

    # Gel sélectif (optionnel — fine-tuning complet par défaut)
    if freeze_enc:
        for name, p in model.named_parameters():
            if "encoder" in name or "quant_conv" in name:
                p.requires_grad = False
    if freeze_dec:
        for name, p in model.named_parameters():
            if "decoder" in name or "post_quant_conv" in name:
                p.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main():
        print(
            f"  Paramètres : {n_total / 1e6:.1f}M total, "
            f"{n_trainable / 1e6:.1f}M entraînables"
        )

    if is_dist:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_dist else model

    # ── Optimiseur ──────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    # Cosine decay avec linear warmup
    def _lr_lambda(step: int) -> float:
        if step < lr_warmup:
            return float(step) / max(lr_warmup, 1)
        progress = (step - lr_warmup) / max(total_steps - lr_warmup, 1)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # ── Reprise ──────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main():
            print(f"  → Reprise depuis step {start_step}: {resume_path}\n")

    weights_dir = output_dir / "weights"
    amp_dtype = torch.float16 if use_amp else torch.float32

    # ── Gestion auto-requeue SLURM (signal USR1 envoyé 120s avant timeout) ──
    _requeue_triggered = [False]

    def _handle_requeue(signum, frame):
        _requeue_triggered[0] = True
        if is_main():
            print(
                f"\n[SIGNAL USR1] Timeout imminent — sauvegarde d'urgence en cours..."
            )

    signal.signal(signal.SIGUSR1, _handle_requeue)

    # ── Boucle d'entraînement ────────────────────────────────────────────────
    step = start_step
    t0 = time.time()
    recent_loss = deque(maxlen=print_every)
    data_iter = iter(train_loader)

    if is_main():
        print(f"{'─' * 70}\n  Démarrage step {step + 1}/{total_steps}\n{'─' * 70}")

    while step < total_steps:
        model.train()

        # Rechargement du DataLoader en fin d'époque
        try:
            x = next(data_iter).to(device)
        except StopIteration:
            if is_dist and train_sampler is not None:
                train_sampler.set_epoch(step)
            data_iter = iter(train_loader)
            x = next(data_iter).to(device)

        # ── Forward + backward ──────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            recon, kl = medvae_forward(raw_model, x)
            l1_loss = F.l1_loss(recon, x)
            loss = lambda_l1 * l1_loss + lambda_kl * kl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        recent_loss.append(float(loss.item()))
        step += 1

        # ── Log périodique ──────────────────────────────────────────────────
        if is_main() and step % print_every == 0:
            avg = sum(recent_loss) / len(recent_loss)
            elapsed = (time.time() - t0) / 60
            eta = elapsed / step * (total_steps - step) if step > 0 else 0
            cur_lr = scheduler.get_last_lr()[0]
            mem_gb = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0
            )
            kl_str = f"  kl={float(kl.item()):.4f}" if lambda_kl > 0 else ""
            print(
                f"  [{step:6d}/{total_steps}]"
                f"  loss={avg:.4f}"
                f"  l1={float(l1_loss.item()):.4f}"
                f"{kl_str}"
                f"  lr={cur_lr:.2e}"
                f"  mem={mem_gb:.1f}GB"
                f"  t={elapsed:.0f}min  eta={eta:.0f}min"
            )

        # ── Checkpoint + validation périodiques ────────────────────────────
        save_now = (step % save_every == 0) or (step == total_steps)
        if (is_main() and save_now) or _requeue_triggered[0]:
            val_loss = _validate(
                raw_model, val_loader, device, use_amp, amp_dtype, lambda_l1, lambda_kl
            )
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            ckpt = {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "cfg_path": str(cfg_path),
            }

            ckpt_path = weights_dir / f"medvae_step_{step:06d}.pth"
            torch.save(ckpt, ckpt_path)
            print(
                f"  ✓ step {step:6d}  val_loss={val_loss:.4f}"
                + (" ← best" if is_best else "")
                + f"  → {ckpt_path.name}"
            )

            if is_best:
                torch.save(ckpt, weights_dir / "model_best.pth")

            # Auto-requeue : sauvegarder puis demander le requeue SLURM
            if _requeue_triggered[0]:
                torch.save(ckpt, weights_dir / "model_final.pth")
                print(f"  → Sauvegarde d'urgence OK. Requeue du job SLURM...")
                job_id = os.environ.get("SLURM_JOB_ID", "")
                if job_id:
                    os.system(f"scontrol requeue {job_id}")
                raise SystemExit(0)

    # ── Sauvegarde finale ────────────────────────────────────────────────────
    if is_main():
        ckpt_final = {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "cfg_path": str(cfg_path),
        }
        torch.save(ckpt_final, weights_dir / "model_final.pth")
        print(
            f"\n{'=' * 70}\n"
            f"  Fine-tuning terminé — {total_steps} steps\n"
            f"  Best val_loss : {best_val_loss:.4f}\n"
            f"  model_best    : {weights_dir / 'model_best.pth'}\n"
            f"  model_final   : {weights_dir / 'model_final.pth'}\n"
            f"{'=' * 70}\n"
        )

    if is_dist:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tuning MedVAE 3D sur Training_retrospective (toutes modalités).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Chemin vers le fichier de config YAML (ex: configs/medvae_finetune_all.yaml)",
    )
    p.add_argument(
        "--env",
        default=None,
        help="Environnement cible : 'local', 'jeanzay', ou chemin vers un .yaml",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="Reprendre depuis un checkpoint .pth (medvae_step_XXXXXX.pth ou model_best.pth)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Override du data_root défini dans la config/env",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override du output_dir défini dans la config/env",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(
        cfg_path=args.config,
        env_arg=args.env,
        resume_path=args.resume,
        data_root_override=args.data_root,
        output_dir_override=args.output_dir,
    )
