#!/usr/bin/env python3
"""MMFM v1 baseline for MRIxFields.

This script keeps the VAE unchanged and vectorizes its latent before the
flow model. Works with any MRIxFieldsVAE (spatial or vector latent format):
  - Spatial VAEs (AEKL, MedVAE, Pythae VAE/VQ-VAE): latent flattened via
    vae.to_vector() before MMFM, unflattened via vae.from_vector() before decode.
  - Vector VAEs (RHVAE): to_vector() is identity, from_vector() is identity.

Pipeline:
  1. 3D NIfTI volume
  2. VAE encode -> latent tensor (spatial or vector)
  3. vae.to_vector() -> flat vector
  4. Conditional flow matching in vector space (VectorMMFM residual MLP)
  5. Predict vector field
  6. vae.from_vector() -> latent tensor
  7. VAE decode -> 3D volume

Usage:
  # Entraînement
  PYTHONPATH=src python src/cfm/train_mmfm_3d.py \\
      --config configs/mmfm3d_medvae_multimodal.yaml --env local

  # Inférence (toutes les combinaisons source → cible)
  PYTHONPATH=src python src/cfm/train_mmfm_3d.py --mode infer \\
      --config configs/mmfm3d_medvae_multimodal.yaml --env local \\
      --checkpoint outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v1/weights/model_final.pth \\
      --input_dir /path/to/nifti/ --output_dir outputs/predictions/mmfm/ \\
      --source_field 0.1T --source_modality T1W \\
      --target_field 7T   --target_modality T1W

  # Inférence single-volume
  PYTHONPATH=src python src/cfm/train_mmfm_3d.py --mode infer \\
      --config configs/mmfm3d_medvae_multimodal.yaml --env local \\
      --checkpoint outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v1/weights/model_final.pth \\
      --input_volume /path/to/sub_T1W_0.1T_0001.nii.gz \\
      --output_dir outputs/predictions/mmfm/ \\
      --source_field 0.1T --source_modality T1W \\
      --target_field 7T   --target_modality T1W
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.dataset import MultiModalNIfTILatentDataset
from common.distributed import is_main_process, EMAModel
from common.io import (
    DOMAINS,
    MODALITIES,
    DOMAIN_TO_IDX,
    MODALITY_TO_IDX,
    adjust_affine_for_crop_pad,
    load_nifti_volume,
    resample_volume,
)
from models.vae_loader import load_vae

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from cfm.mmfm_vectorized import LatentVectorizer, VectorMMFM


def _flat_class(mod_idx: int, field_idx: int, n_fields: int) -> int:
    return mod_idx * n_fields + field_idx


def _infer_latent_shape(vae, volume_size: Tuple[int, int, int], device: torch.device) -> Tuple[int, ...]:
    dummy = torch.zeros(1, 1, *volume_size, device=device)
    with torch.no_grad():
        z = vae.encode(dummy)
    return tuple(int(v) for v in z.shape[1:])


def build_vector_mmfm(cfg: dict, latent_dim: int, n_classes: int) -> VectorMMFM:
    m = cfg["model"]
    return VectorMMFM(
        latent_dim=latent_dim,
        num_classes=n_classes,
        hidden_dim=int(m.get("hidden_dim", 1024)),
        depth=int(m.get("num_blocks", 4)),
        time_embed_dim=int(m.get("time_embed_dim", 256)),
        class_embed_dim=int(m.get("class_embed_dim", 128)),
        dropout=float(m.get("dropout", 0.0)),
    )


def _save_checkpoint(
    path: Path,
    step: int,
    model: torch.nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    use_scaler: bool,
    cfg_path: str,
    latent_shape: Tuple[int, ...],
) -> None:
    torch.save(
        {
            "iter": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "cfg_path": str(cfg_path),
            "latent_shape": latent_shape,
        },
        path,
    )


def _make_infinite(loader):
    while True:
        yield from loader


def train(cfg_path: str, env_path: Optional[str] = None, resume: Optional[str] = None):
    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))
    if resume is not None:
        cfg["resume"] = resume

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    data_root = data_cfg.get("data_root")
    if data_root is None:
        raise RuntimeError("data_root requis dans config/env")

    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)

    output_dir = Path(data_cfg["output_dir"])
    split = data_cfg.get("split", "retro_train")
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    max_per_class = data_cfg.get("max_volumes_per_class", None)

    raw_vs = data_cfg.get("volume_size", None)
    if raw_vs is None:
        raise RuntimeError("volume_size est requis pour la baseline vectorisée MMFM v1.")
    volume_size = tuple(int(v) for v in raw_vs)

    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    total_iters = int(train_cfg.get("total_iters", 10000))
    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 4))
    lr = float(train_cfg.get("lr", 1e-4))
    sigma = float(train_cfg.get("sigma", 0.0))
    ot_method = train_cfg.get("ot_method", "exact")
    save_every = int(train_cfg.get("save_every", 2000))
    print_every = int(train_cfg.get("print_every", 100))
    use_amp = bool(train_cfg.get("use_amp", True))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    num_targets_per_step = int(train_cfg.get("num_targets_per_step", 2))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        (output_dir / "weights").mkdir(parents=True, exist_ok=True)
        print(f"Output dir : {output_dir}")
        print(f"World size : {world_size} | Device : {device}")

    ds = MultiModalNIfTILatentDataset(
        data_root=Path(data_root),
        split=split,
        modalities=modalities,
        fields=fields,
        percentile_lower=p_lo,
        percentile_upper=p_hi,
        max_per_class=max_per_class,
        target_spacing=target_spacing,
        volume_size=volume_size,
    )

    class_loaders: Dict[int, any] = {}
    n_classes = len(modalities) * len(fields)

    for c_idx in range(n_classes):
        class_indices = [i for i, (_, _, _, c) in enumerate(ds.samples) if c == c_idx]
        if not class_indices:
            continue
        subset = torch.utils.data.Subset(ds, class_indices)
        sampler = DistributedSampler(subset, shuffle=True) if is_distributed else None
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        class_loaders[c_idx] = _make_infinite(loader)

    available_classes = sorted(class_loaders.keys())
    if len(available_classes) < 2:
        raise RuntimeError("Il faut au moins 2 classes modalite/champ non vides.")

    vae = load_vae(cfg, device)
    latent_shape = _infer_latent_shape(vae, volume_size, device)
    # Calculer latent_dim depuis latent_shape réel (volume_size-dependent).
    # NE PAS utiliser vae.vector_dim : il est inféré sur un dummy (32³) et peut
    # différer si volume_size n'est pas un cube.
    latent_dim = int(torch.tensor(latent_shape).prod().item())

    vectorizer = LatentVectorizer(latent_shape)

    mmfm = build_vector_mmfm(cfg, latent_dim, n_classes).to(device)
    if is_distributed:
        mmfm = DDP(mmfm, device_ids=[local_rank])
    raw_mmfm = mmfm.module if is_distributed else mmfm

    ema = EMAModel(raw_mmfm, decay=ema_decay)
    optimizer = torch.optim.AdamW(mmfm.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(step: int) -> float:
        decay_start = total_iters // 2
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(total_iters - decay_start, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError("amp_dtype doit être 'bf16' ou 'fp16'.")
    use_scaler = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if ot_method == "exact":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    else:
        FM = ConditionalFlowMatcher(sigma=sigma)

    start_iter = 0
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        raw_mmfm.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        if "scaler" in state and use_scaler:
            scaler.load_state_dict(state["scaler"])
        start_iter = state.get("iter", 0) + 1
        if is_main_process():
            print(f"Reprise depuis iter {start_iter}: {resume_path}")

    if is_main_process():
        n_params = sum(p.numel() for p in raw_mmfm.parameters() if p.requires_grad)
        print(
            f"Vector MMFM: {n_params/1e6:.1f}M params | classes={n_classes} | "
            f"latent_shape={latent_shape} | flat_dim={latent_dim}"
        )
        print(
            f"Training MMFM v1: iters={total_iters} batch={batch_size} "
            f"targets/step={num_targets_per_step} amp={use_amp} dtype={amp_dtype_name}"
        )

    weights_dir = output_dir / "weights"
    metrics_path = output_dir / "train_metrics.jsonl"
    t0 = time.time()
    last_log_t = t0
    recent_losses: List[float] = []
    mmfm.train()

    for step in range(start_iter, total_iters):
        # Synchroniser le choix src/tgt entre tous les ranks DDP pour garantir
        # que chaque GPU travaille sur la même paire de classes au même step.
        if is_distributed:
            if dist.get_rank() == 0:
                src_idx = torch.tensor(
                    [random.randrange(len(available_classes))],
                    dtype=torch.long, device=device,
                )
            else:
                src_idx = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(src_idx, src=0)
            src_class = available_classes[src_idx.item()]

            tgt_candidates = [c for c in available_classes if c != src_class]
            k = min(num_targets_per_step, len(tgt_candidates))
            if dist.get_rank() == 0:
                tgt_idx_t = torch.tensor(
                    random.sample(range(len(tgt_candidates)), k),
                    dtype=torch.long, device=device,
                )
            else:
                tgt_idx_t = torch.zeros(k, dtype=torch.long, device=device)
            dist.broadcast(tgt_idx_t, src=0)
            tgt_classes = [tgt_candidates[i] for i in tgt_idx_t.tolist()]
        else:
            src_class = random.choice(available_classes)
            tgt_candidates = [c for c in available_classes if c != src_class]
            k = min(num_targets_per_step, len(tgt_candidates))
            tgt_classes = random.sample(tgt_candidates, k=k)

        src_batch = next(class_loaders[src_class])
        src_x = src_batch[0].to(device)

        optimizer.zero_grad(set_to_none=True)
        step_losses = []

        for tgt_class in tgt_classes:
            tgt_batch = next(class_loaders[tgt_class])
            tgt_x = tgt_batch[0].to(device)

            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                z_src = vae.encode(src_x)
                z_tgt = vae.encode(tgt_x)

            # Phase F: API canonique to_vector (fonctionne pour spatial et vector)
            z_src_vec = vae.to_vector(z_src)
            z_tgt_vec = vae.to_vector(z_tgt)
            t_batch, z_t, ut = FM.sample_location_and_conditional_flow(z_src_vec, z_tgt_vec)
            t_vec = t_batch.to(device).float().reshape(z_src_vec.shape[0], -1).squeeze(-1)
            y_tgt = torch.full((z_src_vec.shape[0],), tgt_class, dtype=torch.long, device=device)

            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                v_t = raw_mmfm(z_t, z_src_vec, t_vec, y_tgt)
                loss = F.mse_loss(v_t, ut) / float(k)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_losses.append(float(loss.item() * k))

        if use_scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_mmfm.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_mmfm.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        ema.update(raw_mmfm)

        mean_step_loss = float(np.mean(step_losses)) if step_losses else 0.0
        recent_losses.append(mean_step_loss)
        if len(recent_losses) > print_every:
            recent_losses.pop(0)

        if is_main_process() and (step + 1) % print_every == 0:
            avg_recent = float(np.mean(recent_losses))
            elapsed = time.time() - t0
            win_dt = time.time() - last_log_t
            it_s = print_every / max(win_dt, 1e-9)
            eta_s = (total_iters - step - 1) / max(it_s, 1e-9)
            lr_cur = scheduler.get_last_lr()[0]
            mem_gb = (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"[{step+1:6d}/{total_iters}] loss={avg_recent:.4f} grad={float(grad_norm):.2f} "
                f"lr={lr_cur:.2e} src={src_class} tgts={tgt_classes} speed={it_s:.2f} it/s "
                f"eta={eta_s/3600:.2f}h t={elapsed/60:.1f}min mem={mem_gb:.1f}GB"
            )
            # ── Écriture train_metrics.jsonl ──────────────────────────────
            record = {
                "iter": step + 1,
                "loss": round(avg_recent, 6),
                "grad_norm": round(float(grad_norm), 4),
                "lr": round(lr_cur, 8),
                "speed_it_s": round(it_s, 3),
                "elapsed_s": round(elapsed, 1),
                "mem_gb": round(mem_gb, 2),
            }
            with open(metrics_path, "a") as _mf:
                _mf.write(json.dumps(record) + "\n")
            last_log_t = time.time()

        if is_main_process() and (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step+1}.pth"
            _save_checkpoint(
                ckpt_path,
                step,
                raw_mmfm,
                ema,
                optimizer,
                scaler,
                use_scaler,
                cfg_path,
                latent_shape,
            )
            print(f"  -> Checkpoint: {ckpt_path}")

    if is_main_process():
        final_path = weights_dir / "model_final.pth"
        _save_checkpoint(
            final_path,
            total_iters - 1,
            raw_mmfm,
            ema,
            optimizer,
            scaler,
            use_scaler,
            cfg_path,
            latent_shape,
        )
        print(f"Training MMFM terminé. Modèle final: {final_path}")

    if is_distributed:
        dist.destroy_process_group()


# ===========================================================================
# Inference
# ===========================================================================


@torch.no_grad()
def _euler_integrate_vector(
    mmfm: VectorMMFM,
    z_src_vec: Tensor,
    tgt_class: int,
    n_steps: int,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Intégration Euler du champ vectoriel MMFM : z_src_vec → z_tgt_vec.

    Args:
        mmfm: VectorMMFM chargé et en mode eval.
        z_src_vec: Vecteur latent source, shape (1, D).
        tgt_class: Indice de classe cible (mod_idx * n_fields + field_idx).
        n_steps: Nombre de pas d'intégration Euler.
        device: Périphérique de calcul.
        use_amp: Activer l'autocast AMP.
        amp_dtype: Type de données AMP.

    Returns:
        Vecteur latent prédit, shape (1, D).
    """
    dt = 1.0 / n_steps
    z = z_src_vec.clone().to(device).float()
    z_src = z_src_vec.to(device).float()
    y = torch.tensor([tgt_class], dtype=torch.long, device=device)

    for step_i in range(n_steps):
        t_val = step_i * dt
        t_vec = torch.tensor([t_val], dtype=torch.float32, device=device)
        with torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            vt = mmfm(z, z_src, t_vec, y)
        z = z + dt * vt.float()

    return z


def infer(
    cfg_path: str,
    checkpoint: str,
    output_dir: str,
    source_field: str,
    source_modality: str,
    target_field: str,
    target_modality: str,
    env_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    input_volume: Optional[str] = None,
    n_steps: Optional[int] = None,
    use_ema: bool = True,
) -> None:
    """Inférence MMFM : encode → intégration ODE vectorielle → decode.

    Accepte soit un répertoire de volumes NIfTI (input_dir) soit un volume
    unique (input_volume).

    Args:
        cfg_path: Chemin vers le fichier de configuration YAML.
        checkpoint: Chemin vers le checkpoint MMFM (.pth).
        output_dir: Répertoire de sortie pour les volumes prédits.
        source_field: Champ magnétique source (ex. "0.1T").
        source_modality: Modalité source (ex. "T1W").
        target_field: Champ magnétique cible (ex. "7T").
        target_modality: Modalité cible (ex. "T1W").
        env_path: Chemin vers l'env YAML (local/remote).
        input_dir: Répertoire contenant des fichiers .nii.gz.
        input_volume: Chemin vers un volume .nii.gz unique.
        n_steps: Nombre de pas Euler (défaut: config inference.n_steps ou 20).
        use_ema: Utiliser les poids EMA si disponibles.
    """
    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    use_amp = bool(train_cfg.get("use_amp", True))
    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
        use_amp = False

    n_steps = n_steps or cfg.get("inference", {}).get("n_steps", 20)
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    raw_vs = data_cfg.get("volume_size", None)
    volume_size = tuple(int(v) for v in raw_vs) if raw_vs else None
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(modalities) * n_fields

    # Valider source/cible
    for name, val, lst in [
        ("source_field", source_field, fields),
        ("target_field", target_field, fields),
        ("source_modality", source_modality, modalities),
        ("target_modality", target_modality, modalities),
    ]:
        if val not in lst:
            raise ValueError(f"{name}='{val}' non présent dans la config ({lst})")

    tgt_mod_idx = modalities.index(target_modality)
    tgt_field_idx = fields.index(target_field)
    tgt_class = _flat_class(tgt_mod_idx, tgt_field_idx, n_fields)

    print(
        f"Inférence MMFM v1 : {source_modality}@{source_field} → "
        f"{target_modality}@{target_field}  |  tgt_class={tgt_class}  |  {n_steps} steps Euler"
    )

    # ── VAE ──────────────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    latent_shape = _infer_latent_shape(vae, volume_size, device)
    # Calculer latent_dim depuis latent_shape réel (volume_size-dependent).
    latent_dim = int(torch.tensor(latent_shape).prod().item())

    print(f"  VAE: latent_shape={latent_shape}  flat_dim={latent_dim}")

    # ── MMFM ─────────────────────────────────────────────────────────────────
    mmfm = build_vector_mmfm(cfg, latent_dim, n_classes).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Vérifier cohérence du checkpoint
    saved_latent_shape = ckpt.get("latent_shape", None)
    if saved_latent_shape is not None and tuple(saved_latent_shape) != latent_shape:
        raise RuntimeError(
            f"Incohérence latent_shape : checkpoint={saved_latent_shape}, "
            f"config courante={latent_shape}. "
            "Utilisez la même config que lors de l'entraînement."
        )

    # Charger poids EMA si disponibles
    loaded_from = "model"
    if use_ema and "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        shadow = ema_state.get("shadow_params", None)
        if shadow is not None:
            mmfm.load_state_dict(shadow)
            loaded_from = "ema.shadow_params"
        else:
            mmfm.load_state_dict(ckpt["model"])
    else:
        mmfm.load_state_dict(ckpt["model"])

    mmfm.eval()
    trained_at = ckpt.get("iter", "?")
    print(f"  MMFM chargé (clé: {loaded_from}, iter={trained_at}) depuis {checkpoint}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collecter les fichiers à traiter
    if input_volume is not None:
        input_files = [Path(input_volume)]
    elif input_dir is not None:
        input_files = sorted(Path(input_dir).glob("*.nii.gz"))
        if not input_files:
            raise FileNotFoundError(f"Aucun fichier .nii.gz dans {input_dir}")
    else:
        raise ValueError("Fournir --input_dir ou --input_volume.")

    print(f"  {len(input_files)} volume(s) à prédire → {out_dir}")

    for nii_path in input_files:
        t_start = time.time()

        # ── Chargement & prétraitement ────────────────────────────────────
        vol, _ = load_nifti_volume(
            nii_path,
            target_spacing=target_spacing,
            volume_size=volume_size,
            normalize=True,
            lo_pct=p_lo,
            hi_pct=p_hi,
        )

        # Affine corrigé pour crop/pad
        img_nib = nib.load(str(nii_path))
        orig_spacing = np.abs(np.diag(img_nib.affine)[:3])
        orig_shape = np.array(img_nib.shape[:3])
        if target_spacing is not None:
            resampled_arr = resample_volume(
                np.zeros(orig_shape.tolist(), dtype=np.float32),
                orig_spacing,
                target_spacing,
            )
            resampled_shape = resampled_arr.shape
        else:
            resampled_shape = None

        out_affine = adjust_affine_for_crop_pad(
            img_nib.affine.copy().astype(float),
            orig_shape,
            volume_size,
            resampled_shape,
            target_spacing,
            orig_spacing,
        )

        vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)

        # ── Encode ───────────────────────────────────────────────────────
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            z_src = vae.encode(vol_tensor)

        z_src_vec = vae.to_vector(z_src).float()  # (1, D)

        # ── Intégration Euler vectorielle ────────────────────────────────
        z_tgt_vec = _euler_integrate_vector(
            mmfm, z_src_vec, tgt_class, n_steps, device, use_amp, amp_dtype
        )

        # ── Decode ───────────────────────────────────────────────────────
        # Utiliser LatentVectorizer pour le reshape (from_vector suppose un cube parfait)
        z_tgt = LatentVectorizer(latent_shape).unflatten(z_tgt_vec)
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            recon = vae.decode(z_tgt)

        pred_vol = recon.squeeze().cpu().float().numpy()
        pred_vol = (np.clip(pred_vol, -1.0, 1.0) + 1.0) / 2.0  # [-1,1] → [0,1]

        # ── Sauvegarde ───────────────────────────────────────────────────
        stem = nii_path.name.replace(".nii.gz", "")
        out_name = f"{stem}_{target_modality}_{target_field}_mmfm.nii.gz"
        out_path = out_dir / out_name
        nib.save(nib.Nifti1Image(pred_vol, out_affine), str(out_path))

        elapsed = time.time() - t_start
        print(f"  {nii_path.name} → {out_name}  ({elapsed:.1f}s)")

    print(f"\nInférence MMFM terminée. Prédictions dans : {out_dir}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMFM v1 vectorized baseline — train & infer")
    p.add_argument("--mode", default="train", choices=["train", "infer"],
                   help="'train' : entraînement | 'infer' : inférence sur volumes NIfTI")
    p.add_argument("--config", required=True, help="Chemin vers le YAML de configuration")
    p.add_argument("--env", default=None, help="Env YAML (local / remote / chemin)")
    # Train-only
    p.add_argument("--resume", default=None, help="[train] Reprendre depuis ce checkpoint")
    # Infer-only
    p.add_argument("--checkpoint", default=None,
                   help="[infer] Chemin vers le checkpoint MMFM (.pth)")
    p.add_argument("--input_dir", default=None,
                   help="[infer] Répertoire de volumes .nii.gz source")
    p.add_argument("--input_volume", default=None,
                   help="[infer] Volume .nii.gz unique source")
    p.add_argument("--output_dir", default=None,
                   help="[infer] Répertoire de sortie des prédictions")
    p.add_argument("--source_field", default=None,
                   help="[infer] Champ magnétique source (ex. '0.1T')")
    p.add_argument("--source_modality", default=None,
                   help="[infer] Modalité source (ex. 'T1W')")
    p.add_argument("--target_field", default=None,
                   help="[infer] Champ magnétique cible (ex. '7T')")
    p.add_argument("--target_modality", default=None,
                   help="[infer] Modalité cible (ex. 'T1W')")
    p.add_argument("--n_steps", type=int, default=None,
                   help="[infer] Nombre de pas Euler (défaut: 20)")
    p.add_argument("--no_ema", action="store_true",
                   help="[infer] Ignorer les poids EMA et utiliser les poids bruts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.mode == "train":
        train(args.config, env_path=args.env, resume=args.resume)
    else:
        missing = [
            name for name, val in [
                ("--checkpoint", args.checkpoint),
                ("--output_dir", args.output_dir),
                ("--source_field", args.source_field),
                ("--source_modality", args.source_modality),
                ("--target_field", args.target_field),
                ("--target_modality", args.target_modality),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"Mode infer : arguments manquants : {', '.join(missing)}\n"
                "Fournir aussi --input_dir ou --input_volume."
            )
        if args.input_dir is None and args.input_volume is None:
            raise ValueError("Mode infer : fournir --input_dir ou --input_volume.")

        infer(
            cfg_path=args.config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            source_field=args.source_field,
            source_modality=args.source_modality,
            target_field=args.target_field,
            target_modality=args.target_modality,
            env_path=args.env,
            input_dir=args.input_dir,
            input_volume=args.input_volume,
            n_steps=args.n_steps,
            use_ema=not args.no_ema,
        )
