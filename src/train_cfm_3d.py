#!/usr/bin/env python3
"""
OT-CFM 3D en espace latent — Traduction de contraste IRM 3D volumétrique.

Architecture :
  - VAE 3D pré-entraîné (AutoencoderKL MONAI) → espace latent 8× compressé
  - OT-CFM en espace latent : UNet 3D conditionné sur modalité source + domaine cible
  - Input du UNet : cat(z_t, z_src)  →  (B, 2*C_lat, H', W', D')
  - Output : champ vectoriel de vitesse → (B, C_lat, H', W', D')
  - Entraîné avec ExactOT conditional flow matching (torchcfm)

Pipeline d'inférence :
  T1W volume → VAE encode → z_src → Euler/DOPRI5 ODE (t: 0→1) → z_tgt → VAE decode → T2W volume

Usage :
  # Single-GPU
  python src/train_cfm_3d.py --config configs/cfm3d_latent_T1W.yaml --env local

  # Multi-GPU (torchrun DDP)
  torchrun --nproc_per_node=4 src/train_cfm_3d.py \\
      --config configs/cfm3d_latent_T1W.yaml --env jeanzay

  # Inférence
  python src/train_cfm_3d.py --mode infer \\
      --config configs/cfm3d_latent_T1W.yaml \\
      --checkpoint outputs/cfm3d/.../weights/model_final.pth \\
      --input_dir /data/T1W/0.1T/ \\
      --output_dir /data/predictions/cfm3d/ \\
      --source_domain 0.1T --target_domain 7T
"""

import argparse
import inspect
import os
import random
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as scipy_zoom
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    # MONAI récent (intégration generative dans monai.networks)
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
except ImportError:
    try:
        # MONAI avec sous-module generative
        from monai.generative.networks.nets import AutoencoderKL, DiffusionModelUNet
    except ImportError:
        try:
        # MONAI Generative package (namespace séparé)
            from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
        except ImportError as e:
            raise ImportError(
                "AutoencoderKL/DiffusionModelUNet introuvables. Essayez : "
                "monai>=1.3, monai[generation], ou monai-generative"
            ) from e

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAINS: List[str] = ["0.1T", "1.5T", "3T", "5T", "7T"]
DOMAIN_TO_IDX: Dict[str, int] = {d: i for i, d in enumerate(DOMAINS)}
NUM_DOMAINS = len(DOMAINS)


# ===========================================================================
# Utilitaire : rééchantillonnage isotrope
# ===========================================================================

def _resample_volume(
    vol: np.ndarray,
    original_spacing,
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Rééchantillonne un volume 3D vers target_spacing (mm).

    Exemple : 364×436×364 @ 0.5mm → 182×218×182 @ 1mm (8× moins de voxels).
    """
    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt  = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol
    return scipy_zoom(vol, factors, order=1).astype(np.float32)

_SPLIT_ABBR_TO_DIR = {
    "retro_train": "Training_retrospective",
    "pro_train":   "Training_prospective",
    "pro_val":     "Validating_prospective",
    "pro_test":    "Testing_prospective",
}


# ===========================================================================
# Env / Path resolution
# ===========================================================================

def _load_env(env_arg: Optional[str]) -> Optional[dict]:
    if env_arg is None:
        return None
    env_path = env_arg if env_arg.endswith(".yaml") else f"configs/env/{env_arg}.yaml"
    if not os.path.isabs(env_path):
        if os.path.exists(env_path):
            env_path = os.path.abspath(env_path)
        else:
            project_root = Path(__file__).parent.parent
            candidate = project_root / env_path
            if candidate.exists():
                env_path = str(candidate)
    with open(env_path) as f:
        raw = yaml.safe_load(f)
    return {k: os.path.expandvars(str(v)) for k, v in raw.items()}


def _resolve_paths(cfg: dict, env: Optional[dict]) -> dict:
    if env is None:
        return cfg
    output_root = Path(env["output_root"])
    data = cfg.setdefault("data", {})
    if "output_subdir" in data:
        data["output_dir"] = str(output_root / data["output_subdir"])
    if "data_root" in env:
        data.setdefault("data_root", env["data_root"])
    # VAE checkpoint depuis env si non spécifié dans config
    vae_cfg = cfg.setdefault("vae", {})
    if "vae_root" in env and not vae_cfg.get("checkpoint"):
        subdir = cfg["data"].get("output_subdir", "").replace("cfm3d", "vae3d")
        vae_cfg["checkpoint"] = str(Path(env["output_root"]) / subdir.replace("cfm3d_latent", "vae3d") / "weights" / "model_best.pth")
    return cfg


# ===========================================================================
# Dataset
# ===========================================================================

class NIfTILatentDataset(Dataset):
    """Charge des volumes NIfTI, les normalise et les retourne entiers pour
    l'encodage VAE on-the-fly.

    Retourne : (volume_tensor, domain_idx)
    où volume_tensor est (1, H, W, D) dans [-1, 1].

    Note : L'encodage VAE est fait dans la boucle d'entraînement (sur GPU),
    pas ici (pour bénéficier du cache de gradient checkpointing).
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modality: str,
        domains: List[str],
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        max_per_domain: Optional[int] = None,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
    ):
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.target_spacing = target_spacing
        self.volume_size = volume_size

        split_dir = _SPLIT_ABBR_TO_DIR.get(split, split)
        self.samples: List[Tuple[Path, int]] = []
        for domain in domains:
            domain_dir = Path(data_root) / split_dir / modality / domain
            files = sorted(domain_dir.glob("*.nii.gz"))
            if max_per_domain is not None:
                files = files[:max_per_domain]
            if not files:
                print(f"  [WARN] Aucun volume dans {domain_dir}")
            for f in files:
                self.samples.append((f, DOMAIN_TO_IDX[domain]))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucun volume NIfTI dans {data_root}/{split_dir}/{modality}/"
            )
        print(
            f"  NIfTILatentDataset: {len(self.samples)} volumes"
            f" ({modality}, {split})"
        )

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        lo = np.percentile(vol, self.percentile_lower)
        hi = np.percentile(vol, self.percentile_upper)
        vol = np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        return vol * 2.0 - 1.0  # → [-1, 1]

    def __len__(self) -> int:
        return len(self.samples)

    def _center_crop_or_pad(self, vol: np.ndarray) -> np.ndarray:
        if self.volume_size is None:
            return vol

        th, tw, td = self.volume_size
        h, w, d = vol.shape

        # Pad si le volume est trop petit sur un axe.
        ph = max(0, th - h)
        pw = max(0, tw - w)
        pd = max(0, td - d)
        if ph > 0 or pw > 0 or pd > 0:
            vol = np.pad(
                vol,
                [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2)],
                mode="reflect",
            )
            h, w, d = vol.shape

        sh = max((h - th) // 2, 0)
        sw = max((w - tw) // 2, 0)
        sd = max((d - td) // 2, 0)
        return vol[sh: sh + th, sw: sw + tw, sd: sd + td]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, domain_idx = self.samples[idx]
        img_nib = nib.load(str(path))
        if self.target_spacing is not None:
            spacing = np.abs(np.diag(img_nib.affine)[:3])
            vol = img_nib.get_fdata(dtype=np.float32)
            vol = _resample_volume(vol, spacing, self.target_spacing)
        else:
            vol = img_nib.get_fdata(dtype=np.float32)
        vol = self._normalize(vol)
        vol = self._center_crop_or_pad(vol)
        tensor = torch.from_numpy(vol).unsqueeze(0)  # (1, H, W, D)
        return tensor, domain_idx


def _make_infinite(loader: DataLoader):
    while True:
        yield from loader


# ===========================================================================
# Build VAE (chargement)
# ===========================================================================

def _build_vae_from_config(vae_cfg: dict) -> AutoencoderKL:
    """Instancie l'AutoencoderKL 3D depuis une config YAML."""
    m = vae_cfg["model"]
    sig = inspect.signature(AutoencoderKL.__init__).parameters
    kwargs = {
        "spatial_dims": m["spatial_dims"],
        "in_channels": m["in_channels"],
        "out_channels": m["out_channels"],
        "latent_channels": m["latent_channels"],
        "num_res_blocks": m["num_res_blocks"],
        "norm_num_groups": m["norm_num_groups"],
        "attention_levels": tuple(m["attention_levels"]),
        "with_encoder_nonlocal_attn": m["with_encoder_nonlocal_attn"],
        "with_decoder_nonlocal_attn": m["with_decoder_nonlocal_attn"],
    }

    channels = tuple(m["channels"])
    if "channels" in sig:
        kwargs["channels"] = channels
    elif "num_channels" in sig:
        kwargs["num_channels"] = channels

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig}
    return AutoencoderKL(**filtered_kwargs)


def _load_maisi_autoencoder(cache_dir: Optional[str] = None) -> AutoencoderKL:
    """Télécharge et charge le VAE MAISI depuis le Model Zoo MONAI.

    Le bundle 'maisi_ct_generative' inclut un AutoencoderKL 3D pré-entraîné
    sur ~40k volumes CT (compression 8×, latent_channels=4). Les poids peuvent
    être utilisés comme initialisation pour l'IRM.

    Référence MAISI : https://arxiv.org/abs/2409.11169

    Note : si le téléchargement échoue (pas d'internet, etc.), utilisez
    'vae.source: local' avec un checkpoint pré-entraîné local.
    """
    try:
        from monai.bundle import download
    except ImportError:
        raise ImportError("MONAI bundle API non disponible. Mettez à jour MONAI >= 1.3.")

    bundle_dir = Path(cache_dir or os.path.expanduser("~/.cache/monai/bundles"))
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = "maisi_ct_generative"

    print(f"  Téléchargement du bundle MAISI '{bundle_name}' dans {bundle_dir} ...")
    download(name=bundle_name, source="monai", bundle_dir=str(bundle_dir))

    # Chemin des poids de l'autoencoder dans le bundle MAISI
    autoencoder_ckpt = bundle_dir / bundle_name / "models" / "autoencoder.pt"
    if not autoencoder_ckpt.exists():
        # Chercher dans les fichiers disponibles
        models_dir = bundle_dir / bundle_name / "models"
        candidates = sorted(models_dir.glob("*autoencoder*")) if models_dir.exists() else []
        if not candidates:
            raise FileNotFoundError(
                f"Autoencoder MAISI introuvable dans {models_dir}.\n"
                f"Fichiers disponibles : {list(models_dir.iterdir()) if models_dir.exists() else 'dir absent'}"
            )
        autoencoder_ckpt = candidates[0]

    # Config MAISI pour AutoencoderKL 3D
    # Architecture standard MAISI : latent_channels=4, compression 8×
    maisi_model_cfg = {
        "model": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "channels": [64, 128, 256, 512],
            "num_res_blocks": 2,
            "norm_num_groups": 32,
            "attention_levels": [False, False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        }
    }
    vae = _build_vae_from_config(maisi_model_cfg)
    state = torch.load(str(autoencoder_ckpt), map_location="cpu", weights_only=False)
    # Le checkpoint MAISI peut avoir différentes clés
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]
    vae.load_state_dict(state, strict=True)
    print(f"  VAE MAISI chargé depuis : {autoencoder_ckpt}")
    return vae


def load_vae(cfg: dict, device: torch.device) -> AutoencoderKL:
    """Charge le VAE 3D selon cfg['vae']['source'].

    source options :
      'local'       : checkpoint pré-entraîné local (vae.checkpoint, défaut)
      'maisi'       : autoencoder MAISI (téléchargé depuis MONAI Model Zoo)
      'random'      : poids aléatoires (pour débugage uniquement)

    Note NVIDIA NV-Generate-MR-Brain :
      Le repo HuggingFace (nvidia/NV-Generate-MR-Brain) ne contient que le
      diffusion UNet (diff_unet_3d_rflow-mr-brain_v0.pt, 2.17 GB). La VAE
      n'est pas disponible séparément. Utilisez 'maisi' ou 'local'.
    """
    vae_source = cfg["vae"].get("source", "local")

    if vae_source == "maisi":
        cache_dir = cfg["vae"].get("maisi_cache_dir", None)
        vae = _load_maisi_autoencoder(cache_dir)
    else:
        # 'local' ou 'random'
        vae_config_path = cfg["vae"].get("vae_config", "configs/vae3d_T1W.yaml")
        if not os.path.isabs(vae_config_path):
            project_root = Path(__file__).parent.parent
            vae_config_path = str(project_root / vae_config_path)

        with open(vae_config_path) as f:
            vae_cfg = yaml.safe_load(f)

        vae = _build_vae_from_config(vae_cfg)

        if vae_source != "random":
            ckpt_path = cfg["vae"].get("checkpoint", "")
            if ckpt_path and Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                vae.load_state_dict(state["model"])
                print(f"  VAE chargé depuis : {ckpt_path}")
            else:
                print(f"  [WARN] VAE checkpoint introuvable : '{ckpt_path}' — poids aléatoires.")

    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ===========================================================================
# Build UNet 3D pour le flow matching en espace latent
# ===========================================================================

def build_unet_3d(cfg: dict, latent_channels: int) -> DiffusionModelUNet:
    """UNet 3D conditionné sur le temps et le domaine cible.

    in_channels = 2 * latent_channels  (cat z_t + z_src)
    out_channels = latent_channels      (champ de vitesse)
    num_class_embeds = NUM_DOMAINS      (label du domaine cible)
    """
    m = cfg["model"]
    channel_mult = tuple(m.get("channel_mult", [1, 2, 4]))
    base_channels = m.get("model_channels", 128)
    channels = tuple(base_channels * c for c in channel_mult)

    _sig = inspect.signature(DiffusionModelUNet.__init__).parameters
    _ch_kwarg = "num_channels" if "num_channels" in _sig else "channels"

    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2 * latent_channels,         # cat(z_t, z_src)
        out_channels=latent_channels,
        **{_ch_kwarg: channels},
        attention_levels=tuple(m.get("attention_levels", [False, True, True])),
        num_res_blocks=m.get("num_res_blocks", 2),
        num_head_channels=m.get("num_head_channels", 64),
        norm_num_groups=m.get("norm_num_groups", 32),
        use_flash_attention=m.get("use_flash_attention", False),
        num_class_embeds=NUM_DOMAINS,            # conditioning sur domaine cible
        with_conditioning=False,                 # pas de cross-attention texte
        resblock_updown=True,
    )


# ===========================================================================
# EMA (Exponential Moving Average)
# ===========================================================================

class EMAModel:
    """Copie EMA des poids du modèle pour une meilleure généralisation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        src = model.module if hasattr(model, "module") else model
        for s_param, param in zip(self.shadow.parameters(), src.parameters()):
            s_param.copy_(s_param * self.decay + param.data * (1.0 - self.decay))

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict):
        self.shadow.load_state_dict(state)


# ===========================================================================
# Training
# ===========================================================================

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def train(
    cfg_path: str,
    env_path: Optional[str] = None,
    resume: Optional[str] = None,
) -> None:
    # ── Config ──────────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, _load_env(env_path))
    if resume is not None:
        cfg["resume"] = resume

    data_root  = cfg["data"].get("data_root")
    if data_root is None:
        raise RuntimeError("data_root requis dans la config ou l'env.")

    output_dir  = Path(cfg["data"]["output_dir"])
    modality    = cfg["data"]["modality"]
    split       = cfg["data"].get("split", "retro_train")
    domains     = cfg["data"].get("domains", DOMAINS)
    p_lo        = cfg["data"].get("percentile_lower", 0.5)
    p_hi        = cfg["data"].get("percentile_upper", 99.5)
    max_per_dom = cfg["data"].get("max_volumes_per_domain", None)
    raw_vs      = cfg["data"].get("volume_size", None)
    volume_size = tuple(int(v) for v in raw_vs) if raw_vs else None
    raw_ts      = cfg["data"].get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    total_iters = cfg["train"]["total_iters"]
    batch_size  = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 4)
    lr          = cfg["train"]["lr"]
    sigma       = cfg["train"].get("sigma", 0.0)
    ot_method   = cfg["train"].get("ot_method", "exact")
    save_every  = cfg["train"].get("save_every", 5000)
    print_every = cfg["train"].get("print_every", 200)
    use_amp     = cfg["train"].get("use_amp", True)
    grad_clip   = cfg["train"].get("grad_clip", 1.0)
    ema_decay   = cfg["train"].get("ema_decay", 0.9999)

    # ── Distributed ─────────────────────────────────────────────────────────
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

    if is_main_process() and target_spacing:
        print(f"  Resampling → spacing cible : {target_spacing} mm")

    # ── VAE (figé) ───────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    latent_channels = cfg["model"]["latent_channels"]

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_ds = NIfTILatentDataset(
        data_root=Path(data_root),
        split=split,
        modality=modality,
        domains=domains,
        percentile_lower=p_lo,
        percentile_upper=p_hi,
        max_per_domain=max_per_dom,
        target_spacing=target_spacing,
        volume_size=volume_size,
    )

    # Pour chaque domaine, un loader infini séparé (comme en 2D)
    domain_loaders: Dict[str, any] = {}
    for d in domains:
        # Sous-ensemble par domaine
        domain_indices = [i for i, (_, di) in enumerate(train_ds.samples)
                         if di == DOMAIN_TO_IDX[d]]
        domain_subset = torch.utils.data.Subset(train_ds, domain_indices)
        if not domain_subset:
            continue
        sampler = DistributedSampler(domain_subset, shuffle=True) if is_distributed else None
        loader  = DataLoader(
            domain_subset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        domain_loaders[d] = _make_infinite(loader)

    available_domains = list(domain_loaders.keys())
    if len(available_domains) < 2:
        raise RuntimeError(f"Besoin d'au moins 2 domaines, {len(available_domains)} disponible(s).")

    # ── UNet 3D ──────────────────────────────────────────────────────────────
    unet = build_unet_3d(cfg, latent_channels).to(device)
    if is_distributed:
        unet = DDP(unet, device_ids=[local_rank])

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    if is_main_process():
        print(f"UNet 3D : {n_params / 1e6:.1f}M paramètres")

    ema = EMAModel(unet.module if is_distributed else unet, decay=ema_decay)

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(step: int) -> float:
        decay_start = total_iters // 2
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / (total_iters - decay_start))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # ── Flow Matcher ─────────────────────────────────────────────────────────
    try:
        if ot_method == "exact":
            FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        else:
            FM = ConditionalFlowMatcher(sigma=sigma)
        if is_main_process():
            print(f"Flow matcher: OT-CFM ({ot_method}), sigma={sigma}")
    except Exception:
        FM = ConditionalFlowMatcher(sigma=sigma)
        if is_main_process():
            print("Flow matcher: CFM (indépendant, fallback)")

    # ── Reprise ───────────────────────────────────────────────────────────────
    start_iter = 0
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        raw_unet = unet.module if is_distributed else unet
        raw_unet.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        start_iter = state.get("iter", 0) + 1
        if is_main_process():
            print(f"Reprise depuis iter {start_iter} : {resume_path}")

    weights_dir = output_dir / "weights"
    amp_dtype   = torch.float16 if use_amp else torch.float32

    if is_main_process():
        print(
            f"\nEntraînement OT-CFM 3D latent : {total_iters} iters"
            f" | batch={batch_size}"
            f" | lr={lr}"
            f" | AMP={'oui' if use_amp else 'non'}\n"
        )

    unet.train()
    t0 = time.time()
    last_log_t = t0
    ema_loss: Optional[float] = None
    recent_losses: deque = deque(maxlen=print_every)
    raw_unet = unet.module if is_distributed else unet

    for step in range(start_iter, total_iters):
        # ── Tirer deux domaines distincts ────────────────────────────────────
        src_domain, tgt_domain = random.sample(available_domains, 2)
        tgt_idx = DOMAIN_TO_IDX[tgt_domain]

        # ── Batches volumétriques ─────────────────────────────────────────────
        src_vol, _ = next(domain_loaders[src_domain])   # (B, 1, H, W, D) in [-1,1]
        tgt_vol, _ = next(domain_loaders[tgt_domain])
        src_vol = src_vol.to(device)
        tgt_vol = tgt_vol.to(device)

        # ── Encoder VAE (sans gradient, VAE est figé) ─────────────────────────
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            z_src, _ = vae.encode(src_vol)   # (B, C_lat, H', W', D')
            z_tgt, _ = vae.encode(tgt_vol)

        # ── OT-CFM matching ───────────────────────────────────────────────────
        t_batch, z_t, ut = FM.sample_location_and_conditional_flow(z_src, z_tgt)
        # t_batch : (B,)  z_t : (B, C_lat, H', W', D')  ut = z_tgt - z_src : cible

        # ── Forward UNet 3D ───────────────────────────────────────────────────
        z_in = torch.cat([z_t, z_src], dim=1)   # (B, 2*C_lat, H', W', D')
        t_vec = t_batch.to(device).float()
        y = torch.full((z_src.shape[0],), tgt_idx, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            vt = raw_unet(x=z_in, timesteps=t_vec, class_labels=y)  # (B, C_lat, ...)
            loss = F.mse_loss(vt, ut)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_unet.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        ema.update(unet)

        loss_val = float(loss.item())
        recent_losses.append(loss_val)
        ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)

        # ── Logging ───────────────────────────────────────────────────────────
        if is_main_process() and (step + 1) % print_every == 0:
            avg_recent = sum(recent_losses) / len(recent_losses)
            elapsed = time.time() - t0
            window_dt = time.time() - last_log_t
            iter_per_s = print_every / max(window_dt, 1e-9)
            eta_sec = (total_iters - step - 1) / max(iter_per_s, 1e-9)
            lr_cur = scheduler.get_last_lr()[0]
            mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f"[{step+1:6d}/{total_iters}]"
                f"  loss={avg_recent:.4f}"
                f"  ema={ema_loss:.4f}"
                f"  grad={float(grad_norm):.2f}"
                f"  lr={lr_cur:.2e}"
                f"  pair={src_domain}→{tgt_domain}"
                f"  speed={iter_per_s:.2f} it/s"
                f"  eta={eta_sec/3600:.2f}h"
                f"  t={elapsed/60:.1f}min"
                f"  mem={mem_gb:.1f}GB"
            )
            last_log_t = time.time()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if is_main_process() and (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step+1}.pth"
            torch.save({
                "iter": step,
                "model": raw_unet.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg_path": str(cfg_path),
            }, ckpt_path)
            print(f"  → Checkpoint : {ckpt_path}")

    if is_main_process():
        final_path = weights_dir / "model_final.pth"
        torch.save({
            "iter": total_iters - 1,
            "model": raw_unet.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg_path": str(cfg_path),
        }, final_path)
        print(f"\nEntraînement terminé. Modèle final : {final_path}")

    if is_distributed:
        dist.destroy_process_group()


# ===========================================================================
# Inference : T1W volume → T2W volume via ODE integration
# ===========================================================================

@torch.no_grad()
def _euler_integrate(
    unet: DiffusionModelUNet,
    z_src: torch.Tensor,
    tgt_idx: int,
    n_steps: int,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype = torch.float16,
) -> torch.Tensor:
    """Intégration Euler : z_src → z_tgt via le champ vectoriel appris."""
    dt = 1.0 / n_steps
    z = z_src.clone().to(device)
    y = torch.tensor([tgt_idx], dtype=torch.long, device=device)

    for step_i in range(n_steps):
        t_val = step_i * dt
        t_vec = torch.tensor([t_val], dtype=torch.float32, device=device)
        z_in = torch.cat([z, z_src], dim=1)   # (1, 2*C_lat, H', W', D')
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            vt = unet(x=z_in, timesteps=t_vec, class_labels=y)
        z = z + dt * vt

    return z


def infer(
    cfg_path: str,
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    source_domain: str,
    target_domain: str,
    env_path: Optional[str] = None,
    n_steps: Optional[int] = None,
    use_ema: bool = True,
) -> None:
    """Inférence complète : encode (T1W volumes) → intégration ODE → decode (T2W volumes)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, _load_env(env_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"].get("use_amp", False)
    amp_dtype = torch.float16 if use_amp else torch.float32
    latent_channels = cfg["model"]["latent_channels"]
    n_steps = n_steps or cfg["inference"].get("n_steps", 50)
    tgt_idx = DOMAIN_TO_IDX[target_domain]
    p_lo = cfg["data"].get("percentile_lower", 0.5)
    p_hi = cfg["data"].get("percentile_upper", 99.5)

    print(f"Inférence CFM 3D : {source_domain} → {target_domain} | {n_steps} steps")

    # ── Charger VAE ──────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)

    # ── Charger UNet ─────────────────────────────────────────────────────────
    unet = build_unet_3d(cfg, latent_channels).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    key = "ema" if (use_ema and "ema" in state) else "model"
    unet.load_state_dict(state[key])
    unet.eval()
    print(f"  UNet chargé depuis : {checkpoint} (clé: {key})")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(Path(input_dir).glob("*.nii.gz"))
    if not input_files:
        raise FileNotFoundError(f"Aucun fichier .nii.gz dans {input_dir}")
    print(f"  {len(input_files)} volumes à inférer → {out_dir}")

    for nii_path in input_files:
        t_start = time.time()
        img_nib = nib.load(str(nii_path))
        vol = img_nib.get_fdata(dtype=np.float32)

        # Normalisation per-volume
        lo = np.percentile(vol, p_lo)
        hi = np.percentile(vol, p_hi)
        vol_norm = np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 2.0 - 1.0
        vol_tensor = torch.from_numpy(vol_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)

        # Encode
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            z_src, _ = vae.encode(vol_tensor)   # (1, C_lat, H', W', D')

        # ODE integration
        z_pred = _euler_integrate(unet, z_src, tgt_idx, n_steps, device, use_amp, amp_dtype)

        # Decode
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            recon = vae.decode(z_pred)   # (1, 1, H, W, D)

        pred_vol = recon.squeeze().cpu().numpy()
        # Dénormalisation vers [0, 1] (espace de données d'origine)
        pred_vol = (np.clip(pred_vol, -1.0, 1.0) + 1.0) / 2.0

        # Sauvegarde avec le même espace/affine que l'entrée
        out_nii = nib.Nifti1Image(pred_vol, img_nib.affine, img_nib.header)
        out_path = out_dir / nii_path.name
        nib.save(out_nii, str(out_path))

        elapsed = time.time() - t_start
        print(f"  {nii_path.name} → {out_path.name}  ({elapsed:.1f}s)")

    print(f"\nInférence terminée. Prédictions dans : {out_dir}")


# ===========================================================================
# CLI
# ===========================================================================

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OT-CFM 3D — Traduction de contraste IRM 3D")
    p.add_argument("--mode",          default="train", choices=["train", "infer"])
    p.add_argument("--config",        required=True)
    p.add_argument("--env",           default=None)
    p.add_argument("--resume",        default=None)
    # Inférence
    p.add_argument("--checkpoint",    default=None)
    p.add_argument("--input_dir",     default=None)
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--source_domain", default=None)
    p.add_argument("--target_domain", default=None)
    p.add_argument("--n_steps",       type=int, default=None)
    p.add_argument("--no_ema",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.mode == "train":
        train(args.config, env_path=args.env, resume=args.resume)
    else:
        if not all([args.checkpoint, args.input_dir, args.output_dir,
                    args.source_domain, args.target_domain]):
            raise ValueError(
                "Mode infer : --checkpoint, --input_dir, --output_dir, "
                "--source_domain, --target_domain requis."
            )
        infer(
            cfg_path=args.config,
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            env_path=args.env,
            n_steps=args.n_steps,
            use_ema=not args.no_ema,
        )
