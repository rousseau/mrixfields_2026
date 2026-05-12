#!/usr/bin/env python3
"""QC visuel du VAE 3D pour MRIxFields 2026.

Le script charge un checkpoint VAE 3D entraîné, évalue quelques volumes
sélectionnés par domaine, puis sauvegarde une figure de comparaison:
entrée, reconstruction, erreur absolue, sur une ou plusieurs vues.

Objectif: vérifier rapidement la qualité du VAE sans reproduire les gros
volumes de la boucle CFM qui ont déclenché des OOM.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml

from train_vae_3d import build_vae


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path("/home/rousseau/Data/MRIxFields_20260414")
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_VAE_CONFIG = PROJECT_DIR / "configs" / "vae3d_T1W.yaml"
DEFAULT_CHECKPOINT = PROJECT_DIR / "outputs" / "vae3d" / "runs" / "vae3d_T1W" / "weights" / "model_best.pth"

DOMAINS = ["0.1T", "1.5T", "3T", "5T", "7T"]
SPLIT_MAP = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}
VIEW_TO_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def _load_env(env_arg: Optional[str]) -> Optional[dict]:
    if env_arg is None:
        return None
    env_path = env_arg if env_arg.endswith(".yaml") else f"configs/env/{env_arg}.yaml"
    if not os.path.isabs(env_path):
        candidate = PROJECT_DIR / env_path
        env_path = str(candidate) if candidate.exists() else env_path
    with open(env_path) as f:
        raw = yaml.safe_load(f)
    return {k: os.path.expandvars(str(v)) for k, v in raw.items()}


def _resolve_paths(cfg: dict, env: Optional[dict]) -> dict:
    if env is None:
        return cfg
    data = cfg.setdefault("data", {})
    if "output_subdir" in data:
        data["output_dir"] = str(Path(env["output_root"]) / data["output_subdir"])
    if "data_root" in env:
        data.setdefault("data_root", env["data_root"])
    return cfg


def _resample_volume(vol: np.ndarray, original_spacing, target_spacing: Sequence[float]) -> np.ndarray:
    from scipy.ndimage import zoom as scipy_zoom

    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def _normalize(vol: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def _center_crop_or_pad(vol: np.ndarray, size: Optional[Sequence[int]]) -> np.ndarray:
    if size is None:
        return vol
    th, tw, td = (int(v) for v in size)
    h, w, d = vol.shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    pad_d = max(0, td - d)
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        vol = np.pad(
            vol,
            [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (pad_d // 2, pad_d - pad_d // 2)],
            mode="reflect",
        )
        h, w, d = vol.shape
    start_h = max((h - th) // 2, 0)
    start_w = max((w - tw) // 2, 0)
    start_d = max((d - td) // 2, 0)
    return vol[start_h : start_h + th, start_w : start_w + tw, start_d : start_d + td]


def _extract_subject_id(path: Path) -> str:
    stem = path.name.replace(".nii.gz", "")
    parts = stem.split("_")
    return parts[-1] if parts else stem


def _load_volume(path: Path, target_spacing: Optional[Sequence[float]], lo_pct: float, hi_pct: float, patch_size: Optional[Sequence[int]]) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    if target_spacing is not None:
        spacing = np.abs(np.diag(img.affine)[:3])
        vol = _resample_volume(vol, spacing, target_spacing)
    vol = _normalize(vol, lo_pct, hi_pct)
    vol = _center_crop_or_pad(vol, patch_size)
    return vol


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    if not isinstance(state, dict):
        model.load_state_dict(state, strict=True)
        return

    # Compat DDP: certains checkpoints sauvegardent les poids avec le préfixe "module.".
    if state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}

    model_keys = set(model.state_dict().keys())
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key in model_keys:
            remapped[key] = value
            continue

        alt_a = key.replace(".conv.conv.", ".postconv.conv.")
        alt_b = key.replace(".postconv.conv.", ".conv.conv.")

        if alt_a in model_keys and alt_a not in remapped:
            remapped[alt_a] = value
        elif alt_b in model_keys and alt_b not in remapped:
            remapped[alt_b] = value
        else:
            remapped[key] = value

    model.load_state_dict(remapped, strict=True)


def _select_samples(data_root: Path, split: str, modality: str, domains: Sequence[str], n_per_domain: int, subjects: Optional[Sequence[str]]) -> list[tuple[str, Path]]:
    split_dir = SPLIT_MAP.get(split, split)
    selected: list[tuple[str, Path]] = []
    wanted = set(subjects) if subjects else None
    for domain in domains:
        domain_dir = data_root / split_dir / modality / domain
        files = sorted(domain_dir.glob("*.nii.gz"))
        if wanted is not None:
            files = [p for p in files if _extract_subject_id(p) in wanted]
        else:
            files = files[:n_per_domain]
        for path in files:
            selected.append((domain, path))
    return selected


def _slice_by_view(vol: np.ndarray, view: str) -> np.ndarray:
    axis = VIEW_TO_AXIS[view]
    return np.take(vol, vol.shape[axis] // 2, axis=axis)


def _render_grid(rows: list[dict], views: Sequence[str], out_path: Path) -> None:
    n_cols = len(views) * 3
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.5 * n_rows), squeeze=False)

    col_titles: list[str] = []
    for view in views:
        col_titles.extend([f"{view}\ninput", f"{view}\nrecon", f"{view}\n|err|"])

    for row_idx, row in enumerate(rows):
        vol_in = row["input"]
        vol_rec = row["recon"]
        vol_err = np.abs(vol_in - vol_rec)
        for view_idx, view in enumerate(views):
            base = view_idx * 3
            slices = (
                _slice_by_view(vol_in, view),
                _slice_by_view(vol_rec, view),
                _slice_by_view(vol_err, view),
            )
            for offset, sl in enumerate(slices):
                ax = axes[row_idx][base + offset]
                cmap = "magma" if offset == 2 else "gray"
                ax.imshow(sl.T, cmap=cmap, origin="lower")
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(col_titles[base + offset], fontsize=8)
        axes[row_idx][0].set_ylabel(f"{row['domain']}\n{row['subject']}", fontsize=8)

    fig.suptitle("QC VAE 3D — entrée / reconstruction / erreur absolue", fontsize=10, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="QC visuel du VAE 3D")
    parser.add_argument("--env", default=None, help="Environnement YAML (local, jeanzay, ...) pour résoudre les chemins")
    parser.add_argument("--config", default=str(DEFAULT_VAE_CONFIG), help="Config VAE 3D utilisée pour construire le modèle")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint VAE à charger")
    parser.add_argument("--data-root", default=None, help="Override du data_root")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Répertoire de sortie des figures")
    parser.add_argument("--split", default="retro_train", choices=["retro_train", "pro_train", "pro_val", "pro_test"], help="Split à inspecter")
    parser.add_argument("--modality", default="T1W", choices=["T1W", "T2W", "T2FLAIR"], help="Modalité à inspecter")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domaines/champs à comparer")
    parser.add_argument("--n-per-domain", type=int, default=2, help="Nombre de volumes à afficher par domaine")
    parser.add_argument("--subjects", nargs="+", default=None, help="IDs de sujets explicites à utiliser")
    parser.add_argument("--views", nargs="+", default=["axial"], choices=list(VIEW_TO_AXIS.keys()), help="Vues à afficher")
    parser.add_argument("--patch-size", nargs=3, type=int, default=None, metavar=("H", "W", "D"), help="Taille de crop/pad avant reconstruction")
    parser.add_argument("--target-spacing", nargs=3, type=float, default=None, metavar=("SX", "SY", "SZ"), help="Spacing cible pour le rééchantillonnage")
    parser.add_argument("--percentile-lower", type=float, default=0.5, help="Percentile bas pour la normalisation")
    parser.add_argument("--percentile-upper", type=float, default=99.5, help="Percentile haut pour la normalisation")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--no-amp", action="store_true", help="Désactive autocast AMP même sur GPU")
    parser.add_argument("--out", default=None, help="Chemin de sortie de la figure")
    args = parser.parse_args()

    env = _load_env(args.env)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, env)

    data_root = Path(args.data_root or cfg["data"].get("data_root") or DEFAULT_DATA_DIR)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (PROJECT_DIR / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")

    target_spacing = tuple(float(v) for v in args.target_spacing) if args.target_spacing else tuple(cfg["data"].get("target_spacing", []) or [])
    if not target_spacing:
        target_spacing = None

    patch_size = tuple(int(v) for v in args.patch_size) if args.patch_size else tuple(cfg["data"].get("patch_size", []) or [])
    if not patch_size:
        patch_size = None

    samples = _select_samples(
        data_root=data_root,
        split=args.split,
        modality=args.modality,
        domains=args.domains,
        n_per_domain=args.n_per_domain,
        subjects=args.subjects,
    )
    if not samples:
        raise RuntimeError("Aucun volume sélectionné pour le QC VAE.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (not args.no_amp) and device.type == "cuda"

    vae = build_vae(cfg)
    _load_checkpoint(vae, checkpoint, device)
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    rows: list[dict] = []
    with torch.inference_mode():
        for domain, path in samples:
            vol = _load_volume(path, target_spacing, args.percentile_lower, args.percentile_upper, patch_size)
            inp = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = vae(inp)
            recon = out[0] if isinstance(out, (tuple, list)) else out
            recon = recon.clamp(-1.0, 1.0).float().cpu().numpy()[0, 0]
            inp_np = inp.float().cpu().numpy()[0, 0]
            mae = float(np.mean(np.abs(inp_np - recon)))
            mse = float(np.mean((inp_np - recon) ** 2))
            subject_id = _extract_subject_id(path)
            rows.append({"domain": domain, "subject": subject_id, "input": inp_np, "recon": recon})
            print(f"[{domain}] {subject_id}  MAE={mae:.5f}  MSE={mse:.5f}")

    out_path = Path(args.out) if args.out else (Path(args.results_dir) / f"qc_vae3d_{args.modality}_{args.split}.png")
    _render_grid(rows, args.views, out_path)
    print(f"Figure sauvegardée : {out_path}")
    print(f"Volumes traités : {len(rows)} | vues : {', '.join(args.views)} | device : {device}")


if __name__ == "__main__":
    main()