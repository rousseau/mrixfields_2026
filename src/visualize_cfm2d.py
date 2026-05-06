#!/usr/bin/env python3
"""
Visualisation des résultats OT-CFM 2D — MRIxFields 2026

Génère une figure de comparaison multi-sujet / multi-champ pour une modalité :
  - Lignes : sujets (Training_prospective)
  - Colonnes : 0.1T input | 1.5T pred | 1.5T GT | 3T pred | 3T GT | 5T pred | 5T GT | 7T pred | 7T GT

Utilisation :
    python src/visualize_cfm2d.py --checkpoint outputs/cfm2d/runs/cfm2d_T1W/weights/checkpoint_5000.pth \\
                                   --config configs/cfm_T1W.yaml

Prérequis :
    - Si --checkpoint fourni : l'inférence est lancée à la volée sur les sujets prospectifs
    - Sinon : les prédictions doivent être dans $EXP_DIR/cfm2d/predictions/<modality>/0.1T_to_<field>/
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Chemins — valeurs par défaut (machine locale), surchargées via --env
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/rousseau/Data/MRIxFields_20260414")
EXP_DIR = Path("/home/rousseau/Exp/mrixfields_2026/outputs/cfm2d")
RESULTS_DIR = Path("/home/rousseau/Exp/mrixfields_2026/results")

TARGET_FIELDS = ["1.5T", "3T", "5T", "7T"]
SOURCE_FIELD = "0.1T"

PYTHON = "/home/rousseau/miniforge3/bin/python"
TRAIN_SCRIPT = Path("/home/rousseau/Exp/mrixfields_2026/src/train_cfm2d.py")

# Symboles importés depuis train_cfm2d (chargement différé dans infer_middle_slices)
_cfm_imports: dict = {}


def _setup_paths(env_arg: str | None) -> None:
    """Charge l'env YAML et met à jour les chemins globaux."""
    global DATA_DIR, EXP_DIR, RESULTS_DIR, PYTHON, TRAIN_SCRIPT
    if env_arg is None:
        return
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
    env = {k: Path(os.path.expandvars(str(v))) for k, v in raw.items()}
    output_root = env["output_root"]
    project_root = env["project_root"]
    DATA_DIR = env["data_root"]
    EXP_DIR = output_root / "cfm2d"
    RESULTS_DIR = project_root / "results"
    PYTHON = str(env.get("python", Path("python3")))
    TRAIN_SCRIPT = project_root / "src" / "train_cfm2d.py"
    _cfm_imports.clear()  # force re-import si l'env change


# ---------------------------------------------------------------------------
# Inférence à la volée — coupe centrale uniquement (in-process)
# ---------------------------------------------------------------------------

def _load_cfm_symbols() -> dict:
    """Importe les symboles nécessaires depuis train_cfm2d (chargement différé)."""
    if _cfm_imports:
        return _cfm_imports
    src_dir = str(TRAIN_SCRIPT.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import train_cfm2d as _cfm  # noqa: F401
    _cfm_imports.update({
        "build_unet":       _cfm.build_unet,
        "_euler_integrate": _cfm._euler_integrate,
        "DOMAIN_TO_IDX":    _cfm.DOMAIN_TO_IDX,
        "_pad_slice":       _cfm._pad_slice,
        "_unpad_slice":     _cfm._unpad_slice,
        "_resolve_paths":   _cfm._resolve_paths,
        "_load_env":        _cfm._load_env,
    })
    return _cfm_imports


def infer_middle_slices(
    checkpoint: Path,
    config: Path,
    modality: str,
    subjects: list,
    axis: int = 2,
    env_arg: str | None = None,
) -> dict:
    """Charge le modèle une fois et infère la coupe centrale de chaque sujet × champ.

    Retourne {subject_id: {target_field: np.ndarray}} — tableaux normalisés [0, 1].
    """
    import torch
    import yaml
    import nibabel as nib

    sym = _load_cfm_symbols()

    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg = sym["_resolve_paths"](cfg, sym["_load_env"](env_arg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sym["build_unet"](cfg, use_checkpoint=False).to(device)
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    model.eval()
    print(f"Modèle chargé : {checkpoint.name}  ({device})")

    img_size = cfg["model"]["img_size"]
    n_steps  = cfg.get("inference", {}).get("n_steps", 50)
    input_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD

    result: dict = {}
    total = len(subjects) * len(TARGET_FIELDS)
    done  = 0
    for sid in subjects:
        src_file = _find_subject_file(input_dir, sid)
        if src_file is None:
            print(f"  [ATTENTION] Fichier source introuvable : {sid}")
            result[sid] = {sid: {} for tgt in TARGET_FIELDS}
            continue

        nii_img = nib.load(str(src_file))
        vol     = nii_img.get_fdata(dtype=np.float32)
        sl_idx  = vol.shape[axis] // 2
        sl_src  = np.clip(np.take(vol, sl_idx, axis=axis), 0.0, 1.0)

        result[sid] = {}
        sl_padded, pad_params = sym["_pad_slice"](sl_src, img_size, img_size)
        orig_h, orig_w, ph1, pw1 = pad_params
        x_src = torch.from_numpy(sl_padded * 2.0 - 1.0).float()
        x_src = x_src.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        for tgt in TARGET_FIELDS:
            done += 1
            print(f"  [{done}/{total}] {SOURCE_FIELD}→{tgt}  sujet {sid} …", flush=True)
            tgt_idx = sym["DOMAIN_TO_IDX"][tgt]
            x_out   = sym["_euler_integrate"](model, x_src, tgt_idx, device, n_steps)
            sl_out  = x_out.squeeze().cpu().numpy()
            sl_out  = np.clip((sl_out + 1.0) / 2.0, 0.0, 1.0)
            sl_out  = sym["_unpad_slice"](sl_out, orig_h, orig_w, ph1, pw1)
            result[sid][tgt] = sl_out
            print(f"      OK")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mid_slice(nifti_path: Path, axis: int = 2) -> np.ndarray:
    """Charge un volume NIfTI et retourne la coupe centrale selon l'axe."""
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    idx = data.shape[axis] // 2
    return np.take(data, idx, axis=axis)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalisation robuste [0, 1] par percentile 1–99."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _find_subject_file(directory: Path, subject_id: str) -> Path | None:
    """Trouve le NIfTI d'un sujet dans un répertoire."""
    if not directory.exists():
        return None
    pattern = re.compile(rf".*_{subject_id}\.nii\.gz$")
    for f in sorted(directory.glob("*.nii.gz")):
        if pattern.match(f.name):
            return f
    return None


def _list_subjects(modality: str) -> list:
    """Retourne les IDs de sujets disponibles dans Training_prospective."""
    src_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
    if not src_dir.exists():
        raise FileNotFoundError(f"Répertoire source introuvable : {src_dir}")
    subjects = []
    for f in sorted(src_dir.glob("*.nii.gz")):
        parts = f.stem.replace(".nii", "").split("_")
        if parts:
            subjects.append(parts[-1])
    return subjects


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    modality: str,
    subjects: list,
    axis: int,
    out_path: Path,
    pred_dir: Path | None = None,
    pred_slices: dict | None = None,
) -> None:
    """Génère la grille de comparaison.

    pred_slices : dict {subject_id: {target_field: np.ndarray}} pré-calculé.
    pred_dir    : répertoire contenant les sous-dossiers <src>_to_<tgt>/
                  (utilisé si pred_slices est None).
    Si les deux sont None, utilise EXP_DIR/predictions/<modality>/.
    """
    if pred_slices is None and pred_dir is None:
        pred_dir = EXP_DIR / "predictions" / modality

    col_labels = [f"{SOURCE_FIELD}\n(input)"]
    for tgt in TARGET_FIELDS:
        col_labels.append(f"{tgt}\n(pred)")
        col_labels.append(f"{tgt}\n(GT)")
    n_cols = len(col_labels)
    n_rows = len(subjects)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    for row_idx, subject_id in enumerate(subjects):
        # Source (0.1T input)
        src_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
        src_file = _find_subject_file(src_dir, subject_id)
        sl_src = _normalize(_load_mid_slice(src_file, axis)) if src_file else np.zeros((64, 64))
        axes[row_idx][0].imshow(sl_src.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row_idx][0].set_title(col_labels[0] if row_idx == 0 else "", fontsize=7)
        axes[row_idx][0].set_ylabel(f"Sujet {subject_id}", fontsize=7)
        axes[row_idx][0].axis("off")

        for tgt_idx, tgt_field in enumerate(TARGET_FIELDS):
            col_pred = 1 + 2 * tgt_idx
            col_gt = col_pred + 1

            # Prédiction — depuis dict pré-calculé ou depuis fichier
            if pred_slices is not None:
                arr = pred_slices.get(subject_id, {}).get(tgt_field)
                sl_pred = _normalize(arr) if arr is not None else np.zeros((64, 64))
                if arr is None:
                    print(f"  [ATTENTION] Prédiction manquante dans pred_slices : {subject_id}/{tgt_field}")
            else:
                tgt_pred_dir = pred_dir / f"{SOURCE_FIELD}_to_{tgt_field}"
                pred_file = _find_subject_file(tgt_pred_dir, subject_id)
                if pred_file is not None:
                    sl_pred = _normalize(_load_mid_slice(pred_file, axis))
                else:
                    sl_pred = np.zeros((64, 64))
                    print(f"  [ATTENTION] Prédiction manquante : {tgt_pred_dir} / {subject_id}")

            # Ground truth
            gt_dir = DATA_DIR / "Training_prospective" / modality / tgt_field
            gt_file = _find_subject_file(gt_dir, subject_id)
            sl_gt = _normalize(_load_mid_slice(gt_file, axis)) if gt_file else np.zeros((64, 64))

            for col_idx, sl in ((col_pred, sl_pred), (col_gt, sl_gt)):
                axes[row_idx][col_idx].imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=1)
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(col_labels[col_idx], fontsize=7)
                axes[row_idx][col_idx].axis("off")

    fig.suptitle(
        f"OT-CFM 2D — {modality} : 0.1T → champs cibles\n"
        f"(Training_prospective, coupe centrale axe {axis})",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisation OT-CFM 2D")
    parser.add_argument("--modality", default="T1W", choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument("--axis", type=int, default=2,
                        help="Axe de coupe (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="IDs de sujets (défaut : tous dans Training_prospective)")
    parser.add_argument("--out", default=None, help="Chemin de sauvegarde de la figure")
    parser.add_argument("--checkpoint", default=None,
                        help="Chemin vers un checkpoint (.pth) — lance l'inférence à la volée")
    parser.add_argument("--config", default=None,
                        help="Config YAML à utiliser avec --checkpoint")
    parser.add_argument(
        "--env", default=None,
        help="Env config : 'local', 'jeanzay', ou chemin vers configs/env/*.yaml",
    )
    args = parser.parse_args()

    # Résoudre les chemins selon l'environnement
    _setup_paths(args.env)

    if args.subjects is None:
        args.subjects = _list_subjects(args.modality)
    print(f"Sujets : {args.subjects}")

    pred_dir = None

    if args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"[ERREUR] Checkpoint introuvable : {ckpt}", file=sys.stderr)
            sys.exit(1)

        step_tag = ckpt.stem  # ex: "checkpoint_5000" ou "model_final"

        # Config : argument ou auto-détecté depuis project_root
        if args.config:
            config = Path(args.config)
        else:
            project_root = TRAIN_SCRIPT.parent.parent
            config = project_root / "configs" / f"cfm_{args.modality}.yaml"
        if not config.exists():
            print(f"[ERREUR] Config introuvable : {config}", file=sys.stderr)
            sys.exit(1)

        if args.out is None:
            args.out = str(RESULTS_DIR / f"cfm2d_{args.modality.lower()}_{step_tag}.png")

        pred_slices = infer_middle_slices(
            ckpt, config, args.modality, args.subjects,
            axis=args.axis, env_arg=args.env,
        )
    else:
        pred_slices = None

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"cfm2d_{args.modality.lower()}.png"
    )

    make_figure(
        modality=args.modality,
        subjects=args.subjects,
        axis=args.axis,
        out_path=out_path,
        pred_dir=pred_dir,
        pred_slices=pred_slices,
    )


if __name__ == "__main__":
    main()
