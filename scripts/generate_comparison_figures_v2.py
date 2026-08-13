#!/usr/bin/env python3
"""Figures comparatives qualitatives — FOV complet + zoom patch.

Pour chaque paire clé et chaque sujet prospectif :
    Rang 1: FOV complet [ Source | GT | <méthode 1> | <méthode 2> | ... ]
    Rang 2: Zoom patch  [ Source | GT | <méthode 1> | <méthode 2> | ... ]
    + Rectangle rouge sur Rang 1 indiquant le boundary de la prédiction.

Les méthodes comparées ne sont plus figées dans le code : `--pred-dir
name=path` peut être répété pour ajouter/remplacer une entrée (ex: la
méthode active `mmfm_v2`) sans éditer ce fichier.

Usage :
    python scripts/generate_comparison_figures_v2.py
    python scripts/generate_comparison_figures_v2.py \\
        --pred-dir mmfm_v2=outputs/predictions/mmfm_v2/task3/T1W \\
        --pairs 0.1T:7T 1.5T:7T \\
        --subjects 0006 0007
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
import numpy as np

DATA_ROOT = Path("/home/rousseau/Data/MRIxFields_20260414/Training_prospective")

DEFAULT_PRED_DIRS = {
    "mlp":     Path("results/mmfm/visuals/mmfm_mlp_all_tasks"),
    "unet_v1": Path("results/mmfm/visuals/mmfm_unet_all_tasks"),
    "unet_v2": Path("results/mmfm/visuals/mmfm_unet_v2_all_tasks"),
}
DEFAULT_OUT_DIR = Path("results/mmfm/visuals/comparison_figures")

DEFAULT_PAIRS = [
    ("0.1T", "7T"),
    ("1.5T", "7T"),
    ("0.1T", "1.5T"),
    ("7T", "0.1T"),
]

DEFAULT_SUBJECTS = ["0006", "0007", "0009"]
DEFAULT_MODALITIES = ["T1W", "T2W", "T2FLAIR"]
AXES = [2, 1, 0]  # axial, coronal, sagittal
AXIS_NAMES = ["Axial", "Coronal", "Sagittal"]


def _load(path: Path, normalize=True, ref_nii=None):
    """Load volume. If ref_nii is provided, resample into ref space."""
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    if ref_nii is not None:
        if data.shape != ref_nii.shape or not np.allclose(nii.affine, ref_nii.affine, atol=1e-3):
            img = nib.Nifti1Image(data, nii.affine)
            resampled = nibabel.processing.resample_from_to(
                img, ref_nii, order=3, mode="constant", cval=0.0
            )
            data = resampled.get_fdata(dtype=np.float32)
    if normalize:
        lo, hi = np.percentile(data, 1), np.percentile(data, 99)
        if hi - lo < 1e-6:
            return np.zeros_like(data)
        data = np.clip((data - lo) / (hi - lo), 0, 1)
    return data


def _mid_slice(vol: np.ndarray, axis: int):
    return np.take(vol, vol.shape[axis] // 2, axis=axis)


def find_pred_file(directory: Path, modality: str, src: str, subject: str, tgt: str):
    """Find a prediction file, trying the official naming convention first
    (P_{MOD}_{TGT}_{ID}.nii.gz) then legacy source-tagged conventions.

    Also tries a `{src}_to_{tgt}/` subdirectory — the layout produced by
    infer_mmfm_unified.py's batch mode (`process_volume_unified`), the
    current convention for evaluation predictions (as opposed to the older
    flat layout from one-off generator scripts like
    generate_all_predictions_mmfm_mlp.sh)."""
    pair_dir = directory / f"{src}_to_{tgt}"
    candidates = [
        directory / f"P_{modality}_{tgt}_{subject}.nii.gz",
        pair_dir / f"P_{modality}_{tgt}_{subject}.nii.gz",
        directory / f"P_{modality}_{src}_{subject}_{modality}_{tgt}_mmfm_unet.nii.gz",
        directory / f"P_{modality}_{src}_{subject}_{modality}_{tgt}_mmfm.nii.gz",
        directory / f"P_{modality}_{src}_{subject}_{modality}_{tgt}_mmfm_unet_v2.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    pattern = f"P_{modality}_{src}_{subject}_{modality}_{tgt}_mmfm*.nii.gz"
    matches = list(directory.glob(pattern))
    return matches[0] if matches else None


def _bbox_nonzero_2d(arr2d, margin=5):
    """Return (y0, y1, x0, x1) around non-zero region with margin."""
    mask = arr2d > arr2d.max() * 0.02
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0, arr2d.shape[0], 0, arr2d.shape[1])
    y0 = max(0, ys.min() - margin)
    y1 = min(arr2d.shape[0], ys.max() + 1 + margin)
    x0 = max(0, xs.min() - margin)
    x1 = min(arr2d.shape[1], xs.max() + 1 + margin)
    return (y0, y1, x0, x1)


def generate_figure(
    subject: str,
    modality: str,
    src: str,
    tgt: str,
    pred_dirs: dict,
    out_dir: Path,
):
    method_names = list(pred_dirs.keys())

    src_path = DATA_ROOT / modality / src / f"P_{modality}_{src}_{subject}.nii.gz"
    tgt_path = DATA_ROOT / modality / tgt / f"P_{modality}_{tgt}_{subject}.nii.gz"

    pred_files = {
        name: find_pred_file(d, modality, src, subject, tgt)
        for name, d in pred_dirs.items()
    }

    for p in [src_path, tgt_path]:
        if not p.exists():
            print(f"Missing {p}, skip {modality} {subject} {src}->{tgt}")
            return

    missing_methods = [name for name, p in pred_files.items() if p is None]
    if missing_methods:
        print(f"Missing predictions for {missing_methods} on {modality} {subject} {src}->{tgt}, skip")
        return

    # Load source as reference space
    src_nii = nib.load(str(src_path))
    src_vol = _load(src_path)
    tgt_vol = _load(tgt_path, ref_nii=src_nii)
    pred_vols = {name: _load(p, ref_nii=src_nii) for name, p in pred_files.items()}

    rows_axes = len(AXES)
    cols = 2 + len(method_names)
    # Double rows per axis: FOV + zoom
    fig, axes = plt.subplots(rows_axes * 2, cols, figsize=(cols * 3, rows_axes * 6))
    if rows_axes == 1:
        axes = axes.reshape(2, cols)

    col_labels = ["Source", "GT"] + method_names
    for c, lab in enumerate(col_labels):
        axes[0, c].set_title(lab, fontsize=13, fontweight="bold")

    for r_idx, axis in enumerate(AXES):
        r_fov = r_idx * 2
        r_zoom = r_idx * 2 + 1

        axes[r_fov, 0].set_ylabel(f"{AXIS_NAMES[r_idx]} FOV", fontsize=11, labelpad=10)
        axes[r_zoom, 0].set_ylabel(f"{AXIS_NAMES[r_idx]} zoom", fontsize=11, labelpad=10)

        ims = [
            _mid_slice(src_vol, axis),
            _mid_slice(tgt_vol, axis),
        ] + [_mid_slice(pred_vols[name], axis) for name in method_names]

        # Compute bbox from GT slice for zoom
        bbox = _bbox_nonzero_2d(ims[1], margin=10)

        for c, im in enumerate(ims):
            # FOV row
            ax_fov = axes[r_fov, c]
            ax_fov.imshow(im.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            # Red rectangle around bbox
            rect = patches.Rectangle(
                (bbox[2], bbox[0]), bbox[3] - bbox[2], bbox[1] - bbox[0],
                linewidth=1.5, edgecolor="red", facecolor="none"
            )
            ax_fov.add_patch(rect)
            ax_fov.axis("off")

            # Zoom row
            ax_zoom = axes[r_zoom, c]
            im_zoom = im[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            ax_zoom.imshow(im_zoom.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax_zoom.axis("off")

    plt.suptitle(f"{modality} | {subject} | {src} → {tgt}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{modality}_{subject}_{src}_to_{tgt}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _parse_pred_dir(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--pred-dir attend 'name=path', reçu: {spec!r}"
        )
    name, path = spec.split("=", 1)
    return name, Path(path)


def _parse_pair(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--pairs attend 'SRC:TGT', reçu: {spec!r}"
        )
    src, tgt = spec.split(":", 1)
    return src, tgt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figures comparatives qualitatives multi-méthodes")
    p.add_argument(
        "--pred-dir", action="append", default=[], metavar="NAME=PATH",
        help="Ajoute/remplace une méthode comparée (répétable), ex: "
             "mmfm_v2=outputs/predictions/mmfm_v2/task3/T1W",
    )
    p.add_argument(
        "--no-default-methods", action="store_true",
        help="N'utilise que les méthodes passées via --pred-dir (ignore les 3 défauts)",
    )
    p.add_argument(
        "--pairs", nargs="+", default=None, metavar="SRC:TGT",
        help="Paires source:cible (défaut: 0.1T:7T 1.5T:7T 0.1T:1.5T 7T:0.1T)",
    )
    p.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES,
                    choices=["T1W", "T2W", "T2FLAIR"])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pred_dirs = {} if args.no_default_methods else dict(DEFAULT_PRED_DIRS)
    for spec in args.pred_dir:
        name, path = _parse_pred_dir(spec)
        pred_dirs[name] = path
    if not pred_dirs:
        raise SystemExit("Aucune méthode à comparer : fournir --pred-dir NAME=PATH")

    pairs = [_parse_pair(s) for s in args.pairs] if args.pairs else DEFAULT_PAIRS

    for modality in args.modalities:
        for subject in args.subjects:
            for src, tgt in pairs:
                print(f"Generating {modality} {subject} {src}->{tgt}...")
                generate_figure(subject, modality, src, tgt, pred_dirs, args.out_dir)
    print(f"\nAll comparative figures saved in {args.out_dir}")


if __name__ == "__main__":
    main()
