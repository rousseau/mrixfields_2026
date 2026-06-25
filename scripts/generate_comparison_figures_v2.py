#!/usr/bin/env python3
"""Figures comparatives qualitatives — FOV complet + zoom patch.

Pour chaque paire clé et chaque sujet prospectif :
    Rang 1: FOV complet [ Source | GT | MLP | UNet V1 | UNet V2 ]
    Rang 2: Zoom patch  [ Source | GT | MLP | UNet V1 | UNet V2 ]
    + Rectangle rouge sur Rang 1 indiquant le boundary de la prédiction.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel
import nibabel.processing
import numpy as np

DATA_ROOT = Path("/home/rousseau/Data/MRIxFields_20260414/Training_prospective")
PRED_DIRS = {
    "mlp":     Path("results/mmfm/visuals/mmfm_mlp_all_tasks"),
    "unet_v1": Path("results/mmfm/visuals/mmfm_unet_all_tasks"),
    "unet_v2": Path("results/mmfm/visuals/mmfm_unet_v2_all_tasks"),
}
OUT_DIR = Path("results/mmfm/visuals/comparison_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("0.1T", "7T"),
    ("1.5T", "7T"),
    ("0.1T", "1.5T"),
    ("7T", "0.1T"),
]

SUBJECTS = ["0006", "0007", "0009"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
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
    """Find a prediction file using multiple possible suffixes."""
    candidates = [
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


def generate_figure(subject: str, modality: str, src: str, tgt: str):
    src_path = DATA_ROOT / modality / src / f"P_{modality}_{src}_{subject}.nii.gz"
    tgt_path = DATA_ROOT / modality / tgt / f"P_{modality}_{tgt}_{subject}.nii.gz"

    pred_files = {
        name: find_pred_file(d, modality, src, subject, tgt)
        for name, d in PRED_DIRS.items()
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
    cols = 5
    # Double rows per axis: FOV + zoom
    fig, axes = plt.subplots(rows_axes * 2, cols, figsize=(cols * 3, rows_axes * 6))
    if rows_axes == 1:
        axes = axes.reshape(2, cols)

    col_labels = ["Source", "GT", "MLP", "UNet V1", "UNet V2"]
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
            _mid_slice(pred_vols["mlp"], axis),
            _mid_slice(pred_vols["unet_v1"], axis),
            _mid_slice(pred_vols["unet_v2"], axis),
        ]

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

    out_path = OUT_DIR / f"compare_{modality}_{subject}_{src}_to_{tgt}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    for modality in MODALITIES:
        for subject in SUBJECTS:
            for src, tgt in PAIRS:
                print(f"Generating {modality} {subject} {src}->{tgt}...")
                generate_figure(subject, modality, src, tgt)
    print(f"\nAll comparative figures saved in {OUT_DIR}")
