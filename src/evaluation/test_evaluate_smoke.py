#!/usr/bin/env python3
"""
Smoke test for src/evaluation/evaluate.py
"""

import sys

sys.path.insert(0, "/home/rousseau/Exp/mrixfields_2026/src")

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.evaluate import (
    compute_dice,
    compute_lpips,
    compute_nrmse,
    compute_ssim,
    compute_volume_consistency,
    extract_subject_id,
    get_voxel_size,
    load_nifti,
)


def test_nrmse():
    """Test NRMSE computation."""
    pred = np.random.rand(64, 64, 32)
    target = pred + 0.01 * np.random.rand(64, 64, 32)

    nrmse = compute_nrmse(pred, target)
    assert 0 < nrmse < 0.1, f"NRMSE should be small, got {nrmse}"
    print(f"✓ NRMSE: {nrmse:.6f}")


def test_ssim():
    """Test SSIM computation."""
    pred = np.random.rand(64, 64, 32)
    target = pred + 0.01 * np.random.rand(64, 64, 32)

    ssim = compute_ssim(pred, target)
    assert 0 < ssim <= 1, f"SSIM should be in (0, 1], got {ssim}"
    print(f"✓ SSIM: {ssim:.6f}")


def test_dice():
    """Test Dice score computation."""
    # Create synthetic segmentations
    seg_pred = np.zeros((32, 32, 16), dtype=np.int32)
    seg_target = np.zeros((32, 32, 16), dtype=np.int32)

    # Overlapping region
    seg_pred[8:24, 8:24, 4:12] = 10  # Thalamus
    seg_target[10:26, 10:26, 6:14] = 10

    scores = compute_dice(seg_pred, seg_target)
    dice = scores["L_Thalamus"]
    assert 0 < dice <= 1, f"Dice should be in (0, 1], got {dice}"
    print(f"✓ Dice: {dice:.4f}")


def test_volume_consistency():
    """Test volume consistency computation."""
    seg_pred = np.zeros((32, 32, 16), dtype=np.int32)
    seg_target = np.zeros((32, 32, 16), dtype=np.int32)

    # Identical region
    seg_pred[8:24, 8:24, 4:12] = 10
    seg_target[8:24, 8:24, 4:12] = 10

    results = compute_volume_consistency(seg_pred, seg_target)
    vol = results["L_Thalamus"]
    assert 0.99 < vol <= 1.0, f"Volume consistency should be near 1.0, got {vol}"
    print(f"✓ Volume consistency: {vol:.4f}")


def test_subject_id_extraction():
    """Test subject ID extraction from filename."""
    filenames = [
        "P_T1W_0.1T_0006.nii.gz",
        "R_T2W_7T_0123.nii.gz",
        "P_T2FLAIR_3T_0042.nii.gz",
    ]
    expected = ["0006", "0123", "0042"]

    for fname, exp_sid in zip(filenames, expected):
        sid = extract_subject_id(fname)
        assert sid == exp_sid, f"Expected {exp_sid}, got {sid}"
        print(f"✓ {fname} -> {sid}")


def test_nifti_io():
    """Test NIfTI I/O with a dummy volume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy NIfTI
        data = np.random.rand(32, 32, 16).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmpdir / "test.nii.gz"
        nib.save(img, str(path))

        # Load and verify
        loaded, affine = load_nifti(path)
        assert loaded.shape == (32, 32, 16)
        assert affine.shape == (4, 4)

        # Get voxel size
        voxel_size = get_voxel_size(path)
        assert len(voxel_size) == 3
        assert all(v > 0 for v in voxel_size)

        print(f"✓ NIfTI I/O: shape={loaded.shape}, voxel_size={voxel_size}")


def main():
    print("=" * 60)
    print("Smoke tests for src/evaluation/evaluate.py")
    print("=" * 60)

    test_nrmse()
    test_ssim()
    test_dice()
    test_volume_consistency()
    test_subject_id_extraction()
    test_nifti_io()

    print()
    print("=" * 60)
    print("✅ All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
