#!/usr/bin/env python3
"""Smoke tests for the vectorized MMFM v1 helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from cfm.mmfm_vectorized import LatentVectorizer, VectorMMFM


def test_latent_vectorizer_roundtrip():
    vectorizer = LatentVectorizer((1, 32, 32, 20))
    latent = torch.randn(2, 1, 32, 32, 20)
    latent_vec = vectorizer.flatten(latent)

    assert latent_vec.shape == (2, vectorizer.flat_dim)
    restored = vectorizer.unflatten(latent_vec)
    assert restored.shape == latent.shape
    assert torch.allclose(restored, latent)


def test_vector_mmfm_forward_shape():
    latent_dim = 1 * 32 * 32 * 20
    model = VectorMMFM(
        latent_dim=latent_dim,
        num_classes=15,
        hidden_dim=128,
        depth=2,
        time_embed_dim=32,
        class_embed_dim=16,
    )
    z_t = torch.randn(3, latent_dim)
    z_src = torch.randn(3, latent_dim)
    t = torch.rand(3)
    class_labels = torch.tensor([0, 7, 14], dtype=torch.long)

    out = model(z_t, z_src, t, class_labels)

    assert out.shape == z_t.shape


if __name__ == "__main__":
    test_latent_vectorizer_roundtrip()
    test_vector_mmfm_forward_shape()
    print("MMFM v1 smoke tests passed")
