#!/usr/bin/env python3
"""Smoke test pour valider la Phase 1 (refactoring n_classes) du MMFM v2."""

import sys
import torch

sys.path.insert(0, "/home/rousseau/Exp/mrixfields_2026/src")

from cfm.mmfm_vectorized import VectorMMFM, LatentVectorizer


def test_vector_mmfm_n_classes():
    """Test que VectorMMFM accepte correctement num_classes=3 (v2) vs 15 (v1)."""
    latent_dim = 16384  #typical MedVAE flat dim for 128³
    n_fields = 5
    
    # Test v1 (15 classes = 3 modalités × 5 champs)
    n_classes_v1 = 3 * n_fields
    model_v1 = VectorMMFM(
        latent_dim=latent_dim,
        num_classes=n_classes_v1,
        hidden_dim=1024,
        depth=4,
        time_embed_dim=256,
        class_embed_dim=128,
    )
    
    # Test v2 (3 classes = modalités seules)
    n_classes_v2 = 3
    model_v2 = VectorMMFM(
        latent_dim=latent_dim,
        num_classes=n_classes_v2,
        hidden_dim=1024,
        depth=4,
        time_embed_dim=256,
        class_embed_dim=128,
    )
    
    # Vérifier la forme de l'embedding
    assert model_v1.class_embed.weight.shape[0] == n_classes_v1, \
        f"V1 embedding wrong size: {model_v1.class_embed.weight.shape[0]} vs {n_classes_v1}"
    assert model_v2.class_embed.weight.shape[0] == n_classes_v2, \
        f"V2 embedding wrong size: {model_v2.class_embed.weight.shape[0]} vs {n_classes_v2}"
    
    print(f"✅ VectorMMFM: n_classes={n_classes_v1} (v1) and {n_classes_v2} (v2) OK")
    
    # Test forward pass avec shapes correctes
    batch_size = 2
    z_t_vec = torch.randn(batch_size, latent_dim)
    z_src_vec = torch.randn(batch_size, latent_dim)
    timesteps = torch.rand(batch_size)
    
    # V1: labels dans [0, 14]
    y_tgt_v1 = torch.randint(0, n_classes_v1, (batch_size,))
    with torch.no_grad():
        out_v1 = model_v1(z_t_vec, z_src_vec, timesteps, y_tgt_v1)
    assert out_v1.shape == (batch_size, latent_dim), f"V1 output shape wrong: {out_v1.shape}"
    
    # V2: labels dans [0, 2]
    y_tgt_v2 = torch.randint(0, n_classes_v2, (batch_size,))
    with torch.no_grad():
        out_v2 = model_v2(z_t_vec, z_src_vec, timesteps, y_tgt_v2)
    assert out_v2.shape == (batch_size, latent_dim), f"V2 output shape wrong: {out_v2.shape}"
    
    print(f"✅ Forward pass: batch_size={batch_size}, latent_dim={latent_dim} OK")
    
    return True


def test_class_embedding_shapes():
    """Test que les embeddings sont correctement dimensionnés."""
    latent_dim = 16384
    
    for n_classes in [3, 15]:
        model = VectorMMFM(
            latent_dim=latent_dim,
            num_classes=n_classes,
            hidden_dim=1024,
        )
        
        # Vérifier l'embedding
        assert model.class_embed.num_embeddings == n_classes, \
            f"num_embeddings={model.class_embed.num_embeddings} != {n_classes}"
        
        # Test avec un batch d'index
        batch_size = 4
        class_labels = torch.arange(n_classes).repeat(batch_size // n_classes + 1)[:batch_size]
        class_feat = model.class_embed(class_labels)
        
        assert class_feat.shape == (batch_size, 128), \
            f"class_embed output shape wrong: {class_feat.shape}"
    
    print("✅ Class embedding shapes: n_classes ∈ {3, 15} OK")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("SMOTE TEST PHASE 1: MMFM v2 n_classes refactoring")
    print("=" * 60)
    
    try:
        print("\n[1/2] Testing VectorMMFM with different n_classes...")
        test_vector_mmfm_n_classes()
        
        print("\n[2/2] Testing class embedding shapes...")
        test_class_embedding_shapes()
        
        print("\n" + "=" * 60)
        print("✅ ALL SMOTE TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
