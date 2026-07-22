import torch
import numpy as np
from pathlib import Path
from common.io import DOMAINS, MODALITIES
from cfm.mmfm_vectorized import VectorMMFM
from cfm.train_mmfm_3d import _field_to_time, _euler_integrate_vector

def test_field_mapping():
    print("Testing field mapping...")
    n_fields = len(DOMAINS) # 5
    expected = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
    for i in range(n_fields):
        val = _field_to_time(i, n_fields)
        assert abs(val - expected[i]) < 1e-6, f"Field {i} mapping failed: expected {expected[i]}, got {val}"
    print("✅ Field mapping OK.")

def test_model_shapes():
    print("\nTesting model shapes (v2)...")
    latent_dim = 1024
    num_classes = 3 # modalities
    model = VectorMMFM(
        latent_dim=latent_dim,
        num_classes=num_classes,
        hidden_dim=512,
        depth=2,
        time_embed_dim=64,
        class_embed_dim=32
    )
    
    z_t = torch.randn(1, latent_dim)
    z_src = torch.randn(1, latent_dim)
    t_val = torch.tensor([0.5])
    y_mod = torch.tensor([0])
    
    out = model(z_t, z_src, t_val, y_mod)
    assert out.shape == (1, latent_dim), f"Model output shape mismatch: {out.shape}"
    print("✅ Model forward shape OK.")

def test_euler_v2_logic():
    print("\nTesting Euler v2 integration limits...")
    latent_dim = 128
    num_classes = 3
    model = VectorMMFM(latent_dim, num_classes)
    model.eval()
    
    z_src_vec = torch.randn(1, latent_dim)
    tgt_class = 0
    n_steps = 10
    device = torch.device("cpu")
    
    # Case 1: identity transition (0.1T -> 0.1T)
    # We expect the integrate to run, but the actual "delta" is handled by the model.
    # In v2 mapping, t_start = t_end.
    z_out_id = _euler_integrate_vector(
        model, z_src_vec, tgt_class, n_steps, device,
        method="mmfm3d_vectorized_v2",
        source_field_idx=0,
        target_field_idx=0,
        n_fields=5
    )
    assert z_out_id.shape == (1, latent_dim)
    
    # Case 2: forward transition (0.1T -> 7T)
    z_out_fwd = _euler_integrate_vector(
        model, z_src_vec, tgt_class, n_steps, device,
        method="mmfm3d_vectorized_v2",
        source_field_idx=0,
        target_field_idx=4,
        n_fields=5
    )
    assert z_out_fwd.shape == (1, latent_dim)
    print("✅ Euler v2 integration flow OK.")

if __name__ == "__main__":
    try:
        test_field_mapping()
        test_model_shapes()
        test_euler_v2_logic()
        print("\nAll technical tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
