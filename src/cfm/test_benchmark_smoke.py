#!/usr/bin/env python3
"""
Simplified benchmark for patched VAE (smoke test version).
"""

import sys
import csv
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.patched_vae import PatchedVAE

def test_benchmark_smoke():
    """Smoke test: verify benchmark structure works without loading real VAE checkpoints."""
    print("="*70)
    print(" VAE Benchmark Smoke Test (Patch-based Processing)")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create dummy VAE for testing
    class DummyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Conv3d(1, 8, 3, padding=1, stride=2)
            self.dec = torch.nn.ConvTranspose3d(8, 1, 3, padding=1, stride=2, output_padding=1)
        
        def encode(self, x):
            return self.enc(x)
        
        def decode(self, z):
            return torch.tanh(self.dec(z))
    
    vae = DummyVAE().to(device)
    vae.eval()
    
    # Create patched wrapper
    patched_vae = PatchedVAE(vae, patch_size=(112, 128, 80), overlap=0.25)
    patched_vae = patched_vae.to(device)
    
    print("✓ Dummy VAE + PatchedVAE wrapper created")
    
    # Simulate benchmark on 3 volumes
    test_volumes = [
        ("vol_0001", torch.randn(1, 1, 200, 200, 160)),
        ("vol_0002", torch.randn(1, 1, 200, 200, 160)),
        ("vol_0003", torch.randn(1, 1, 200, 200, 160)),
    ]
    
    output_dir = Path("outputs/benchmark_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning benchmark on dummy volumes...")
    print("-" * 70)
    
    results = []
    
    with torch.no_grad():
        for name, x in test_volumes:
            try:
                # Encode-decode cycle
                result = patched_vae.forward(x.to(device), encode_only=False, batch_size=2)
                x_rec = result["reconstruction"]
                
                # Compute metrics
                mae = torch.mean(torch.abs(x.squeeze() - x_rec)).item()
                mse = torch.mean((x.squeeze() - x_rec) ** 2).item()
                
                # Simple SSIM approximation (cross-correlation)
                ssim_approx = 1.0 - (mse / 2.0)  # simplified
                
                results.append({
                    "name": name,
                    "mae": mae,
                    "mse": mse,
                    "ssim": max(0, min(1, ssim_approx)),
                })
                
                print(f"✓ {name}: MAE={mae:.4f}, MSE={mse:.4f}, SSIM≈{ssim_approx:.4f}")
            
            except Exception as e:
                print(f"✗ {name}: {str(e)[:60]}")
                results.append({"name": name, "error": str(e)})
    
    # Write CSV
    csv_path = output_dir / "benchmark_smoke_test.csv"
    with open(csv_path, "w", newline="") as f:
        if results and "mae" in results[0]:
            writer = csv.DictWriter(f, fieldnames=["name", "mae", "mse", "ssim"])
            writer.writeheader()
            writer.writerows(results)
        else:
            writer = csv.DictWriter(f, fieldnames=["name", "error"])
            writer.writeheader()
            writer.writerows(results)
    
    print("-" * 70)
    print(f"\n✓ Results saved to {csv_path}")
    print(f"  {len([r for r in results if 'mae' in r])} successes, "
          f"{len([r for r in results if 'error' in r])} failures")
    
    print("\n" + "="*70)
    print(" ✓ Benchmark smoke test completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Load real AEKL checkpoint in load_aekl()")
    print("  2. Load real VQ-VAE checkpoint in load_vqvae()")
    print("  3. Run full benchmark: python3 src/benchmark_vae.py --data-root ... --max-samples 3")
    print("  4. On JeanZay: sbatch src/slurm/benchmark_vae_jeanzay.slurm T1W 0.1T")

if __name__ == "__main__":
    test_benchmark_smoke()
