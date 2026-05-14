#!/usr/bin/env python3
"""
Minimal benchmark test for patched VAE wrapper.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.patched_vae import PatchedVAE, create_patched_vae

def test_patched_vae():
    """Test PatchedVAE with dummy model."""
    print("="*70)
    print(" Testing PatchedVAE wrapper with dummy VAE")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create a dummy VAE (simple autoencoder)
    class DummyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Sequential(
                torch.nn.Conv3d(1, 8, 3, padding=1, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv3d(8, 16, 3, padding=1, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv3d(16, 32, 3, padding=1, stride=2),
            )
            self.dec = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(32, 16, 3, padding=1, stride=2, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(16, 8, 3, padding=1, stride=2, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(8, 1, 3, padding=1, stride=2, output_padding=1),
            )
        
        def encode(self, x):
            return self.enc(x)
        
        def decode(self, z):
            return torch.tanh(self.dec(z))
    
    # Create model
    vae = DummyVAE().to(device)
    vae.eval()
    print("✓ Created dummy VAE")
    
    # Wrap in PatchedVAE
    patched = PatchedVAE(vae, patch_size=(64, 64, 64), overlap=0.25)
    patched = patched.to(device)
    print("✓ Created PatchedVAE wrapper")
    
    # Create dummy input (smaller than full-res for speed)
    x_full = torch.randn(1, 1, 128, 128, 128)
    print(f"✓ Created input volume: {x_full.shape}")
    
    # Test encoding
    print("\nTesting encode...")
    latents, positions = patched.encode(x_full, batch_size=2)
    print(f"✓ Encoded {len(positions)} patches")
    print(f"  Latent shape: {latents.shape}")
    print(f"  Positions sample: {positions[:3]}")
    
    # Test decoding
    print("\nTesting decode...")
    x_rec = patched.decode(latents, positions, tuple(x_full.shape[2:]), device)
    print(f"✓ Decoded back to volume: {x_rec.shape}")
    
    # Compute simple MSE
    mse = torch.mean((x_full.squeeze() - x_rec) ** 2)
    print(f"✓ Reconstruction MSE: {mse.item():.6f}")
    
    # Test full forward
    print("\nTesting full forward pass...")
    result = patched.forward(x_full, encode_only=False, batch_size=2)
    print(f"✓ Forward pass successful")
    print(f"  Latent shape: {result['latent'].shape}")
    print(f"  Reconstruction shape: {result['reconstruction'].shape}")
    
    print("\n" + "="*70)
    print(" ✓ All PatchedVAE tests passed!")
    print("="*70)

if __name__ == "__main__":
    test_patched_vae()
