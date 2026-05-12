#!/usr/bin/env python3
"""
Smoke test for all 3 VAE architectures.
Tests 1 training step on CPU for each VAE.
"""

import subprocess
import sys
from pathlib import Path

def run_test(vae_type: str, steps: int = 1) -> bool:
    """Run smoke test for a single VAE."""
    print(f"\n{'='*70}")
    print(f"  Testing {vae_type.upper()} (1 step, CPU)")
    print(f"{'='*70}")

    cmd = [
        sys.executable,
        "src/train_vae.py",
        f"--vae={vae_type}",
        f"--steps={steps}",
        "--batch-size=1",
        "--device=cpu",
        "--max-samples=1",
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print(f"✓ {vae_type.upper()} smoke test PASSED")
        return True
    else:
        print(f"✗ {vae_type.upper()} smoke test FAILED")
        return False


def main():
    vae_types = ["aekl", "vqvae", "medvae"]
    results = {}

    for vae_type in vae_types:
        try:
            results[vae_type] = run_test(vae_type)
        except Exception as e:
            print(f"✗ {vae_type.upper()} ERROR: {e}")
            results[vae_type] = False

    # Summary
    print(f"\n{'='*70}")
    print(f"  SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    for vae_type, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{vae_type.upper():10s} {status}")
    print(f"{'='*70}\n")

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
