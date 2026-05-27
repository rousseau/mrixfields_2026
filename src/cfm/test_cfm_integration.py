#!/usr/bin/env python3
"""Phase F smoke test: vérifier l'intégration load_vae + to_vector + from_vector
dans les pipelines CFM 3D et MMFM vectorisé.

Ce test valide sans checkpoint réel (poids aléatoires) et sans GPU :
  - encode(x) -> z
  - to_vector(z) -> z_vec  (flatten)
  - from_vector(z_vec) -> z_back  (round-trip shape)
  - decode(z_back) -> recon

Testé pour : aekl, pythae_vae, pythae_vqvae, pythae_rhvae, medvae_finetune

Usage:
    PYTHONPATH=src python src/cfm/test_cfm_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn

DEVICE = torch.device("cpu")
# Volume réduit pour tests rapides (divisible par 8 pour les conv strides)
VOLUME_SIZE = (32, 32, 32)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _dummy_volume(size=VOLUME_SIZE) -> torch.Tensor:
    return torch.randn(1, 1, *size)


def _run_cfm_pipeline(vae, vol: torch.Tensor, label: str):
    """Simule le pipeline train_cfm_3d : encode, flow step, decode."""
    vae.eval()
    with torch.no_grad():
        z = vae.encode(vol)
        assert z.ndim >= 2, f"[{label}] encode() retourne un tenseur de dim < 2"

        if vae.latent_format == "vector":
            # RHVAE ne doit pas passer par train_cfm_3d (garde levée à l'init)
            print(f"  [{label}] latent_format=vector -> train_cfm_3d refuse correctement (OK)")
            return

        # Simule un step de flow (z_t = z + bruit gaussien)
        z_t = z + 0.01 * torch.randn_like(z)
        recon = vae.decode(z_t)
        assert recon.shape == vol.shape, (
            f"[{label}] decode shape {recon.shape} != vol shape {vol.shape}"
        )
    print(f"  [{label}] CFM spatial OK  z={tuple(z.shape)}  recon={tuple(recon.shape)}")


def _run_mmfm_pipeline(vae, vol: torch.Tensor, label: str):
    """Simule le pipeline train_mmfm_3d : encode, to_vector, from_vector, decode."""
    vae.eval()
    with torch.no_grad():
        z = vae.encode(vol)

        # Phase F: API canonique to_vector / from_vector
        z_vec = vae.to_vector(z)
        assert z_vec.ndim == 2, (
            f"[{label}] to_vector() retourne dim {z_vec.ndim}, attendu 2"
        )

        # Round-trip shape
        z_back = vae.from_vector(z_vec)
        assert z_back.shape == z.shape, (
            f"[{label}] from_vector shape {z_back.shape} != encode shape {z.shape}"
        )

        # Simule un step de flow dans l'espace vectoriel
        z_t_vec = z_vec + 0.01 * torch.randn_like(z_vec)
        z_t = vae.from_vector(z_t_vec)
        recon = vae.decode(z_t)
        assert recon.shape == vol.shape, (
            f"[{label}] decode shape {recon.shape} != vol shape {vol.shape}"
        )

    print(
        f"  [{label}] MMFM vectorized OK  "
        f"z={tuple(z.shape)}  z_vec={tuple(z_vec.shape)}  "
        f"recon={tuple(recon.shape)}"
    )


# ---------------------------------------------------------------------------
# VAE builders (poids aléatoires, sans checkpoint)
# ---------------------------------------------------------------------------

def _build_aekl():
    from models.vae_wrappers import AEKLWrapper
    try:
        from monai.networks.nets import AutoencoderKL
    except ImportError:
        from generative.networks.nets import AutoencoderKL

    inner = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        channels=(8, 16),
        num_res_blocks=1,
        norm_num_groups=4,
        attention_levels=(False, False),
    )
    return AEKLWrapper(inner)


def _build_pythae_vae():
    from pythae.models.base.base_utils import ModelOutput as PythaeOutput
    from models.pythae_vae import PythaeVAE3D, Encoder3D, Decoder3D
    from pythae.models import VAE
    from pythae.models.base import BaseAE
    from pythae.models.nn import BaseEncoder, BaseDecoder

    # On utilise la factory via vae_loader pattern ou directement
    from pythae.models import VAEConfig
    cfg = VAEConfig(input_dim=(1,) + VOLUME_SIZE, latent_dim=8 * 4 * 4 * 4)
    enc = Encoder3D(cfg, in_channels=1, base_channels=8, latent_channels=4, num_groups=4)
    dec = Decoder3D(cfg, out_channels=1, base_channels=8, latent_channels=4, num_groups=4)
    return PythaeVAE3D(enc, dec, latent_channels=4)


def _build_pythae_vqvae():
    from models.pythae_vqvae import build_pythae_vqvae_3d
    return build_pythae_vqvae_3d(
        latent_channels=4,
        base_channels=8,
        num_embeddings=16,
        num_groups=4,
    )


def _build_pythae_rhvae():
    from models.pythae_rhvae import PythaeRHVAE3D
    return PythaeRHVAE3D(
        latent_dim=16,
        base_channels=8,
        num_groups=4,
        spatial_size=4,    # taille de la feature map après encode
        n_lf=1,
        eps_lf=0.01,
        beta_zero=0.3,
        temperature=1.5,
        regularization=0.01,
    )


def _build_medvae_finetune():
    """MedVAEFineTuneWrapper avec un inner model factice (sans HuggingFace).

    Reproduit la structure attendue par MedVAEFineTuneWrapper.__init__ :
      - mvae.encode(x) -> tensor (mode)
      - mvae.decode(z) -> tensor
      - mvae.model.encode(x) -> DiagonalGaussianDistribution
      - mvae.model.decode(z) -> tensor
    """
    from models.maisi_vae import MedVAEFineTuneWrapper

    class _FakeDist:
        def __init__(self, z):
            self._z = z
        def sample(self): return self._z
        def mode(self): return self._z
        def kl(self): return torch.zeros(self._z.shape[0])
        @property
        def mean(self): return self._z
        @property
        def logvar(self): return torch.zeros_like(self._z)

    class _FakeMVAEModel(nn.Module):
        """Simule medvae.MVAE.model (l'AutoencoderKL interne)."""
        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv3d(1, 4, 3, stride=2, padding=1)
            self.decoder = nn.ConvTranspose3d(4, 1, 4, stride=2, padding=1)

        def encode(self, x):
            return _FakeDist(self.encoder(x))

        def decode(self, z):
            return self.decoder(z)

    class _FakeMVAE(nn.Module):
        """Simule medvae.MVAE (top-level interface)."""
        def __init__(self):
            super().__init__()
            self.model = _FakeMVAEModel()

        def encode(self, x) -> torch.Tensor:
            # MVAE.encode() retourne le mode directement (tenseur)
            return self.model.encode(x).mode()

        def decode(self, z) -> torch.Tensor:
            return self.model.decode(z)

        def forward(self, x):
            return self.decode(self.encode(x))

    return MedVAEFineTuneWrapper(
        medvae_model=_FakeMVAE(),
        kl_weight=1e-6,
        frozen=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("aekl",            _build_aekl),
    ("pythae_vae",      _build_pythae_vae),
    ("pythae_vqvae",    _build_pythae_vqvae),
    ("pythae_rhvae",    _build_pythae_rhvae),
    ("medvae_finetune", _build_medvae_finetune),
]


def main():
    print("=" * 60)
    print("Phase F — Smoke test CFM integration (CPU, random weights)")
    print(f"Volume size: {VOLUME_SIZE}")
    print("=" * 60)

    failures = []

    for label, builder in TESTS:
        print(f"\n[{label}]")
        try:
            vae = builder().to(DEVICE)
            vol = _dummy_volume(VOLUME_SIZE).to(DEVICE)
            _run_cfm_pipeline(vae, vol, label)
            _run_mmfm_pipeline(vae, vol, label)
        except Exception as e:
            import traceback
            print(f"  [{label}] ECHEC: {e}")
            traceback.print_exc()
            failures.append(label)

    print("\n" + "=" * 60)
    if failures:
        print(f"ECHECS ({len(failures)}): {failures}")
        sys.exit(1)
    else:
        print(f"Tous les tests passent ({len(TESTS)} VAE types).")
        print("=" * 60)


if __name__ == "__main__":
    main()
