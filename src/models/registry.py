# Registry of VAE models and their configurations for benchmarking and QC.

from pathlib import Path
from typing import List, Dict, Any, Tuple

# Constants for consistent benchmarking
PROSPECTIVE_SUBJECTS: List[str] = ["0006", "0007", "0009"]
RHVAE_VOLUME_SIZE: Tuple[int, int, int] = (128, 128, 128)
PATCH_SIZE: Tuple[int, int, int] = (112, 128, 80)
PATCH_OVERLAP: float = 0.25

# VAE Registry
# Each entry: (display_name, vae_cfg_dict, partial_flag, epoch_info)
VAE_REGISTRY = [
    (
        "AEKL_multimodal",
        {
            "vae": {
                "vae_type": "aekl",
                "source": "local",
                "checkpoint": "outputs/vae3d/runs/vae3d_multimodal/weights/model_best.pth",
                "vae_config": "configs/vae3d_multimodal.yaml",
            }
        },
        False,
        "ep~200",
    ),
    (
        "Pythae_VAE",
        {
            "vae": {
                "vae_type": "pythae_vae",
                "source": "local",
                "checkpoint": "outputs/pythae_vae3d/runs/pythae_vae3d_multimodal/weights/model_best.pth",
                "vae_config": "configs/pythae_vae_multimodal.yaml",
            }
        },
        False,
        "ep100",
    ),
    (
        "Pythae_VQVAE",
        {
            "vae": {
                "vae_type": "pythae_vqvae",
                "source": "local",
                "checkpoint": "outputs/pythae_vqvae3d/runs/pythae_vqvae3d_multimodal/weights/model_best.pth",
                "vae_config": "configs/pythae_vqvae_multimodal.yaml",
            }
        },
        False,
        "ep90",
    ),
    (
        "Pythae_RHVAE",
        {
            "vae": {
                "vae_type": "pythae_rhvae",
                "source": "local",
                "checkpoint": "outputs/pythae_rhvae3d/runs/pythae_rhvae3d_multimodal/weights/model_best.pth",
                "vae_config": "configs/pythae_rhvae_multimodal.yaml",
            }
        },
        False,
        "epoch100",
    ),
    (
        "MedVAE_frozen",
        {
            "vae": {
                "vae_type": "medvae",
                "source": "frozen",
                "model_name": "medvae_4_1_3d",
            }
        },
        False,
        "pretrained",
    ),
    (
        "MedVAE_finetuned",
        {
            "vae": {
                "vae_type": "medvae_finetune",
                "source": "local",
                "model_name": "medvae_4_1_3d",
                "frozen": True,
                "checkpoint": "outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth",
            }
        },
        False,
        "20k_steps",
    ),
    (
        "NV_Generate",
        {
            "vae": {
                "vae_type": "maisi",
                "source": "nvgenerate",
                "checkpoint": "outputs/nvgenerate/models/autoencoder_v2.pt",
            }
        },
        False,
        "ep351",
    ),
]
