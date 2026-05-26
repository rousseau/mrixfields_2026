#!/usr/bin/env python3
"""
Générateur de configs CFM3D — Consolide les configs YAML en un template + overrides

Usage :
    python src/utils/generate_cfm_configs.py

Crée :
    - configs/cfm3d_T1W_aekl.yaml
    - configs/cfm3d_T1W_vqvae.yaml
    - configs/cfm3d_T1W_medvae_frozen.yaml
    - configs/cfm3d_T1W_medvae_finetuned.yaml
    - configs/cfm3d_T1W_medvae_0p1T_7T.yaml
"""

from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    """Load YAML file (supports !include directive)."""
    with open(path) as f:
        content = f.read()
    return yaml.safe_load(content)


def save_yaml(path: str, data: dict):
    """Save YAML file with proper formatting."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge override into base (override takes precedence)."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def main():
    project_root = Path(__file__).resolve().parents[2]  # Go to repo root
    configs_dir = project_root / "configs"

    # Base config (template)
    base_path = configs_dir / "cfm3d_base.yaml"
    base_cfg = load_yaml(str(base_path))

    # Override specs (one per architecture)
    overrides = [
        {
            "name": "cfm3d_T1W_aekl.yaml",
            "task_name": "cfm3d_T1W_aekl",
            "vae": {"vae_type": "aekl", "source": "local"},
            "model": {"model_channels": 128},
            "train": {"batch_size": 2},
            "inference": {"source_domain": "0.1T"},
        },
        {
            "name": "cfm3d_T1W_vqvae.yaml",
            "task_name": "cfm3d_T1W_vqvae",
            "vae": {
                "vae_type": "vqvae",
                "source": "local",
                "checkpoint": "outputs/vqvae3d/runs/vqvae3d_T1W/weights/model_best.pth",
                "vae_config": "configs/vqvae3d_T1W.yaml",
            },
            "model": {"model_channels": 64, "num_head_channels": 32},
            "train": {"batch_size": 1},  # VQ-VAE plus lourd (64ch)
            "inference": {"source_domain": "0.1T"},
        },
        {
            "name": "cfm3d_T1W_medvae_frozen.yaml",
            "task_name": "cfm3d_T1W_medvae_frozen",
            "vae": {
                "vae_type": "medvae",
                "source": "frozen",
                "model_name": "medvae_4_1_3d",
            },
            "model": {"model_channels": 128},
            "train": {"batch_size": 2},
            "inference": {"source_domain": "0.1T"},
        },
        {
            "name": "cfm3d_T1W_medvae_finetuned.yaml",
            "task_name": "cfm3d_T1W_medvae_finetuned",
            "vae": {
                "vae_type": "medvae",
                "source": "local",
                "checkpoint": "outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth",
                "model_name": "medvae_4_1_3d",
            },
            "model": {"model_channels": 128},
            "train": {"batch_size": 2},
            "inference": {"source_domain": "0.1T"},
        },
        {
            "name": "cfm3d_T1W_medvae_0p1T_7T.yaml",
            "task_name": "cfm3d_T1W_medvae_0p1T_7T",
            "vae": {
                "vae_type": "medvae",
                "source": "local",
                "checkpoint": "outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth",
                "model_name": "medvae_4_1_3d",
            },
            "data": {"domains": ["0.1T", "7T"]},
            "train": {"total_iters": 100000},
            "inference": {"source_domain": "0.1T"},
        },
    ]

    print("=" * 60)
    print("Générateur de configs CFM3D")
    print("=" * 60)

    for override in overrides:
        name = override.pop("name")
        cfg = merge_dicts(base_cfg, override)
        cfg["task_name"] = override.get("task_name", name.replace(".yaml", ""))

        output_path = configs_dir / name
        save_yaml(str(output_path), cfg)
        print(f"✓ {name}")

    print()
    print("Configs générées :")
    for p in sorted(configs_dir.glob("cfm3d_*.yaml")):
        print(f"  - {p.relative_to(project_root)}")

    print()
    print("Usage :")
    print(
        "  python src/cfm/train_cfm_3d.py --config configs/cfm3d_T1W_aekl.yaml --env local"
    )


if __name__ == "__main__":
    main()
