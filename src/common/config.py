#!/usr/bin/env python3
"""Configuration loading and environment resolution utilities.

Consolidated from:
  - src/cfm/train_cfm_3d.py
  - src/vae3d/train_vae_3d.py
  - src/cfm/train_mmfm_3d.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_with_include(path: str | Path) -> dict:
    """Load YAML with recursive !include directive support.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed dict.
    """
    path = Path(path)

    def _include(loader, node):
        include_path = Path(node.value)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        with open(include_path) as f:
            return yaml.safe_load(f)

    yaml.add_constructor("!include", _include)

    with open(path) as f:
        return yaml.safe_load(f)


def load_env(env_arg: Optional[str]) -> Optional[dict]:
    """Load environment configuration from YAML.

    Supports:
      - Absolute path
      - Relative path (tried from CWD and project root)
      - Named env (local, jeanzay, dgx -> configs/env/{name}.yaml)

    Args:
        env_arg: Path, name, or None.

    Returns:
        Dict with expanded env vars, or None.
    """
    if env_arg is None:
        return None

    env_path = env_arg if str(env_arg).endswith(".yaml") else f"configs/env/{env_arg}.yaml"
    if not os.path.isabs(env_path):
        if os.path.exists(env_path):
            env_path = os.path.abspath(env_path)
        else:
            project_root = Path(__file__).resolve().parents[2]
            candidate = project_root / env_path
            if candidate.exists():
                env_path = str(candidate)

    with open(env_path) as f:
        raw = yaml.safe_load(f)

    return {k: os.path.expandvars(str(v)) for k, v in raw.items()}


def resolve_paths(cfg: dict, env: Optional[dict]) -> dict:
    """Resolve data_root and output_dir from env config.

    Updates cfg in-place with:
      - data["data_root"] from env["data_root"]
      - data["output_dir"] from env["output_root"] / data["output_subdir"]
      - vae checkpoint paths from env["vae_root"]

    Args:
        cfg: Loaded config dict.
        env: Loaded env dict or None.

    Returns:
        Updated cfg dict.
    """
    if env is None:
        return cfg

    output_root = Path(env["output_root"])
    data = cfg.setdefault("data", {})

    if "output_subdir" in data:
        data["output_dir"] = str(output_root / data["output_subdir"])

    if "data_root" in env:
        data.setdefault("data_root", env["data_root"])

    # VAE checkpoint resolution
    vae_cfg = cfg.setdefault("vae", {})
    if "vae_root" in env and not vae_cfg.get("checkpoint"):
        subdir = data.get("output_subdir", "").replace("cfm3d", "vae3d").replace("mmfm", "vae3d")
        vae_cfg["checkpoint"] = str(
            Path(env["output_root"])
            / subdir.replace("cfm3d_latent", "vae3d").replace("mmfm3d", "vae3d")
            / "weights"
            / "model_best.pth"
        )

    return cfg


def save_yaml(cfg: dict, path: str | Path):
    """Save a config dict to YAML."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
