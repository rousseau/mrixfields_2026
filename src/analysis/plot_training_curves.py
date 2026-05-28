#!/usr/bin/env python3
"""plot_training_curves.py — Visualisation des courbes d'apprentissage.

Découvre automatiquement tous les fichiers train_metrics.jsonl dans outputs/
et génère une figure comparative dans results/training_curves.png.

Usage :
    python src/analysis/plot_training_curves.py [OPTIONS]

Options :
    --outputs-dir PATH   Répertoire de base des outputs (défaut: outputs/)
    --output-fig PATH    Fichier de sortie (défaut: results/training_curves.png)
    --filter STR         Filtrer les runs dont le nom contient STR
    --smooth N           Fenêtre de lissage (moving average, défaut: 10)
    --max-iter N         Tronquer les courbes à N itérations
    --show               Ouvrir la figure dans une fenêtre (nécessite display)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Palette de couleurs distinctes ─────────────────────────────────────────
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _load_metrics(jsonl_path: Path) -> dict:
    """Charge un train_metrics.jsonl et retourne un dict de listes."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        return {}
    # Transposer en dict de listes
    keys = records[0].keys()
    return {k: [r.get(k) for r in records if k in r] for k in keys}


def discover_runs(outputs_dir: Path, filt: str | None) -> list[tuple[str, Path]]:
    """Retourne [(run_name, jsonl_path)] triés par run_name."""
    found = []
    for jsonl in sorted(outputs_dir.rglob("train_metrics.jsonl")):
        # run_name = répertoire immédiatement parent du jsonl
        run_name = jsonl.parent.name
        if filt and filt.lower() not in run_name.lower():
            continue
        found.append((run_name, jsonl))
    return found


def plot_curves(
    outputs_dir: Path,
    output_fig: Path,
    filt: str | None = None,
    smooth: int = 10,
    max_iter: int | None = None,
    show: bool = False,
) -> None:
    runs = discover_runs(outputs_dir, filt)

    if not runs:
        msg = f"Aucun train_metrics.jsonl trouvé dans {outputs_dir}"
        if filt:
            msg += f" (filtre: '{filt}')"
        print(msg)
        sys.exit(0)

    print(f"Runs trouvés : {len(runs)}")
    for name, path in runs:
        print(f"  {name}  ←  {path}")

    # ── Figure : 2 sous-graphes (loss + grad_norm) ──────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_loss, ax_grad = axes
    ax_loss.set_title("Courbes d'apprentissage", fontsize=13, fontweight="bold")
    ax_loss.set_ylabel("Loss (CFM/MSE)")
    ax_grad.set_ylabel("Gradient norm")
    ax_grad.set_xlabel("Itération")

    for i, (run_name, jsonl_path) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        data = _load_metrics(jsonl_path)
        if not data or "iter" not in data or "loss" not in data:
            print(f"  [WARN] {run_name} : données insuffisantes, ignoré")
            continue

        iters = np.array(data["iter"], dtype=float)
        loss  = np.array(data["loss"], dtype=float)
        grad  = np.array(data.get("grad_norm", [np.nan] * len(iters)), dtype=float)

        # Troncature
        if max_iter is not None:
            mask = iters <= max_iter
            iters, loss, grad = iters[mask], loss[mask], grad[mask]

        if len(iters) == 0:
            continue

        n_iters = len(iters)
        label = f"{run_name}  ({n_iters} pts)"

        # Courbe brute (transparente)
        ax_loss.plot(iters, loss, color=color, alpha=0.15, linewidth=0.8)
        ax_grad.plot(iters, grad, color=color, alpha=0.15, linewidth=0.8)

        # Lissage
        if n_iters >= smooth:
            loss_sm = _moving_average(loss.tolist(), smooth)
            grad_sm = _moving_average(grad.tolist(), smooth)
            iters_sm = iters[smooth - 1:]
            ax_loss.plot(iters_sm, loss_sm, color=color, linewidth=1.8, label=label)
            ax_grad.plot(iters_sm, grad_sm, color=color, linewidth=1.8)
        else:
            ax_loss.plot(iters, loss, color=color, linewidth=1.8, label=label)
            ax_grad.plot(iters, grad, color=color, linewidth=1.8)

        # Annotation dernière valeur
        ax_loss.annotate(
            f"{loss[-1]:.4f}",
            xy=(iters[-1], loss[-1]),
            fontsize=7, color=color,
            xytext=(4, 0), textcoords="offset points",
        )

    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.grid(True, alpha=0.3)
    ax_grad.grid(True, alpha=0.3)
    ax_grad.set_yscale("log")

    plt.tight_layout()
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=130, bbox_inches="tight")
    print(f"\nFigure sauvegardée : {output_fig}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-dir", default="outputs",
                   help="Répertoire racine des outputs (défaut: outputs/)")
    p.add_argument("--output-fig", default="results/training_curves.png",
                   help="Fichier figure de sortie")
    p.add_argument("--filter", default=None,
                   help="Filtrer les runs dont le nom contient cette chaîne")
    p.add_argument("--smooth", type=int, default=10,
                   help="Fenêtre moving average (défaut: 10, 1=désactivé)")
    p.add_argument("--max-iter", type=int, default=None,
                   help="Tronquer à N itérations max")
    p.add_argument("--show", action="store_true",
                   help="Ouvrir la figure (nécessite un display)")
    args = p.parse_args()

    plot_curves(
        outputs_dir=Path(args.outputs_dir),
        output_fig=Path(args.output_fig),
        filt=args.filter,
        smooth=args.smooth,
        max_iter=args.max_iter,
        show=args.show,
    )


if __name__ == "__main__":
    main()
