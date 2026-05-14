#!/usr/bin/env python3
"""
Script d'évaluation UNIFIÉ — MRIxFields 2026
=============================================
Évalue toutes les méthodes avec le même protocole, garantissant la cohérence
des comparaisons entre Étape 1 (StarGAN), Étape 2 (VAE), et Étape 3 (CFM).

Usage :
    # Évaluation visuelle (3 sujets prospectifs × 5 champs)
    python src/evaluation/evaluate.py \\
        --method stargan2d \\
        --checkpoint outputs/stargan2d/runs/task3_any_to_any_T1W/weights/model_final.pth \\
        --subjects prospective_5fields \\
        --output results/qc/

    # Évaluation quantitative (tableau incrémental)
    python src/evaluation/evaluate.py --mode quantitative \\
        --method aekl_cfm3d \\
        --vae-checkpoint outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \\
        --cfm-checkpoint outputs/cfm3d/runs/cfm3d_T1W/weights/model_final.pth \\
        --output results/evaluation_table.csv

Méthodes supportées :
    stargan2d           — StarGAN v2 2D (Étape 1)
    aekl_cfm3d          — AEKL + OT-CFM 3D (Étapes 2+3)
    vqvae_cfm3d         — VQ-VAE + OT-CFM 3D (Étapes 2+3)
    medvae_frozen_cfm   — MedVAE frozen + CFM (Étapes 2+3)
    medvae_ft_cfm       — MedVAE fine-tuné + CFM (Étapes 2+3)

Métriques (identiques au challenge) :
    nRMSE               — normalized Root Mean Square Error
    SSIM                — Structural Similarity Index
    LPIPS               — Learned Perceptual Image Patch Similarity
    Dice                — sur segmentations automatiques (FSL/FreeSurfer)
    VolumeConsistency   — consistance volumique normalisée

TODO : ce script est à implémenter lors de l'Étape 3.
"""

import argparse
import csv
import sys
from pathlib import Path


SUPPORTED_METHODS = [
    "stargan2d",
    "aekl_cfm3d",
    "vqvae_cfm3d",
    "medvae_frozen_cfm",
    "medvae_ft_cfm",
]

METRICS = ["nRMSE", "SSIM", "LPIPS", "Dice", "VolumeConsistency"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script d'évaluation unifié — MRIxFields 2026"
    )
    parser.add_argument(
        "--mode", choices=["visual", "quantitative", "both"], default="both",
        help="Mode d'évaluation : visuel, quantitatif, ou les deux"
    )
    parser.add_argument(
        "--method", required=True, choices=SUPPORTED_METHODS,
        help="Méthode à évaluer"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Chemin vers le checkpoint (StarGAN 2D)"
    )
    parser.add_argument(
        "--vae-checkpoint", type=str, default=None,
        help="Chemin vers le checkpoint VAE (AEKL / VQ-VAE / MedVAE)"
    )
    parser.add_argument(
        "--cfm-checkpoint", type=str, default=None,
        help="Chemin vers le checkpoint CFM 3D"
    )
    parser.add_argument(
        "--subjects", type=str, default="prospective_5fields",
        help="Sujets à évaluer (default: 3 sujets prospectifs × 5 champs)"
    )
    parser.add_argument(
        "--modality", type=str, default="T1W",
        choices=["T1W", "T2W", "T2FLAIR"],
        help="Modalité IRM à évaluer"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Répertoire de sortie (visual) ou fichier CSV (quantitative)"
    )
    parser.add_argument(
        "--env", type=str, default="local", choices=["local", "jeanzay"],
        help="Environnement d'exécution"
    )
    return parser.parse_args()


def evaluate_visual(args):
    """
    Évaluation visuelle sur les 3 sujets prospectifs × 5 champs magnétiques.
    Génère 1 figure par méthode × sujet × modalité montrant la traduction
    entre tous les champs (axiale, coronale, sagittale).

    Figures sauvegardées dans :
        results/qc/<methode>_<sujet>_<modalite>.png
    """
    raise NotImplementedError(
        "evaluate_visual() à implémenter — voir AGENTS.md Étape 3"
    )


def evaluate_quantitative(args):
    """
    Évaluation quantitative avec les métriques officielles du challenge.
    Met à jour results/evaluation_table.csv de façon incrémentale.

    Métriques : nRMSE, SSIM, LPIPS, Dice, VolumeConsistency
    """
    raise NotImplementedError(
        "evaluate_quantitative() à implémenter — voir AGENTS.md Étape 3"
    )


def main():
    args = parse_args()

    print(f"[evaluate.py] Méthode : {args.method}")
    print(f"[evaluate.py] Mode    : {args.mode}")
    print(f"[evaluate.py] Sortie  : {args.output}")
    print()
    print("⚠️  Ce script est un squelette — implémentation requise (Étape 3).")
    print("    Voir AGENTS.md pour le protocole d'évaluation unifié.")
    sys.exit(0)


if __name__ == "__main__":
    main()
