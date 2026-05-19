#!/bin/bash
cd /home/rousseau/Exp/mrixfields_2026
python src/vae3d/visualize_ae_comparison.py --subject-idx 0 --output-dir results/qc/ae_comparison --dpi 150 --vqvae-training-patch 2>/dev/null
