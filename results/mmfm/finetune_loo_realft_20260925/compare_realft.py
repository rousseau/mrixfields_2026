"""Plan 2x2 : recette LOO 0.2843 depuis {zero (= reference), production, GAN}.
Tests apparies par paire (20 paires x 3 contrastes, 8win, slab) :
  ftprod - scratch : un depart pre-entraine bat-il le depart de zero ?
  ftgan  - ftprod  : le gain du GAN survit-il au fine-tuning ? (test GAN principal)
  ftgan  - scratch : repere."""
import csv, os
import numpy as np
from scipy.stats import binomtest, wilcoxon

R = "results/mmfm"
RUNS = {
    "scratch": f"{R}/finetune_loo_long_lpips_20260921/task3_finetune_loo_mixed_v2_8win_{{m}}.csv",
    "ftprod":  f"{R}/finetune_loo_realft_20260925/task3_ftprod_loo_8win_{{m}}.csv",
    "ftgan":   f"{R}/finetune_loo_realft_20260925/task3_ftgan_loo_8win_{{m}}.csv",
    "prod":    f"{R}/finetune_loo_20260916/task3_prod8win_{{m}}.csv",
    "gan":     f"{R}/gan_palier4_task3_20260924/task3_gan5000_8win_{{m}}.csv",
}
MODS, MET = ("T1W", "T2W", "T2FLAIR"), ("nrmse", "ssim", "lpips")
TESTS = [("ftprod", "scratch"), ("ftgan", "ftprod"), ("ftgan", "scratch")]

def load(run, m):
    return {r["pair"]: {k: float(r[f"{k}_mean"]) for k in MET}
            for r in csv.DictReader(open(RUNS[run].format(m=m)))}

avail = [r for r in RUNS if all(os.path.exists(RUNS[r].format(m=m)) for m in MODS)]
tab = {r: {m: load(r, m) for m in MODS} for r in avail}
for r in avail:
    tot = {k: np.mean([tab[r][m][p][k] for m in MODS for p in tab[r][m]]) for k in MET}
    per = "  ".join(f"{m}:{np.mean([v['nrmse'] for v in tab[r][m].values()]):.4f}" for m in MODS)
    print(f"{r:8s} nrmse {tot['nrmse']:.4f}  ssim {tot['ssim']:.4f}  lpips {tot['lpips']:.4f}   | nRMSE {per}")
for a, b in TESTS:
    if a not in avail or b not in avail:
        continue
    print(f"\n=== {a} - {b} ===")
    for scope in MODS + ("TOUS",):
        mods = MODS if scope == "TOUS" else (scope,)
        for k in MET:
            cells = [(m, p) for m in mods for p in sorted(tab[a][m]) if p in tab[b][m]]
            d = np.array([tab[a][m][p][k] - tab[b][m][p][k] for m, p in cells])
            nz = d[d != 0]
            ps = binomtest(int((nz < 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
            pw = wilcoxon(d).pvalue if len(nz) else 1.0
            print(f"  {scope:8s} {k:6s} delta {d.mean():+.4f}  {a}<{b} {int((d < 0).sum())}/{len(d)}"
                  f"  signes p={ps:.3g}  Wilcoxon p={pw:.3g}")
