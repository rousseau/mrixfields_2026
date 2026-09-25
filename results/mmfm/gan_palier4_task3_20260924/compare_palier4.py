"""Palier 4 GAN latent : comparaison appariee par paire (20 paires x 3 contrastes,
8win, region slab officielle). GAN vs temoin au MEME budget (5000 iters) = test
principal ; production et reference 0.2843 = reperes (non a budget egal)."""
import csv, sys
import numpy as np
from scipy.stats import binomtest, wilcoxon

R = "results/mmfm"
RUNS = {
    "gan":     f"{R}/gan_palier4_task3_20260924/task3_gan5000_8win_{{m}}.csv",
    "temoin":  f"{R}/gan_palier4_task3_20260924/task3_control5000_8win_{{m}}.csv",
    "prod":    f"{R}/finetune_loo_20260916/task3_prod8win_{{m}}.csv",
    "ref2843": f"{R}/finetune_loo_long_lpips_20260921/task3_finetune_loo_mixed_v2_8win_{{m}}.csv",
}
MODS = ("T1W", "T2W", "T2FLAIR")
MET = ("nrmse", "ssim", "lpips")

def load(run, m):
    return {r["pair"]: {k: float(r[f"{k}_mean"]) for k in MET}
            for r in csv.DictReader(open(RUNS[run].format(m=m)))}

def test(a, b):
    d = np.array(a) - np.array(b)
    nz = d[d != 0]
    ps = binomtest(int((nz < 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
    pw = wilcoxon(d).pvalue if len(nz) else 1.0
    return d.mean(), int((d < 0).sum()), len(d), ps, pw

avail = [r for r in RUNS if all(__import__("os").path.exists(RUNS[r].format(m=m)) for m in MODS)]
print("runs disponibles :", avail)
tab = {r: {m: load(r, m) for m in MODS} for r in avail}
for r in avail:
    line = []
    for k in MET:
        v = [tab[r][m][p][k] for m in MODS for p in sorted(tab[r][m])]
        line.append(f"{k} {np.mean(v):.4f}")
    per = "  ".join(f"{m}:{np.mean([tab[r][m][p]['nrmse'] for p in tab[r][m]]):.4f}" for m in MODS)
    print(f"{r:8s} ({len(v)} cellules)  " + "  ".join(line) + "   | nRMSE " + per)

for base in [b for b in ("temoin", "prod", "ref2843") if b in avail and "gan" in avail]:
    print(f"\n=== GAN - {base} ===")
    for scope in MODS + ("TOUS",):
        mods = MODS if scope == "TOUS" else (scope,)
        for k in MET:
            pairs = [(m, p) for m in mods for p in sorted(tab["gan"][m]) if p in tab[base][m]]
            a = [tab["gan"][m][p][k] for m, p in pairs]; b = [tab[base][m][p][k] for m, p in pairs]
            dm, nneg, n, ps, pw = test(a, b)
            print(f"  {scope:8s} {k:6s} delta {dm:+.4f}  GAN<base {nneg}/{n}  signes p={ps:.3g}  Wilcoxon p={pw:.3g}")
