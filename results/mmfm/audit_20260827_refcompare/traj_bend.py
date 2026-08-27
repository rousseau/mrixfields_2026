"""La trajectoire du modele entraine peut-elle se COURBER ?

Compare la deviation a la corde de (a) la trajectoire produite par le modele,
(b) les marginales reelles (deja mesuree a 0.72-1.32x la corde dans l'audit).
"""
import torch, json, glob, sys, yaml, random
sys.path.insert(0, "src")
from cfm.arch_vector import build_vector_mmfm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open("configs/mmfm/vectorized.yaml"))
ck = torch.load("outputs/mmfm/vectorized/weights/model_final.pth", map_location="cpu", weights_only=False)
D = 129024
model = build_vector_mmfm(cfg, D, 3).to(dev).eval()
model.load_state_dict(ck["ema"]); print("EMA chargee.")

root = "outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train"
idx = json.load(open(root + "/index.json"))["samples"]
base = "outputs/mmfm/latent_cache/vectorized/"
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]

def load(mod, field, n, seed=0):
    s = [r for r in idx if r["modality"] == mod and r["field"] == field]
    random.Random(seed).shuffle(s)
    out = []
    for r in s[:n]:
        o = torch.load(base + r["path"], map_location="cpu", weights_only=False)
        z = o["latent"] if isinstance(o, dict) and "latent" in o else o
        out.append(torch.as_tensor(z).float().reshape(-1))
    return torch.stack(out)

def bend(pts):
    """pts: (5, D) -> deviation de chaque point intermediaire a la corde p0->p4,
    normalisee par la longueur de la corde."""
    p0, p4 = pts[0], pts[-1]
    chord = p4 - p0; L = chord.norm()
    devs = []
    for k in (1, 2, 3):
        alpha = k / 4.0
        devs.append(((pts[k] - (p0 + alpha * chord)).norm() / L).item())
    return devs, L.item()

N = 8
for mod in ["T1W", "T2W", "T2FLAIR"]:
    y = torch.full((N,), ["T1W","T2W","T2FLAIR"].index(mod), dtype=torch.long, device=dev)
    z0 = load(mod, "0.1T", N).to(dev)
    # --- trajectoire du MODELE : integration 0 -> 1, on note z aux 5 ancres
    z = z0.clone(); z_anchor = z0.clone()
    n_steps, snaps = 200, {0: z0.clone()}
    dt = 1.0 / n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t = i * dt
            tv = torch.full((N,), t, device=dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
                v = model(z, z_anchor, tv, y)
            z = z + dt * v.float()
            if (i + 1) % (n_steps // 4) == 0:
                snaps[(i + 1) // (n_steps // 4)] = z.clone()
    traj = torch.stack([snaps[k] for k in range(5)], dim=1)   # (N,5,D)
    md = [bend(traj[b].cpu()) for b in range(N)]
    mdev = torch.tensor([m[0] for m in md]).mean(0)

    # --- marginales REELLES : moyennes de classe (couplage OT indisponible sans appariement)
    real = torch.stack([load(mod, f, 32, seed=1).mean(0) for f in FIELDS])
    rdev, _ = bend(real)

    print(f"\n=== {mod} ===")
    print(f"  MODELE  deviation/corde aux ancres 1.5T/3T/5T : "
          f"{mdev[0]:.3f}  {mdev[1]:.3f}  {mdev[2]:.3f}")
    print(f"  DONNEES (moyennes de classe)                  : "
          f"{rdev[0]:.3f}  {rdev[1]:.3f}  {rdev[2]:.3f}")
    print(f"  ratio courbure modele / donnees               : "
          f"{mdev[0]/rdev[0]:.3f}  {mdev[1]/rdev[1]:.3f}  {mdev[2]/rdev[2]:.3f}")
