import torch, json, sys, yaml, random
sys.path.insert(0, "src")
from cfm.arch_vector import build_vector_mmfm
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open("configs/mmfm/vectorized.yaml"))
ck = torch.load("outputs/mmfm/vectorized/weights/model_final.pth", map_location="cpu", weights_only=False)
model = build_vector_mmfm(cfg, 129024, 3).to(dev).eval(); model.load_state_dict(ck["ema"])
root = "outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train"
idx = json.load(open(root + "/index.json"))["samples"]; base = "outputs/mmfm/latent_cache/vectorized/"
def load(mod, field, n, seed=0):
    s = [r for r in idx if r["modality"] == mod and r["field"] == field]
    random.Random(seed).shuffle(s)
    return torch.stack([torch.as_tensor(torch.load(base+r["path"], map_location="cpu", weights_only=False)).float().reshape(-1) for r in s[:n]])

N = 8
print("Sensibilite de la vitesse v(z, z_src, t, y) au TEMPS et au SUJET")
print("(z fixe, seul t varie : le modele voit-il t ?)\n")
for mi, mod in enumerate(["T1W", "T2W", "T2FLAIR"]):
    z0 = load(mod, "0.1T", N).to(dev); y = torch.full((N,), mi, dtype=torch.long, device=dev)
    vs = []
    with torch.no_grad():
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tv = torch.full((N,), t, device=dev)
            vs.append(model(z0, z0, tv, y).float())
    V = torch.stack(vs)                              # (5,N,D)
    vbar = V.mean(0, keepdim=True)
    rel = ((V - vbar).norm(dim=2) / vbar.norm(dim=2)).mean().item()
    # cos entre v(t=0) et v(t=1)
    c = torch.nn.functional.cosine_similarity(V[0], V[-1], dim=1).mean().item()
    # comparaison : sensibilite au SUJET (z varie, t fixe)
    with torch.no_grad():
        tv = torch.full((N,), 0.5, device=dev); Vs = model(z0, z0, tv, y).float()
    rel_subj = ((Vs - Vs.mean(0, keepdim=True)).norm(dim=1) / Vs.mean(0).norm()).mean().item()
    print(f"  {mod:8s} variation relative de v selon t : {100*rel:6.2f} %   "
          f"cos(v(t=0), v(t=1)) = {c:.6f}")
    print(f"           {'':21s} selon le SUJET : {100*rel_subj:6.2f} %")
