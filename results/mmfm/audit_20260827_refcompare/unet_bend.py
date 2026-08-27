import torch, json, sys, yaml, random
sys.path.insert(0, "src")
from cfm.arch_unet import build_unet_3d, remap_monai_attention_keys
dev = torch.device("cuda")
cfg = yaml.safe_load(open("configs/mmfm/unet.yaml")); cfg["model"]["use_adagn"] = False
ck = torch.load("outputs/mmfm/unet/weights/model_final.pth", map_location="cpu", weights_only=False)
m = build_unet_3d(cfg, 1, 3).to(dev).eval(); m.load_state_dict(remap_monai_attention_keys(ck["ema"]))
root = "outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train"
idx = json.load(open(root+"/index.json"))["samples"]; base = "outputs/mmfm/latent_cache/vectorized/"
def load(mod, field, n, seed=0):
    s = [r for r in idx if r["modality"]==mod and r["field"]==field]; random.Random(seed).shuffle(s)
    return torch.stack([torch.as_tensor(torch.load(base+r["path"],map_location="cpu",weights_only=False)).float().reshape(1,48,56,48) for r in s[:n]])
def bend(pts):
    p0,p4 = pts[0],pts[-1]; ch = p4-p0; L = ch.norm()
    return [((pts[k]-(p0+k/4.0*ch)).norm()/L).item() for k in (1,2,3)]
FIELDS=["0.1T","1.5T","3T","5T","7T"]
for mi,mod in [(1,"T2W")]:
    N=1; z0 = load(mod,"0.1T",N).to(dev); y = torch.full((N,),mi,dtype=torch.long,device=dev)
    z = z0.clone(); n_steps=20; dt=1.0/n_steps; snaps={0:z0.clone()}
    with torch.no_grad():
        for i in range(n_steps):
            v = m(x=torch.cat([z,z0],1), timesteps=torch.full((N,),i*dt,device=dev), class_labels=y).float()
            z = z + dt*v
            if (i+1)%(n_steps//4)==0: snaps[(i+1)//(n_steps//4)] = z.clone()
    traj = torch.stack([snaps[k].flatten(1) for k in range(5)],1).cpu()
    md = torch.tensor([bend(traj[b]) for b in range(N)]).mean(0)
    real = torch.stack([load(mod,f,16,seed=1).flatten(1).mean(0) for f in FIELDS])
    rd = bend(real)
    print(f"{mod}: MODELE {md[0]:.3f} {md[1]:.3f} {md[2]:.3f} | DONNEES {rd[0]:.3f} {rd[1]:.3f} {rd[2]:.3f} "
          f"| ratio {md[0]/rd[0]:.3f} {md[1]/rd[1]:.3f} {md[2]/rd[2]:.3f}", flush=True)
