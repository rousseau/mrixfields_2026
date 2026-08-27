import torch, json, sys, yaml, copy, random
sys.path.insert(0, "src")
from cfm.arch_unet import build_unet_3d, remap_monai_attention_keys
dev = torch.device("cuda")
cfg = yaml.safe_load(open("configs/mmfm/unet.yaml")); cfg["model"]["use_adagn"] = False
ck = torch.load("outputs/mmfm/unet/weights/model_final.pth", map_location="cpu", weights_only=False)
m = build_unet_3d(cfg, 1, 3).to(dev).eval(); m.load_state_dict(remap_monai_attention_keys(ck["ema"]))
print("EMA UNet chargee (iter", ck.get("iter"), ")", flush=True)
root = "outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train"
idx = json.load(open(root+"/index.json"))["samples"]; base = "outputs/mmfm/latent_cache/vectorized/"
def load(mod, n=2):
    s = [r for r in idx if r["modality"]==mod and r["field"]=="0.1T"]; random.Random(0).shuffle(s)
    return torch.stack([torch.as_tensor(torch.load(base+r["path"],map_location="cpu",weights_only=False)).float().reshape(1,48,56,48) for r in s[:n]])
for mi, mod in enumerate(["T1W","T2W","T2FLAIR"]):
    z0 = load(mod).to(dev); y = torch.full((z0.shape[0],), mi, dtype=torch.long, device=dev)
    vs = []
    with torch.no_grad():
        for t in [0.0,0.25,0.5,0.75,1.0]:
            vs.append(m(x=torch.cat([z0,z0],1), timesteps=torch.full((z0.shape[0],),t,device=dev), class_labels=y).float())
    V = torch.stack(vs).flatten(2); vbar = V.mean(0,keepdim=True)
    rel = ((V-vbar).norm(dim=2)/vbar.norm(dim=2)).mean().item()
    c = torch.nn.functional.cosine_similarity(V[0],V[-1],dim=1).mean().item()
    Vs = V[2]; rs = ((Vs-Vs.mean(0,keepdim=True)).norm(dim=1)/Vs.mean(0).norm()).mean().item()
    print(f"  {mod:8s} v selon t : {100*rel:6.2f} %   cos(v(0),v(1)) = {c:.6f}   | selon le SUJET : {100*rs:6.2f} %", flush=True)
