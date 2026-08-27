"""Courbure et sensibilite au temps d'un checkpoint, sur latents REELS.

Les deux quantites qui ont revele le defaut du 2026-08-27, et qui se mesurent en
deux minutes -- contre ~6 h d'inference pour un score Task 3. A lancer AVANT
d'engager une evaluation complete : si cos(v(0),v(1)) vaut encore 1.000000, le
modele est aveugle au temps et le score ne fera que confirmer la production.

C'est ce qui a permis de conclure sur R1 (time_scale seul) sans attendre son
evaluation : cos = 1.000000 inchange, courbure 0.003-0.054 de celle exigee.

Usage :
    PYTHONPATH=src python src/cfm/measure_flow_geometry.py \
        --config configs/mmfm/vectorized_rbest.yaml --tag R-best
"""
import torch, json, sys, yaml, random, argparse
sys.path.insert(0, "src")
from cfm.arch_vector import build_vector_mmfm
ap=argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--tag")
a=ap.parse_args()
dev=torch.device("cuda"); cfg=yaml.safe_load(open(a.config))
sub=cfg["data"]["output_subdir"]
ck=torch.load(f"outputs/{sub}/weights/model_final.pth",map_location="cpu",weights_only=False)
m=build_vector_mmfm(cfg,129024,3).to(dev).eval(); m.load_state_dict(ck["ema"])
root="outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train"
idx=json.load(open(root+"/index.json"))["samples"]; base="outputs/mmfm/latent_cache/vectorized/"
FIELDS=["0.1T","1.5T","3T","5T","7T"]
def load(mod,f,n,seed=0):
    s=[r for r in idx if r["modality"]==mod and r["field"]==f]; random.Random(seed).shuffle(s)
    return torch.stack([torch.as_tensor(torch.load(base+r["path"],map_location="cpu",weights_only=False)).float().reshape(-1) for r in s[:n]])
def bend(p):
    p0,pl=p[0],p[-1]; ch=pl-p0; L=ch.norm()
    return [float(((p[k]-(p0+k/4.0*ch)).norm()/L).item()) for k in (1,2,3)]
print(f"=== {a.tag} (iter {ck.get('iter')}) ===")
for mi,mod in enumerate(["T1W","T2W","T2FLAIR"]):
    N=4; z0=load(mod,"0.1T",N).to(dev); y=torch.full((N,),mi,dtype=torch.long,device=dev)
    with torch.no_grad():
        v0=m(z0,z0,torch.zeros(N,device=dev),y).float(); v1=m(z0,z0,torch.ones(N,device=dev),y).float()
    c=torch.nn.functional.cosine_similarity(v0,v1,dim=1).mean().item()
    z=z0.clone(); ns=100; dt=1.0/ns; sn={0:z0.clone()}
    with torch.no_grad():
        for i in range(ns):
            with torch.amp.autocast("cuda",dtype=torch.bfloat16,enabled=True):
                v=m(z,z0,torch.full((N,),i*dt,device=dev),y)
            z=z+dt*v.float()
            if (i+1)%(ns//4)==0: sn[(i+1)//(ns//4)]=z.clone()
    tr=torch.stack([sn[k] for k in range(5)],1).cpu()
    md=torch.tensor([bend(tr[b]) for b in range(N)]).mean(0)
    real=torch.stack([load(mod,f,24,seed=1).mean(0) for f in FIELDS]); rd=bend(real)
    print(f"  {mod:8s} cos(v(0),v(1))={c:.6f} | courbure {md[0]:.3f} {md[1]:.3f} {md[2]:.3f}"
          f" | donnees {rd[0]:.3f} {rd[1]:.3f} {rd[2]:.3f} | fraction {md[1]/rd[1]:.3f}")
