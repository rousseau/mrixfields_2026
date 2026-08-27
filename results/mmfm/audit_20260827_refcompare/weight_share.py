import torch, sys, glob, math
sys.path.insert(0, "src")
from cfm.mmfm_vectorized import sinusoidal_time_embedding

c = torch.load('outputs/mmfm/vectorized/weights/model_final.pth', map_location='cpu', weights_only=False)
W = c['ema']['input_proj.0.weight'].float()      # (1024, 258432)
D = 129024
blocks = {'z_t': (0, D), 'z_src': (D, 2*D), 'temps': (2*D, 2*D+256), 'contraste': (2*D+256, 2*D+384)}

# variance reelle de chaque canal d'entree
t_train = torch.rand(20000)                       # t ~ U[0,1] comme a l'entrainement
temb = sinusoidal_time_embedding(t_train, 256)
var_t = temb.var(0)                               # (256,)
cemb = c['ema']['class_embed.weight'].float()     # (3,128)
var_c = cemb.var(0)

# latents reels standardises : lire quelques caches
fs = glob.glob('outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train/*/*/*.pt')[:64]
zs = []
for f in fs:
    o = torch.load(f, map_location='cpu', weights_only=False)
    z = o['latent'] if isinstance(o, dict) and 'latent' in o else (o if torch.is_tensor(o) else list(o.values())[0])
    zs.append(torch.as_tensor(z).float().reshape(-1))
Z = torch.stack(zs)
lm = c.get('cfg_latent_mean', 0.0)
print(f"latents lus : {Z.shape}, std brut {Z.std():.3e}")
import yaml
cfg = yaml.safe_load(open('configs/mmfm/vectorized.yaml'))
mean = float(cfg['model'].get('latent_mean', 0.0)); scale = float(cfg['model'].get('latent_scale', 1.0))
print(f"standardisation config : mean={mean} scale={scale}")
Zn = (Z - mean) / scale
var_z = Zn.var(0)                                 # (129024,)
print(f"std du latent standardise : {Zn.std():.3f}")

print("\n--- part de variance apportee a la 1ere couche (modele ENTRAINE) ---")
tot = 0.0; contrib = {}
for name, (a, b) in blocks.items():
    Wb = W[:, a:b]
    v = {'z_t': var_z, 'z_src': var_z, 'temps': var_t, 'contraste': var_c}[name]
    cc = (Wb.pow(2) * v[None, :]).sum().item()
    contrib[name] = cc; tot += cc
for name in blocks:
    print(f"  {name:10s} : {contrib[name]:12.4e}  ({100*contrib[name]/tot:7.4f} %)   "
          f"||W|| moyen/canal {W[:, blocks[name][0]:blocks[name][1]].pow(2).mean().sqrt():.3e}")
