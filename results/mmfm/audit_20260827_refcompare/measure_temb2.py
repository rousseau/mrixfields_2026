import torch, sys
sys.path.insert(0, "src")
from cfm.mmfm_vectorized import sinusoidal_time_embedding
T = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

for scale, tag in [(1.0, "ACTUEL"), (1000.0, "REFERENCE x1000")]:
    for dt, dtag in [(torch.float32, "fp32"), (torch.bfloat16, "bf16 (AMP)")]:
        e = sinusoidal_time_embedding(T * scale, 256).to(dt).float()
        ec = e - e.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(ec)[:4]
        # nb de champs encore distinguables : sv > eps*sv[0]
        print(f"{tag:16s} {dtag:11s} valeurs singulieres : " +
              " ".join(f"{v:9.2e}" for v in sv) +
              f"   | conditionnement {sv[0]/sv[3]:.1e}")
print()
# Part de variance a l'entree du reseau concatene
import yaml
cfg = yaml.safe_load(open("configs/mmfm/vectorized.yaml"))
print("latent_shape indices dans la config :", {k: v for k, v in cfg.get("model", {}).items() if "dim" in str(k)})
print("amp :", cfg.get("train", {}).get("amp_dtype", cfg.get("train", {}).get("use_amp")))
