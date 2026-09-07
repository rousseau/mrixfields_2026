#!/usr/bin/env python3
"""Banc de validation de la couche de REPRESENTATION (MedVAE), source par source.

POURQUOI CE SCRIPT EXISTE
Le "plafond de representation" mesure jusqu'ici (nRMSE 0.1048) est produit par une
inference identite qui traverse TOUTE la chaine : reechantillonnage 0.5 -> 1mm,
normalisation avec ecretage aux percentiles, crop, encodage TUILE par tuile en
bfloat16, decodage, denormalisation, retour en 0.5mm. Ce chiffre agrege donc au
moins six termes d'erreur, et rien dans le depot ne dit lequel domine. Tant que
cette decomposition n'existe pas, "le VAE plafonne a 0.10" n'est pas une
affirmation sur le VAE : c'est une affirmation sur la chaine.

Ce banc mesure chaque terme separement, avec le MEME code et les MEMES volumes :

  A. TEMOIN SANS VAE  (`--arms noop,protocol`) : la chaine complete avec l'identite
     a la place du VAE. C'est le plancher impose par le protocole seul.
  B. VAE SEUL         (colonnes `*_norm`) : encode -> decode compare a l'entree
     normalisee, dans l'espace de travail. Aucun reechantillonnage, aucun ecretage,
     aucune denormalisation n'entre dans ce chiffre.
  C. CHAINE COMPLETE  (colonnes `*_slab`, `*_full`) : le chiffre comparable au
     classement, obtenu par le meme chemin geometrique que
     `infer_mmfm_unified.process_volume_unified` avec le flow retire.

La difference C - A est le cout reel du VAE de bout en bout ; B est sa fidelite
intrinseque. Les deux sont necessaires : un VAE excellent (B bas) derriere un
protocole destructeur (A haut) donne le meme C qu'un VAE mediocre.

REGION NOTEE. Par defaut les metriques de bout en bout sont calculees sur la
tranche axiale [150, 180) que le classement note reellement (common/io.py
::Z_CLIP_RANGE), pas sur les 364 coupes. Les colonnes `*_full` donnent le volume
entier pour comparaison avec les chiffres du CHANGELOG anterieurs au 2026-09-04.

Usage :
    PYTHONPATH=src python src/cfm/bench_representation.py \
        --arms prod,prod_mode,prod_fp32,official,noop \
        --subjects 0006 --outdir results/mmfm/representation_20260906

    # matrice complete sur les 3 sujets d'evaluation
    PYTHONPATH=src python src/cfm/bench_representation.py --arms all --subjects 0006,0007,0009
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import nibabel.processing as nib_proc
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import Z_CLIP_RANGE, apply_z_clip, center_crop_or_pad_np
from models.tiled_vae import tiled_encode, tiled_decode

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
DEFAULT_SUBJECTS = ["0006"]
SPLIT_DIR = "Training_prospective"

FIELD_NORM_STATS = Path("configs/mmfm/field_norm_stats.json")

# Geometrie de production (configs/mmfm/vectorized.yaml)
PROD_SPACING = (1.0, 1.0, 1.0)
PROD_VOLUME = (192, 224, 192)
PROD_TILE = (96, 112, 96)
PROD_MARGIN = 16


# --------------------------------------------------------------------------- #
#  Metriques — formules officielles (verifiees bit a bit, ecart max 2.2e-16,
#  cf. eval_quantitative_3arch.py)
# --------------------------------------------------------------------------- #


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = pred.astype(np.float64), target.astype(np.float64)
    n = np.linalg.norm(t)
    return float(np.linalg.norm(p - t) / n) if n > 1e-10 else 0.0


def ssim3d(pred: np.ndarray, target: np.ndarray, slice_axis: int = 2) -> float:
    p, t = pred.astype(np.float64), target.astype(np.float64)
    dr = t.max() - t.min()
    if dr < 1e-10:
        return 1.0
    vals = []
    for i in range(p.shape[slice_axis]):
        s = [slice(None)] * p.ndim
        s[slice_axis] = i
        s = tuple(s)
        if t[s].max() - t[s].min() < 1e-10:
            continue
        vals.append(structural_similarity(p[s], t[s], data_range=dr))
    return float(np.mean(vals)) if vals else 1.0


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


# --------------------------------------------------------------------------- #
#  Normalisations
# --------------------------------------------------------------------------- #


def norm_fixed(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Recette de PRODUCTION : bornes fixes par (modalite, champ), ecretage."""
    v = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (v * 2.0 - 1.0).astype(np.float32)


def norm_percentile(vol: np.ndarray, p_lo: float, p_hi: float) -> Tuple[np.ndarray, float, float]:
    lo, hi = float(np.percentile(vol, p_lo)), float(np.percentile(vol, p_hi))
    return norm_fixed(vol, lo, hi), lo, hi


def norm_minmax(vol: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Recette OFFICIELLE MedVAE (utils/loaders.py::load_mri_3d) :
    ScaleIntensity(minv=0, maxv=1) puis Normalize(mean=.5, std=.5).
    Aucun ecretage : la chaine reste exactement inversible."""
    lo, hi = float(vol.min()), float(vol.max())
    return norm_fixed(vol, lo, hi), lo, hi


def denorm(vol_pm1: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """[-1,1] -> [0,1] -> intensites natives. Miroir exact de la normalisation."""
    v01 = (np.clip(vol_pm1, -1.0, 1.0) + 1.0) / 2.0
    return (v01 * (hi - lo) + lo).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Crops
# --------------------------------------------------------------------------- #


def foreground_bbox(vol_pm1: np.ndarray, k_divisible: int = 16,
                    threshold: float = 0.0) -> Tuple[slice, slice, slice]:
    """Equivalent de MONAI CropForeground(select_fn=is_positive, k_divisible=16),
    tel que la recette officielle MedVAE l'applique APRES normalisation."""
    mask = vol_pm1 > threshold
    if not mask.any():
        return tuple(slice(0, s) for s in vol_pm1.shape)
    out = []
    for ax in range(3):
        proj = mask.any(axis=tuple(a for a in range(3) if a != ax))
        idx = np.where(proj)[0]
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        size = hi - lo
        pad = (-size) % k_divisible
        lo = max(0, lo - pad // 2)
        hi = min(vol_pm1.shape[ax], lo + size + pad)
        lo = max(0, hi - size - pad)
        out.append(slice(lo, hi))
    return tuple(out)


# --------------------------------------------------------------------------- #
#  Definition des variantes
# --------------------------------------------------------------------------- #


@dataclass
class Arm:
    name: str
    doc: str
    use_vae: bool = True
    norm: str = "field_fixed"          # field_fixed | volume_pct | minmax
    spacing: Optional[Tuple[float, float, float]] = PROD_SPACING
    crop: str = "center"               # center | foreground | none
    volume_size: Tuple[int, int, int] = PROD_VOLUME
    encode: str = "tiled"              # tiled | medvae_sw
    tile: Tuple[int, int, int] = PROD_TILE
    margin: int = PROD_MARGIN
    amp: str = "bf16"                  # bf16 | fp32
    latent: str = "sample"             # sample | mode
    model_name: str = "medvae_4_1_3d"
    ckpt: Optional[str] = None
    p_lo: float = 0.5
    p_hi: float = 99.5
    hi_scale: float = 1.0              # multiplie le `hi` de field_norm_stats
    # Seuil au-delà duquel `MVAE.encode` re-découpe l'entrée en fenêtres glissantes
    # (medvae/utils/extras.py::roi_size_calc). Défaut amont : 160. PIÈGE MESURÉ :
    # une tuile élargie par une marge de 32 fait 160x176x160, donc l'axe W (176 > 160)
    # est re-découpé en fenêtres de 88 — le bras « marge 32 » mesurait alors
    # « marge 32 + sous-tuilage », pas la marge. À marge 16 la tuile fait 128x144x128,
    # tout passe sous 160, et la mesure est propre.
    gpu_dim: int = 160


ARMS = {a.name: a for a in [
    # --- temoins sans VAE ---------------------------------------------------
    Arm("noop", "aucun traitement : charge le volume et le masque. Doit donner 0.",
        use_vae=False, spacing=None, crop="none", norm="none"),
    Arm("protocol", "chaine complete SANS VAE : 1mm + ecretage percentile + crop + retour 0.5mm. "
        "Plancher impose par le protocole seul.", use_vae=False),
    Arm("protocol_noclip", "meme chaine sans ecretage (normalisation min-max) : isole le cout du "
        "reechantillonnage seul.", use_vae=False, norm="minmax"),
    Arm("protocol_nocrop", "chaine sans VAE et sans crop 192x224x192 : isole le cout du crop.",
        use_vae=False, crop="none"),
    # Le plancher protocolaire de la recette CORRIGEE. Sans lui, comparer `lpips_hi125`
    # au temoin `protocol` melange deux choses : le VAE et l'ecretage. `protocol` est le
    # plancher de la recette de PRODUCTION (hi x1.0) ; celui-ci est le plancher de la
    # recette a hi x1.25, et c'est a lui qu'il faut comparer le correctif.
    Arm("protocol_hi125", "chaine complete SANS VAE, avec `hi` x1.25 : le plancher de la "
        "recette corrigee.", use_vae=False, hi_scale=1.25),

    # --- production et ses variantes une-a-une ------------------------------
    Arm("prod", "recette de PRODUCTION exacte : field_fixed + 1mm + crop centre + tuiles + bf16 + "
        "echantillon de la posterieure."),
    Arm("prod_mode", "production, mais latent = MODE de la posterieure (deterministe).",
        latent="mode"),
    Arm("prod_fp32", "production en float32 au lieu de bfloat16.", amp="fp32"),
    Arm("prod_mode_fp32", "production deterministe et float32 : les deux correctifs gratuits.",
        latent="mode", amp="fp32"),
    Arm("prod_margin32", "production avec marge de contexte 32 voxels au lieu de 16. ATTENTION : "
        "MedVAE re-decoupe la tuile elargie (176 > gpu_dim=160) — ce bras mesure "
        "« marge 32 + sous-tuilage », pas la marge. Voir prod_margin32_clean.", margin=32),
    Arm("prod_margin32_clean", "marge 32 avec gpu_dim releve a 256 : une seule fenetre, la marge "
        "est enfin la SEULE variable qui change par rapport a `prod`.",
        margin=32, gpu_dim=256, latent="mode", amp="fp32"),
    Arm("prod_margin16_clean", "temoin apparie du precedent : marge 16, gpu_dim 256, mode, fp32.",
        margin=16, gpu_dim=256, latent="mode", amp="fp32"),
    Arm("prod_sw", "production, mais encodage par la fenetre glissante gaussienne OFFICIELLE de "
        "MedVAE au lieu du tuilage maison.", encode="medvae_sw"),

    # --- recette officielle MedVAE -----------------------------------------
    Arm("official", "recette OFFICIELLE MedVAE : min-max global, CropForeground(k=16), fenetre "
        "glissante gaussienne, float32, mode.", norm="minmax", crop="foreground",
        encode="medvae_sw", amp="fp32", latent="mode"),
    Arm("official_1mm_pct", "recette officielle mais avec l'ecretage percentile du projet : isole "
        "le cout de la normalisation.", crop="foreground", encode="medvae_sw", amp="fp32",
        latent="mode"),

    # --- occupation de la dynamique ----------------------------------------
    # MedVAE a ete entraine sur des entrees min-max, qui remplissent [-1,1]. La
    # question est de savoir si son erreur ABSOLUE est constante (auquel cas
    # remplir la dynamique divise l'erreur relative) ou proportionnelle au signal.
    Arm("prod_volpct", "percentiles PAR VOLUME au lieu des bornes fixes par champ : chaque volume "
        "remplit [-1,1] a lui seul.", norm="volume_pct", latent="mode", amp="fp32"),
    Arm("prod_p999", "bornes fixes par champ mais percentile haut 99.9 (moins d'ecretage).",
        norm="volume_pct", p_hi=99.9, latent="mode", amp="fp32"),
    Arm("prod_hi050", "bornes fixes par champ, `hi` divise par 2 : le signal remplit deux fois "
        "plus la dynamique, au prix d'un ecretage double.", hi_scale=0.5, latent="mode", amp="fp32"),
    Arm("prod_hi075", "bornes fixes par champ, `hi` x0.75.", hi_scale=0.75, latent="mode", amp="fp32"),
    Arm("prod_hi125", "bornes fixes par champ, `hi` x1.25 : moins d'ecretage, dynamique moins "
        "remplie. L'autre cote du compromis.", hi_scale=1.25, latent="mode", amp="fp32"),
    Arm("prod_hi150", "bornes fixes par champ, `hi` x1.5.", hi_scale=1.5, latent="mode", amp="fp32"),
    Arm("prod_hi200", "bornes fixes par champ, `hi` x2.", hi_scale=2.0, latent="mode", amp="fp32"),

    # --- resolution ---------------------------------------------------------
    Arm("native05", "encodage a la resolution NATIVE 0.5mm (pas de reechantillonnage), crop "
        "384x448x384.", spacing=None, volume_size=(384, 448, 384), latent="mode", amp="fp32"),

    # --- variantes de modele -----------------------------------------------
    # `tiled_encode` code en dur DOWNSAMPLE=4 (models/tiled_vae.py:41) : le 8x DOIT
    # passer par la fenetre glissante officielle, sinon les latents sont assembles
    # a la mauvaise echelle sans qu'aucune erreur ne soit levee.
    Arm("medvae8", "medvae_8_1_3d (compression 8x par dimension) au lieu de 4x.",
        model_name="medvae_8_1_3d", encode="medvae_sw", latent="mode", amp="fp32"),
    Arm("lpips", "MedVAE affine par le projet avec perte perceptuelle LPIPS "
        "(outputs/medvae/runs/medvae_finetune_lpips), recette de production.",
        ckpt="outputs/medvae/runs/medvae_finetune_lpips/weights/model_final.pth",
        latent="mode", amp="fp32"),
    Arm("lpips_p999", "MedVAE LPIPS + la meilleure normalisation trouvee (percentile 99.9 "
        "par volume).", ckpt="outputs/medvae/runs/medvae_finetune_lpips/weights/model_final.pth",
        norm="volume_pct", p_hi=99.9, latent="mode", amp="fp32"),
    # La SEULE combinaison a la fois meilleure et TRANSPOSABLE a la traduction : les
    # percentiles par volume sont indisponibles quand on ne connait pas le volume cible,
    # un `hi` fixe par (contraste, champ) l'est toujours.
    Arm("lpips_hi125", "MedVAE LPIPS + bornes fixes par champ avec `hi` x1.25.",
        ckpt="outputs/medvae/runs/medvae_finetune_lpips/weights/model_final.pth",
        hi_scale=1.25, latent="mode", amp="fp32"),
    Arm("lpips_hi150", "MedVAE LPIPS + bornes fixes par champ avec `hi` x1.5.",
        ckpt="outputs/medvae/runs/medvae_finetune_lpips/weights/model_final.pth",
        hi_scale=1.5, latent="mode", amp="fp32"),
]}


# --------------------------------------------------------------------------- #
#  Chargement du VAE
# --------------------------------------------------------------------------- #


_VAE_CACHE: dict = {}


def get_vae(arm: Arm, device):
    key = (arm.model_name, arm.ckpt)
    if key in _VAE_CACHE:
        return _VAE_CACHE[key]
    from models.maisi_vae import build_medvae_wrapper
    w = build_medvae_wrapper(model_name=arm.model_name, frozen=True, checkpoint=arm.ckpt)
    w = w.to(device).eval()
    _VAE_CACHE[key] = w
    return w


class DeterministicEncode:
    """Adaptateur exposant `encode`/`decode` avec le MODE de la posterieure.

    `MVAE.encode()` en 3D passe par `AutoencoderKL_3D.forward(sample_posterior=True)`
    et renvoie donc `posterior.sample()`, PAS le mode — malgre le commentaire
    "returns mode directly" de maisi_vae.py:130. Mesure : deux appels successifs
    sur la meme entree different, rapport bruit/signal 2.3e-3 du latent.
    """

    def __init__(self, wrapper):
        self.w = wrapper
        self.inner = wrapper.inner

    def encode(self, x):
        return self.inner.encode(x).mode()

    def decode(self, z):
        return self.inner.decode(z)


class SlidingWindowVAE:
    """Adaptateur utilisant la fenetre glissante gaussienne officielle de MedVAE
    (medvae_main.py::MVAE.encode/decode, roi_size_calc(target_gpu_dim=160))."""

    def __init__(self, wrapper, deterministic: bool):
        self.w = wrapper
        self.deterministic = deterministic

    def encode(self, x):
        from monai.inferers import sliding_window_inference
        from medvae.utils.extras import roi_size_calc
        inner = self.w.inner

        def _enc(patch):
            post = inner.encode(patch)
            return post.mode() if self.deterministic else post.sample()

        roi = roi_size_calc(list(x.shape[-3:]), target_gpu_dim=160)
        return sliding_window_inference(inputs=x, roi_size=roi, sw_batch_size=1,
                                        mode="gaussian", predictor=_enc)

    def decode(self, z):
        from monai.inferers import sliding_window_inference
        from medvae.utils.extras import roi_size_calc
        inner = self.w.inner
        cf = int(self.w.medvae.model_name.split("_")[1])
        roi = roi_size_calc([s * cf for s in z.shape[-3:]], target_gpu_dim=160)
        roi = [s // cf for s in roi]
        return sliding_window_inference(inputs=z, roi_size=roi, sw_batch_size=1,
                                        mode="gaussian", predictor=inner.decode)


# --------------------------------------------------------------------------- #
#  Le passage encode -> decode
# --------------------------------------------------------------------------- #


def roundtrip(arm: Arm, vol_norm: np.ndarray, device) -> np.ndarray:
    """(H,W,D) en [-1,1] -> reconstruction (H,W,D) en [-1,1], NON ecretee."""
    if not arm.use_vae:
        return vol_norm.copy()

    wrapper = get_vae(arm, device)
    wrapper.medvae.gpu_dim = arm.gpu_dim
    if arm.encode == "medvae_sw":
        vae = SlidingWindowVAE(wrapper, deterministic=(arm.latent == "mode"))
    elif arm.latent == "mode":
        vae = DeterministicEncode(wrapper)
    else:
        vae = wrapper

    x = torch.from_numpy(vol_norm).unsqueeze(0).unsqueeze(0).to(device)
    use_amp = (arm.amp == "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16

    with torch.no_grad():
        if arm.encode == "tiled":
            z = tiled_encode(vae, x, tile=arm.tile, margin=arm.margin,
                             use_amp=use_amp, amp_dtype=amp_dtype)
            rec = tiled_decode(vae, z, tile=arm.tile, margin=arm.margin,
                               use_amp=use_amp, amp_dtype=amp_dtype)
        else:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                z = vae.encode(x)
                rec = vae.decode(z)
    out = rec.squeeze().float().cpu().numpy()
    del x, z, rec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Une cellule : un volume x une variante
# --------------------------------------------------------------------------- #


def run_cell(arm: Arm, nii_path: Path, lo: float, hi: float, device,
             return_volumes: bool = False):
    """Reproduit la geometrie de `infer_mmfm_unified.process_volume_unified`
    avec le flow retire, et renvoie les trois familles de metriques.

    `return_volumes=True` renvoie en plus `(verite_native, prediction_native)`, sur
    la grille 0.5mm et deja masques — c'est ce que consomme
    `figures_representation_protocol.py`. Les figures empruntent ainsi EXACTEMENT le
    meme chemin que les chiffres : une figure produite par un code parallele finirait
    par illustrer autre chose que ce qu'on a mesure."""
    t0 = time.time()
    img_src = nib.load(str(nii_path))
    vol_native = img_src.get_fdata(dtype=np.float32)

    # 1) resolution de travail
    if arm.spacing is not None:
        img_work = nib_proc.resample_to_output(img_src, voxel_sizes=arm.spacing, order=1)
    else:
        img_work = nib_proc.resample_to_output(img_src, voxel_sizes=(0.5, 0.5, 0.5), order=1)
    vol_work = img_work.get_fdata(dtype=np.float32)

    # 2) normalisation
    if arm.norm == "field_fixed":
        n_lo, n_hi = lo, lo + (hi - lo) * arm.hi_scale
        vol_n = norm_fixed(vol_work, n_lo, n_hi)
    elif arm.norm == "volume_pct":
        vol_n, n_lo, n_hi = norm_percentile(vol_work, arm.p_lo, arm.p_hi)
    elif arm.norm == "minmax":
        vol_n, n_lo, n_hi = norm_minmax(vol_work)
    elif arm.norm == "none":
        vol_n, n_lo, n_hi = vol_work.astype(np.float32), 0.0, 1.0
    else:
        raise ValueError(arm.norm)

    # 3) crop
    native_shape = vol_n.shape
    bbox = None
    if arm.crop == "center":
        vol_in = center_crop_or_pad_np(vol_n, arm.volume_size)
    elif arm.crop == "foreground":
        bbox = foreground_bbox(vol_n, k_divisible=16)
        vol_in = np.ascontiguousarray(vol_n[bbox])
    else:
        vol_in = vol_n

    # 4) VAE
    if arm.norm == "none":
        rec_in = vol_in.copy()
    else:
        rec_in = roundtrip(arm, vol_in, device)

    # --- B. FIDELITE DU VAE SEUL, dans l'espace de travail ------------------
    rec_clip = np.clip(rec_in, -1.0, 1.0)
    fg = vol_in > -0.9
    m = {
        "nrmse_norm": nrmse(rec_clip, vol_in),
        "ssim_norm": ssim3d(rec_clip, vol_in),
        "psnr_norm": psnr(rec_clip, vol_in, data_range=2.0),
        "nrmse_norm_fg": nrmse(rec_clip[fg], vol_in[fg]) if fg.any() else 0.0,
        "fg_fraction": float(fg.mean()),
        "fg_std": float(vol_in[fg].std()) if fg.any() else 0.0,
    }

    # 5) retour a la geometrie de travail complete
    if arm.crop == "center":
        rec_work = center_crop_or_pad_np(rec_clip, native_shape)
    elif arm.crop == "foreground":
        rec_work = np.full(native_shape, -1.0, dtype=np.float32)
        rec_work[bbox] = rec_clip
    else:
        rec_work = rec_clip

    # 6) denormalisation puis retour sur la grille native 0.5mm
    if arm.norm == "none":
        pred_work = rec_work
    else:
        pred_work = denorm(rec_work, n_lo, n_hi)
    img_pred_work = nib.Nifti1Image(pred_work.astype(np.float32), img_work.affine)
    pred_native = nib_proc.resample_from_to(img_pred_work, img_src, order=1).get_fdata(dtype=np.float32)

    mask = vol_native > 1e-6
    pred_native = pred_native * mask

    # --- C. CHAINE COMPLETE, espace natif ----------------------------------
    m["nrmse_slab"] = nrmse(apply_z_clip(pred_native), apply_z_clip(vol_native))
    m["ssim_slab"] = ssim3d(apply_z_clip(pred_native), apply_z_clip(vol_native))
    m["nrmse_full"] = nrmse(pred_native, vol_native)
    m["ssim_full"] = ssim3d(pred_native, vol_native)
    m["seconds"] = time.time() - t0
    if return_volumes:
        return m, vol_native, pred_native
    return m


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="prod,protocol,noop",
                    help="liste separee par des virgules, ou 'all'. Disponibles : "
                         + ", ".join(ARMS))
    ap.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    ap.add_argument("--modalities", default=",".join(MODALITIES))
    ap.add_argument("--fields", default=",".join(FIELDS))
    ap.add_argument("--outdir", default="results/mmfm/representation")
    ap.add_argument("--env", default="local")
    ap.add_argument("--list", action="store_true", help="liste les variantes et sort")
    a = ap.parse_args()

    if a.list:
        for n, arm in ARMS.items():
            print(f"  {n:20s} {arm.doc}")
        return

    names = list(ARMS) if a.arms == "all" else [s.strip() for s in a.arms.split(",")]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"variantes inconnues : {unknown}\ndisponibles : {list(ARMS)}")

    subjects = [s.strip() for s in a.subjects.split(",")]
    modalities = [s.strip() for s in a.modalities.split(",")]
    fields = [s.strip() for s in a.fields.split(",")]

    root = Path(load_env(a.env)["data_root"]) / SPLIT_DIR
    stats = json.load(open(FIELD_NORM_STATS))["stats"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Banc de representation — region notee {Z_CLIP_RANGE}, device {device}")
    print(f"  {len(names)} variantes x {len(modalities)}x{len(fields)}x{len(subjects)} volumes")

    rows = []
    for name in names:
        arm = ARMS[name]
        print(f"\n=== {name} — {arm.doc}")
        t_arm = time.time()
        for mod in modalities:
            for fld in fields:
                for sid in subjects:
                    p = root / mod / fld / f"P_{mod}_{fld}_{sid}.nii.gz"
                    if not p.exists():
                        print(f"  [manquant] {p}")
                        continue
                    entry = stats[mod][fld]
                    try:
                        m = run_cell(arm, p, entry["lo"], entry["hi"], device)
                    except Exception as e:
                        print(f"  [ERREUR] {name} {mod} {fld} {sid}: {type(e).__name__}: {e}")
                        continue
                    rows.append({"arm": name, "modality": mod, "field": fld,
                                 "subject": sid, **m})
                    print(f"  {mod:8s} {fld:5s} {sid}  "
                          f"norm nRMSE {m['nrmse_norm']:.4f} SSIM {m['ssim_norm']:.4f} "
                          f"PSNR {m['psnr_norm']:5.2f} | slab nRMSE {m['nrmse_slab']:.4f} "
                          f"SSIM {m['ssim_slab']:.4f}  ({m['seconds']:.1f}s)", flush=True)
        print(f"  --- {name} termine en {(time.time()-t_arm)/60:.1f} min")

        # ecriture incrementale : un plantage tardif ne perd pas les mesures faites
        out = outdir / "cells.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # resume par variante
    summary = []
    for name in names:
        sel = [r for r in rows if r["arm"] == name]
        if not sel:
            continue
        agg = {"arm": name, "n": len(sel)}
        for k in ("nrmse_norm", "ssim_norm", "psnr_norm", "nrmse_norm_fg",
                  "nrmse_slab", "ssim_slab", "nrmse_full", "ssim_full", "fg_std", "seconds"):
            agg[k] = float(np.mean([r[k] for r in sel]))
        summary.append(agg)

    out_s = outdir / "summary.csv"
    with open(out_s, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print("\n" + "=" * 100)
    print(f"{'variante':22s} {'VAE seul nRMSE':>14s} {'SSIM':>7s} {'PSNR':>7s} "
          f"{'chaine slab':>12s} {'SSIM':>7s} {'full':>7s}")
    print("-" * 100)
    for s in summary:
        print(f"{s['arm']:22s} {s['nrmse_norm']:14.4f} {s['ssim_norm']:7.4f} "
              f"{s['psnr_norm']:7.2f} {s['nrmse_slab']:12.4f} {s['ssim_slab']:7.4f} "
              f"{s['nrmse_full']:7.4f}")
    print("=" * 100)
    print(f"CSV : {outdir}/cells.csv  et  {out_s}")


if __name__ == "__main__":
    main()
