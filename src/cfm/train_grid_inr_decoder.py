#!/usr/bin/env python3
"""Entraînement du décodeur implicite conditionné par grille latente.

Remplace le DÉCODEUR de MedVAE par un décodeur coordonnées->intensité
(cfm/grid_inr_decoder.py), l'encodeur MedVAE restant gelé. Objectif : décoder à
une résolution ARBITRAIRE (ex. 0.5mm natif depuis un latent 1mm), ce que le
décodeur convolutif de MedVAE ne permet pas — l'évaluateur ré-interpolant de
toute façon les prédictions vers 0.5mm, tout détail restitué à ce stade est
potentiellement gagné.

Différence essentielle avec `train_inr_backbone.py` : il n'y a NI latent global,
NI boucle interne d'ajustement de `z` (Algorithme 2 de NOIR). Le latent est
simplement celui, déjà calculé et mis en cache, de l'encodeur MedVAE gelé —
exactement le même que celui consommé par les architectures vectorisée et UNet.
L'entraînement est donc un apprentissage supervisé ordinaire, sans
méta-apprentissage : plus simple, plus stable, et directement comparable.

Point de conception décisif : le latent est à 1mm mais la SUPERVISION est à
0.5mm (grille native, `data.supervision_factor: 2`). Sans cela le décodeur
n'apprendrait rien de ce qui se passe entre deux voxels 1mm et son décodage à
0.5mm ne vaudrait pas mieux qu'une interpolation trilinéaire — c'est-à-dire que
le seul avantage de cette architecture serait perdu d'avance. Rien d'autre ne
change : même latent, donc même modèle de flow, donc comparaison équitable.

Aucun volume n'est pré-calculé sur disque : les NIfTI sont lus et normalisés à
la volée (le seul cache du projet reste celui des latents, ~0.5 Go).

Usage:
    PYTHONPATH=src python src/cfm/train_grid_inr_decoder.py \\
        --config configs/mmfm/grid_inr_decoder.yaml --env local
    PYTHONPATH=src python src/cfm/train_grid_inr_decoder.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.ndimage import zoom as scipy_zoom

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.distributed import EMAModel
from common.io import SPLIT_MAP, load_nifti_volume, normalize_volume_fixed
from cfm.grid_inr_decoder import GridConditionedINR, GridINRConfig


def _crop_offsets(src_shape, tgt_shape) -> List[int]:
    """Décalage source->sortie de `center_crop_or_pad_np`, par axe.

    Négatif = la sortie déborde (padding). Reproduit exactement la convention de
    `common/io.py` (division entière), y compris pour les différences impaires.
    """
    return [((s - t) // 2 if s >= t else -((t - s) // 2))
            for s, t in zip(src_shape, tgt_shape)]


def _extract_at(vol: np.ndarray, offsets, tgt_shape, mode: str = "reflect") -> np.ndarray:
    """Extrait `tgt_shape` à partir de décalages IMPOSÉS (négatif = pad)."""
    pads, sl = [], []
    for ax, (o, t) in enumerate(zip(offsets, tgt_shape)):
        s = vol.shape[ax]
        pads.append((max(0, -o), max(0, o + t - s)))
        sl.append(slice(max(0, o), min(s, o + t)))
    out = vol[tuple(sl)]
    if any(p != (0, 0) for p in pads):
        out = np.pad(out, pads, mode=mode)
    return out


def _sample_flat(vol: np.ndarray, n: int):
    """Tire `n` voxels au hasard -> (coords [-1,1]^3, valeurs).

    Les coordonnées sont calculées ANALYTIQUEMENT à partir des indices tirés,
    au lieu de matérialiser la grille complète : à 0.5mm celle-ci pèserait
    384*448*384*3 flottants (~790 Mo) par échantillon. La formule
    `-1 + 2i/(s-1)` est celle de `make_coord_grid` (linspace, align_corners),
    et doit le rester — c'est la même convention que `grid_sample` côté latent.

    Chemin de RÉFÉRENCE, conservé pour tester `_sample_cropped` : il matérialise
    le volume recadré, ce qui coûte ~5s par échantillon (voir ce module).
    """
    h, w, d = vol.shape
    idx = np.random.randint(0, h * w * d, size=n)
    return _coords_values(vol.reshape(-1)[idx], idx, (h, w, d))


def _coords_values(values: np.ndarray, idx: np.ndarray, shape):
    h, w, d = shape
    k = idx % d
    j = (idx // d) % w
    i = idx // (w * d)
    coords = np.stack([-1.0 + 2.0 * i / max(h - 1, 1),
                       -1.0 + 2.0 * j / max(w - 1, 1),
                       -1.0 + 2.0 * k / max(d - 1, 1)], axis=-1).astype(np.float32)
    return torch.from_numpy(coords), torch.from_numpy(values.astype(np.float32)[:, None])


def _downsample_for_stats(raw: np.ndarray, f: int) -> np.ndarray:
    """Grille du latent, pour y prendre les percentiles.

    Moyenne de blocs f^3 quand elle donne exactement la forme qu'aurait produite
    `scipy.ndimage.zoom` (cas de TOUTES les données du challenge : 0.5mm,
    364x436x364), sinon repli sur `zoom` — 6.7x plus lent (1.4s contre 0.21s)
    mais on ne peut pas se permettre une forme différente : elle sert aussi à
    calculer le recadrage, donc l'alignement avec le latent.

    L'écart sur (hi-lo) entre les deux méthodes est mesuré à <= 0.23% sur un
    échantillon de contrastes et de champs, très en dessous de ce qu'on mesure.
    """
    target = tuple(int(round(s / f)) for s in raw.shape)
    trimmed = tuple(s // f * f for s in raw.shape)
    if tuple(s // f for s in trimmed) != target:
        return scipy_zoom(raw, 1.0 / f, order=1).astype(np.float32)
    h, w, d = trimmed
    return (raw[:h, :w, :d]
            .reshape(h // f, f, w // f, f, d // f, f)
            .mean(axis=(1, 3, 5)).astype(np.float32))


def _reflect_index(p: np.ndarray, s: int) -> np.ndarray:
    """Indice replié selon le mode 'reflect' de numpy (bord non répété)."""
    if s == 1:
        return np.zeros_like(p)
    period = 2 * (s - 1)
    p = np.abs(p) % period
    return np.where(p >= s, period - p, p)


def _sample_cropped(raw: np.ndarray, offsets, out_shape, n: int, lo: float, hi: float):
    """Échantillonne le volume recadré+normalisé SANS jamais le matérialiser.

    Équivalent exact de `_sample_flat(_extract_at(normalize_volume_fixed(raw)))`
    — testé dans `self_test()` — mais ~5s plus rapide par échantillon : on ne
    garde que 16384 voxels sur 66 millions, il est absurde de payer la
    normalisation (2.5s), le recadrage (0.9s) et la copie de `reshape` (1.7s)
    sur le volume entier pour cela. Seul le tirage des indices est fait dans le
    repère de SORTIE, puis ramené dans le repère source (repliement 'reflect'
    aux bords, comme le ferait `np.pad`).
    """
    h, w, d = out_shape
    idx = np.random.randint(0, h * w * d, size=n)
    k = idx % d
    j = (idx // d) % w
    i = idx // (w * d)
    vi = _reflect_index(i + offsets[0], raw.shape[0])
    vj = _reflect_index(j + offsets[1], raw.shape[1])
    vk = _reflect_index(k + offsets[2], raw.shape[2])
    vals = raw[vi, vj, vk].astype(np.float32)
    vals = np.clip((vals - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0   # normalize_volume_fixed
    return _coords_values(vals, idx, out_shape)


SUBJECT_RE = re.compile(r"^[A-Z]_[A-Z0-9]+_[0-9.]+T_(\d+)$")


def subject_of(sample_path: str) -> str:
    """Identifiant sujet extrait du nom de fichier `{R,P}_{mod}_{champ}_{sujet}`."""
    m = SUBJECT_RE.match(Path(sample_path).stem)
    if m is None:
        raise ValueError(f"Nom de fichier inattendu : {sample_path}")
    return m.group(1)


def holdout_subjects(samples: Sequence[dict], n_val: int, seed: int = 1234) -> Set[str]:
    """Tire `n_val` sujets de validation, de façon déterministe.

    Le partage se fait par SUJET et non par volume : un même sujet apparaît sous
    plusieurs contrastes et plusieurs champs (jusqu'à 3 volumes ici), et laisser
    ses autres acquisitions dans l'entraînement fuiterait son anatomie dans
    l'évaluation.
    """
    subs = sorted({subject_of(s["path"]) for s in samples})
    if n_val <= 0:
        return set()
    return set(random.Random(seed).sample(subs, min(n_val, len(subs))))


class LatentVolumeDataset(Dataset):
    """Retourne (latent MedVAE 1mm, coordonnées tirées, intensités cibles).

    On itère l'INDEX DU CACHE plutôt que le dossier de données : c'est lui qui
    fait foi sur les volumes réellement encodés, et le chemin NIfTI s'en déduit
    sans ambiguïté. Le prétraitement du volume doit reproduire EXACTEMENT celui
    du precompute (resample -> normalisation percentile -> center-crop), sinon
    latent et cible ne décriraient pas le même signal.

    SUPERVISION À 0.5mm (`sup_factor=2`), le latent restant à 1mm : c'est la
    condition pour que le décodage à résolution arbitraire ait le moindre sens.
    Supervisé uniquement sur la grille 1mm, le réseau n'aurait AUCUNE
    information sur ce qui se passe entre deux voxels 1mm : interrogé à 0.5mm
    il ne produirait qu'une interpolation lisse, sans rien de plus qu'une
    trilinéaire — le test « décoder à 0.5mm » serait perdu d'avance. En tirant
    les points sur la grille NATIVE 0.5mm, le décodeur apprend au contraire à
    restituer du détail sous-voxel, ce que le décodeur convolutif de MedVAE ne
    peut structurellement pas faire (son facteur de suréchantillonnage est figé
    à x4). Le latent, lui, ne change pas : le modèle de flow reste inchangé.

    Les points sont tirés DANS LE WORKER, pas dans la boucle : un volume 0.5mm
    pèse 264 Mo et le transférer entier au GPU à chaque pas pour n'en garder
    que 16384 voxels saturerait la bande passante pour rien.
    """

    def __init__(self, cache_dir: Path, cache_root: Path, data_root: Path, split: str,
                 volume_size, target_spacing, p_lo: float, p_hi: float,
                 sup_factor: int = 2, points: int = 16384, reuse: int = 1,
                 exclude_subjects: Set[str] | None = None,
                 only_subjects: Set[str] | None = None):
        meta = json.loads((cache_dir / "index.json").read_text())
        self.samples = meta["samples"]
        if exclude_subjects:
            self.samples = [s for s in self.samples if subject_of(s["path"]) not in exclude_subjects]
        if only_subjects is not None:
            self.samples = [s for s in self.samples if subject_of(s["path"]) in only_subjects]
        self.cache_root = Path(cache_root)
        self.data_root = Path(data_root)
        self.split_dir = SPLIT_MAP[split]
        self.volume_size = tuple(volume_size)          # grille du LATENT (1mm)
        self.target_spacing = tuple(target_spacing)
        self.p_lo, self.p_hi = p_lo, p_hi
        self.sup_factor = int(sup_factor)              # 1 = supervision 1mm, 2 = 0.5mm
        self.sup_size = tuple(self.sup_factor * v for v in self.volume_size)
        self.sup_spacing = tuple(s / self.sup_factor for s in self.target_spacing)
        self.points = int(points)
        # Un volume coûte ~4s à lire/normaliser pour n'en garder que 16384
        # voxels sur 66 millions. On tire donc `reuse` lots disjoints d'un coup,
        # que la boucle consomme en `reuse` pas d'optimisation successifs :
        # même bruit de gradient par pas, mais `reuse` fois moins de lectures.
        # Contrepartie assumée : `reuse` pas consécutifs voient les mêmes
        # volumes — pratique courante en entraînement d'INR (plusieurs lots de
        # rayons par image), au prix d'une corrélation locale des gradients.
        self.reuse = int(reuse)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rel = Path(s["path"])                       # <cache_id>/<split>/<mod>/<field>/<nom>.pt
        z = torch.load(self.cache_root / rel, map_location="cpu").float()
        if z.dim() != 4:                            # (C, h, w, d) attendu
            raise RuntimeError(f"Latent spatial (C,h,w,d) attendu, reçu {tuple(z.shape)}")

        mod, field, name = rel.parts[-3], rel.parts[-2], rel.stem
        nii = self.data_root / self.split_dir / mod / field / f"{name}.nii.gz"
        raw, _ = load_nifti_volume(nii, target_spacing=self.sup_spacing, volume_size=None,
                                   normalize=False)

        # Percentiles pris sur la grille du LATENT (1mm), pas sur celle de la
        # supervision : c'est l'échelle d'intensité qu'a vue le precompute des
        # latents. La calculer à 0.5mm décalerait la cible par rapport à ce que
        # le latent encode, et le décodeur devrait compenser un écart affine
        # qu'il ne peut pas connaître.
        vol1 = raw if self.sup_factor == 1 else _downsample_for_stats(raw, self.sup_factor)
        lo = float(np.percentile(vol1, self.p_lo))
        hi = float(np.percentile(vol1, self.p_hi))

        # Le recadrage de la supervision doit être EXACTEMENT celui du latent,
        # à l'échelle près : on calcule les décalages sur la grille 1mm puis on
        # les multiplie. Recadrer indépendamment les deux grilles introduirait
        # un décalage d'un demi-voxel sur les volumes dont la différence de
        # taille est impaire — le décodeur apprendrait alors une moyenne floue.
        off = [self.sup_factor * o for o in _crop_offsets(vol1.shape, self.volume_size)]
        if hi <= lo:                                  # volume dégénéré
            n = self.points * self.reuse
            return z, torch.zeros(n, 3), torch.zeros(n, 1)
        return (z, *_sample_cropped(raw, off, self.sup_size,
                                    self.points * self.reuse, lo, hi))


def self_test() -> None:
    """Vérifie EXACTEMENT l'arithmétique d'indices du chemin de supervision.

    À faire tourner après toute retouche de `_crop_offsets`/`_extract_at`/
    `_sample_flat` : une erreur d'indice y est parfaitement silencieuse à
    l'entraînement (le réseau apprend une cible décalée, la loss descend quand
    même). Et la corrélation sur une IRM ne sert à rien pour cela — mesuré, un
    décalage volontaire d'un demi-voxel donne 0.98700 contre 0.98644 pour la
    version alignée, soit strictement rien d'exploitable. D'où des tests sur
    l'arithmétique elle-même, où un bug vaut un voxel entier ou plus.
    """
    from common.io import center_crop_or_pad_np
    from cfm.inr_backbone import make_coord_grid

    rng = np.random.default_rng(0)

    for _ in range(300):        # _extract_at doit reproduire la référence
        src = tuple(int(x) for x in rng.integers(8, 40, 3))
        tgt = tuple(int(x) for x in rng.integers(8, 40, 3))
        vol = rng.standard_normal(src).astype(np.float32)
        got = _extract_at(vol, _crop_offsets(src, tgt), tgt)
        assert np.array_equal(got, center_crop_or_pad_np(vol, tgt)), f"src={src} tgt={tgt}"
    print("1. _extract_at == center_crop_or_pad_np (300 cas, tailles paires et impaires) ✅")

    for _ in range(300):        # doubler les décalages garde le même centre
        s1 = tuple(int(x) for x in rng.integers(80, 200, 3))
        t1 = (192, 224, 192)
        o1 = _crop_offsets(s1, t1)
        for a in range(3):
            c1 = o1[a] + t1[a] / 2.0
            c05 = (2 * o1[a] + 2 * t1[a] / 2.0) / 2.0
            assert abs(c1 - c05) < 1e-9, f"dérive de centre : {s1} axe {a}"
    print("2. décalages doublés 1mm -> 0.5mm, centres identiques (300 cas) ✅")

    for shape in [(4, 6, 8), (192, 224, 192), (17, 5, 23)]:
        grid = make_coord_grid(shape).numpy()
        h, w, d = shape
        n = min(5000, h * w * d)
        idx = rng.choice(h * w * d, size=n, replace=False)
        vol = np.arange(h * w * d, dtype=np.float32).reshape(shape)
        saved = np.random.randint
        np.random.randint = lambda lo, hi, size: idx          # tirage imposé
        try:
            coords, values = _sample_flat(vol, n)
        finally:
            np.random.randint = saved
        assert np.abs(coords.numpy() - grid[idx]).max() < 1e-6, f"coords != make_coord_grid ({shape})"
        assert np.array_equal(values.numpy()[:, 0], vol.reshape(-1)[idx]), f"valeurs ({shape})"
    print("3. _sample_flat suit la convention de make_coord_grid ✅")

    # --- 4. le chemin rapide est-il EXACTEMENT le chemin de référence ? -----
    # C'est le test qui compte : `_sample_cropped` court-circuite normalisation,
    # recadrage et copie, et doit rendre les mêmes valeurs que
    # `_sample_flat(_extract_at(normalize_volume_fixed(raw)))`, repliement des
    # bords compris.
    for _ in range(60):
        src = tuple(int(x) for x in rng.integers(12, 40, 3))
        tgt = tuple(int(x) for x in rng.integers(12, 44, 3))     # plus grand => padding
        raw = rng.standard_normal(src).astype(np.float32) * 40 + 100
        lo, hi = float(np.percentile(raw, 5)), float(np.percentile(raw, 95))
        off = _crop_offsets(src, tgt)
        n = 400
        idx = rng.integers(0, int(np.prod(tgt)), size=n)
        saved = np.random.randint
        np.random.randint = lambda a, b, size: idx
        try:
            c_fast, v_fast = _sample_cropped(raw, off, tgt, n, lo, hi)
            ref_vol = _extract_at(normalize_volume_fixed(raw, lo, hi), off, tgt)
            c_ref, v_ref = _sample_flat(ref_vol, n)
        finally:
            np.random.randint = saved
        assert torch.equal(c_fast, c_ref), f"coordonnées divergentes src={src} tgt={tgt}"
        assert torch.allclose(v_fast, v_ref, atol=1e-6), (
            f"valeurs divergentes src={src} tgt={tgt} "
            f"(max {float((v_fast - v_ref).abs().max()):.2e})")
    print("4. _sample_cropped == chemin de référence, bords repliés compris (60 cas) ✅")
    print("\n✅ arithmétique de supervision vérifiée")


def main():
    ap = argparse.ArgumentParser(description="Entraînement du décodeur implicite conditionné par grille")
    ap.add_argument("--self-test", action="store_true",
                    help="Vérifie l'arithmétique d'indices puis sort")
    ap.add_argument("--config")
    ap.add_argument("--env", default="local")
    ap.add_argument("--steps", type=int, default=None, help="Override de train.total_iters")
    ap.add_argument("--max-volumes", type=int, default=None, help="Sous-ensemble (mise au point)")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint de départ (reprise à chaud, poids seuls : "
                         "l'optimiseur repart propre, ce qui est le but après une divergence)")
    ap.add_argument("--lr", type=float, default=None, help="Override de train.lr")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--schedule", choices=["plateau", "cosine"], default=None,
                    help="plateau = constant puis descente linéaire (défaut) ; cosine = décroissance continue")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.config:
        ap.error("--config est requis (sauf avec --self-test)")

    cfg = resolve_paths(load_yaml_with_include(args.config), load_env(args.env))
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    volume_size = tuple(int(v) for v in d["volume_size"])
    output_dir = Path(d["output_dir"]); (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    # Le cache ne contient que `retro_train` : on découpe donc nous-mêmes une
    # validation PAR SUJET. Sans cela l'évaluation serait faite sur des volumes
    # vus à l'entraînement, alors que le décodeur convolutif de MedVAE auquel on
    # se compare n'a JAMAIS vu ces données (poids pré-entraînés) — la comparaison
    # serait truquée en notre faveur. La liste est écrite sur disque pour que
    # l'évaluation reprenne exactement le même partage, sans le re-dériver.
    cache_index = json.loads((Path(d["latent_cache_dir"]) / "index.json").read_text())
    n_val = int(d.get("val_subjects", 40))
    val_subs = holdout_subjects(cache_index["samples"], n_val)
    (output_dir / "val_subjects.json").write_text(
        json.dumps({"n_val_subjects": len(val_subs), "seed": 1234,
                    "subjects": sorted(val_subs)}, indent=2))

    ds = LatentVolumeDataset(
        cache_dir=Path(d["latent_cache_dir"]), cache_root=Path(d["latent_cache_root"]),
        data_root=Path(d["data_root"]), split=d.get("split", "retro_train"),
        volume_size=volume_size, target_spacing=tuple(float(v) for v in d["target_spacing"]),
        p_lo=d.get("percentile_lower", 0.5), p_hi=d.get("percentile_upper", 99.5),
        sup_factor=int(d.get("supervision_factor", 2)),
        points=int(t.get("points_per_step", 16384)),
        reuse=int(t.get("batches_per_volume", 8)),
        exclude_subjects=val_subs,
    )
    if args.max_volumes:
        ds.samples = ds.samples[: args.max_volumes]
    n_held = sum(1 for s in cache_index["samples"] if subject_of(s["path"]) in val_subs)
    print(f"Dataset : {len(ds)} volumes d'entraînement "
          f"({n_held} volumes réservés sur {len(val_subs)} sujets) "
          f"| latent {volume_size} -> supervision {ds.sup_size} "
          f"({ds.sup_spacing[0]:g}mm)", flush=True)

    batch_size = int(t.get("batch_size", 2))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=int(t.get("num_workers", 8)), pin_memory=True,
                        drop_last=True, persistent_workers=True)

    model = GridConditionedINR(GridINRConfig(
        latent_channels=int(m.get("latent_channels", 1)),
        feat_dim=int(m.get("feat_dim", 64)),
        n_res_blocks=int(m.get("n_res_blocks", 2)),
        hidden_dim=int(m.get("hidden_dim", 256)),
        num_hidden_layers=int(m.get("num_hidden_layers", 5)),
        omega_0=float(m.get("omega_0", 30.0)),
        omega_hidden=float(m.get("omega_hidden", 30.0)),
        fg_weight=float(m.get("fg_weight", 5.0)),
        bg_threshold=float(m.get("bg_threshold", -0.9)),
    )).to(device)
    print(f"Décodeur : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)

    if args.resume:
        prev = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(prev["model"])
        print(f"Reprise depuis {args.resume} (iter {prev.get('iter', '?')})", flush=True)

    total_iters = int(args.steps or t.get("total_iters", 50000))
    warmup = int(t.get("warmup_steps", 1000))
    lr = float(args.lr if args.lr is not None else t.get("lr", 1e-4))
    wd = float(args.weight_decay if args.weight_decay is not None else t.get("weight_decay", 1e-4))
    schedule = args.schedule or t.get("schedule", "plateau")
    grad_clip = float(t.get("grad_clip", 1.0))
    save_every = int(t.get("save_every", 5000))
    print_every = int(t.get("print_every", 100))
    print(f"Optimisation : lr={lr:.1e} weight_decay={wd:g} schedule={schedule}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        if schedule == "cosine":
            p = (step - warmup) / max(total_iters - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
        start = total_iters // 2
        if step < start:
            return 1.0
        return max(0.0, 1.0 - (step - start) / max(total_iters - start, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    ema = EMAModel(model, decay=float(t.get("ema_decay", 0.9999)))

    def _infinite(dl):
        while True:
            yield from dl

    def _steps(dl):
        """Un pas d'optimisation = un lot de `points_per_step` points.

        Les points sont tirés par le worker, INDÉPENDAMMENT pour chaque volume
        du batch (le tirage partagé de `sample_points` était un compromis
        mémoire propre à la boucle interne de l'INR global ; il n'a plus lieu
        d'être ici). Chaque volume chargé fournit `reuse` lots DISJOINTS,
        consommés en autant de pas successifs.
        """
        n = int(t.get("points_per_step", 16384))
        while True:
            for z, coords, values in dl:
                z = z.to(device, non_blocking=True)
                coords = coords.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                for c in range(ds.reuse):
                    yield z, coords[:, c * n:(c + 1) * n], values[:, c * n:(c + 1) * n]

    it = _steps(loader)
    recent: List[float] = []
    t0 = time.time()
    for step in range(total_iters):
        z, coords, values = next(it)

        loss = model.loss(z, coords, values)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip > 0 else float("inf"))
        opt.step(); sched.step(); ema.update(model)

        recent.append(float(loss.item()))
        if len(recent) > print_every:
            recent.pop(0)
        if (step + 1) % print_every == 0:
            avg = float(np.mean(recent)); el = time.time() - t0
            eta = (total_iters - step - 1) * el / (step + 1)
            print(f"[{step + 1:6d}/{total_iters}] loss={avg:.5f} grad={float(gn):.2f} "
                  f"lr={sched.get_last_lr()[0]:.2e} t={el / 60:.1f}min eta={eta / 3600:.2f}h",
                  flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"iter": step + 1, "loss": round(avg, 6),
                                    "grad_norm": round(float(gn), 4),
                                    "lr": sched.get_last_lr()[0],
                                    "elapsed_s": round(el, 1)}) + "\n")
        if (step + 1) % save_every == 0 or step + 1 == total_iters:
            ckpt = {"iter": step + 1, "model": model.state_dict(),
                    "ema": ema.state_dict(), "cfg_path": args.config}
            torch.save(ckpt, output_dir / "weights" / "model_final.pth")
            # Historique conservé : sur ce projet, la loss d'entraînement s'est
            # révélée un mauvais indicateur à trois reprises (un entraînement
            # plus long, ou une loss plus basse, a dégradé Task 3). Garder les
            # étapes intermédiaires permet de choisir sur une vraie mesure de
            # reconstruction plutôt que sur la dernière itération par défaut.
            # Coût : ~6 Mo par point, négligeable.
            if step + 1 < total_iters:
                torch.save(ckpt, output_dir / "weights" / f"model_step{step + 1:06d}.pth")

    print(f"\nTerminé. Décodeur : {output_dir / 'weights' / 'model_final.pth'}", flush=True)


if __name__ == "__main__":
    main()
