#!/usr/bin/env python3
"""
Statistiques complètes du jeu de données MRIxFields 2026.

Usage :
    python src/dataset_stats.py [--data-root /path/to/MRIxFields_20260414]
"""

import argparse
import re
from pathlib import Path
import pandas as pd

DATA_ROOT_DEFAULT = "/home/rousseau/Data/MRIxFields_20260414"
SPLITS = ["Training_retrospective", "Training_prospective"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "stats"


def collect(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for modality in MODALITIES:
            for field in FIELDS:
                d = data_root / split / modality / field
                if not d.exists():
                    continue
                for fname in d.iterdir():
                    m = FILE_RE.match(fname.name)
                    if m:
                        rows.append({
                            "split": split.replace("Training_", ""),
                            "modality": m.group(1),
                            "field": m.group(2),
                            "subject_id": int(m.group(3)),
                        })
    return pd.DataFrame(rows)


def sep(char="─", n=72): print(char * n)
def section(t): print(); sep("═"); print(f"  {t}"); sep("═")
def subsection(t): print(); sep(); print(f"  {t}"); sep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scan de : {data_root}")
    df = collect(data_root)
    print(f"  → {len(df)} fichiers trouvés\n")

    # ── 1. Volumes par split × modalité × champ ──────────────────────────────
    section("1. Nombre de volumes par split × modalité × champ")
    vol_table = (
        df.groupby(["split", "modality", "field"])
        .size().unstack("field", fill_value=0)
        .reindex(columns=FIELDS, fill_value=0)
    )
    vol_table["TOTAL"] = vol_table.sum(axis=1)
    print(vol_table.to_string())
    vol_table.to_csv(RESULTS_DIR / "volumes_per_split_modality_field.csv")

    # ── 2. Totaux ─────────────────────────────────────────────────────────────
    section("2. Totaux")
    print(f"  Volumes NIfTI total : {len(df)}")
    for split in ["retrospective", "prospective"]:
        sub = df[df["split"] == split]
        print(f"\n  [{split}]")
        print(f"    Volumes           : {len(sub)}")
        print(f"    IDs sujets uniques: {sub['subject_id'].nunique()}")
        for modality in MODALITIES:
            n = sub[sub["modality"] == modality]["subject_id"].nunique()
            print(f"    Sujets {modality:10s} : {n}")

    # ── 3. Prospectif : appariement ───────────────────────────────────────────
    section("3. Training_prospective — couverture champs × modalités")
    pro = df[df["split"] == "prospective"]
    sub_combo = pro.groupby("subject_id").apply(
        lambda g: {"modalities": set(g["modality"]), "fields": set(g["field"])}
    )
    print(f"  Sujets uniques                         : {pro['subject_id'].nunique()}")
    print(f"  Sujets avec 3 modalités (any field)    : {sum(1 for v in sub_combo if len(v['modalities'])==3)}")
    print(f"  Sujets avec 5 champs (any modality)    : {sum(1 for v in sub_combo if len(v['fields'])==5)}")
    print(f"  Sujets avec 3 modalités ET 5 champs    : {sum(1 for v in sub_combo if len(v['modalities'])==3 and len(v['fields'])==5)}")

    subsection("Distribution du nombre de champs par sujet (prospectif)")
    rows_pro = []
    for modality in MODALITIES:
        sub_df = pro[pro["modality"] == modality]
        sub_fields = sub_df.groupby("subject_id")["field"].apply(set)
        print(f"\n  {modality}  ({len(sub_fields)} sujets) :")
        for n_f in range(1, 6):
            subset = sub_fields[sub_fields.apply(len) == n_f]
            if len(subset) == 0:
                continue
            all_f = [f for fset in subset for f in fset]
            freq_str = "  ".join(f"{f}:{all_f.count(f)}" for f in FIELDS if all_f.count(f) > 0)
            print(f"    {n_f} champ(s) : {len(subset):4d} sujets  [{freq_str}]")
            rows_pro.append({"modality": modality, "n_fields": n_f, "n_subjects": len(subset)})
    pd.DataFrame(rows_pro).to_csv(RESULTS_DIR / "prospective_field_coverage.csv", index=False)

    pivot = (
        pro.assign(present=1)
        .pivot_table(index="subject_id", columns=["modality", "field"],
                     values="present", fill_value=0, aggfunc="max")
    )
    pivot.to_csv(RESULTS_DIR / "prospective_subject_matrix.csv")
    print(f"\n  Matrice exportée : prospective_subject_matrix.csv ({pivot.shape[0]}×{pivot.shape[1]})")

    # ── 4. Rétrospectif : chevauchement IDs entre modalités ───────────────────
    section("4. Training_retrospective — chevauchement d'IDs entre modalités (par champ)")
    retro = df[df["split"] == "retrospective"]
    print(f"  {'Champ':>6s}  {'T1W':>6s}  {'T2W':>6s}  {'T2FLAIR':>8s}  "
          f"{'T1W∩T2W':>8s}  {'T1W∩T2FL':>9s}  {'T2W∩T2FL':>9s}  {'∩3mod':>6s}  {'∪total':>7s}")
    sep()
    rows_ov = []
    for field in FIELDS:
        fd = retro[retro["field"] == field]
        s = {m: set(fd[fd["modality"] == m]["subject_id"]) for m in MODALITIES}
        s1, s2, s3 = s["T1W"], s["T2W"], s["T2FLAIR"]
        row = {"field": field, "T1W": len(s1), "T2W": len(s2), "T2FLAIR": len(s3),
               "T1W_T2W": len(s1&s2), "T1W_T2FLAIR": len(s1&s3), "T2W_T2FLAIR": len(s2&s3),
               "inter_all3": len(s1&s2&s3), "union": len(s1|s2|s3)}
        print(f"  {field:>6s}  {row['T1W']:>6d}  {row['T2W']:>6d}  {row['T2FLAIR']:>8d}  "
              f"{row['T1W_T2W']:>8d}  {row['T1W_T2FLAIR']:>9d}  {row['T2W_T2FLAIR']:>9d}  "
              f"{row['inter_all3']:>6d}  {row['union']:>7d}")
        rows_ov.append(row)
    pd.DataFrame(rows_ov).to_csv(RESULTS_DIR / "retrospective_modality_overlap.csv", index=False)

    # ── 5. Rétrospectif : sujets avec N modalités ─────────────────────────────
    section("5. Training_retrospective — sujets avec N modalités (par champ)")
    rows_multi = []
    for field in FIELDS:
        fd = retro[retro["field"] == field]
        sub_mods = fd.groupby("subject_id")["modality"].apply(set)
        print(f"\n  {field}  ({len(sub_mods)} sujets uniques) :")
        for n_mod in [1, 2, 3]:
            count = (sub_mods.apply(len) == n_mod).sum()
            combos = sub_mods[sub_mods.apply(len) == n_mod].apply(frozenset).value_counts()
            combo_str = "  ".join(f"({'+'.join(sorted(c))}×{v})" for c, v in combos.items())
            label = "3 mod (complet)" if n_mod == 3 else f"{n_mod} modalité(s)"
            print(f"    {label:20s} : {count:4d}  {combo_str}")
            rows_multi.append({"field": field, "n_modalities": n_mod, "n_subjects": count})
    pd.DataFrame(rows_multi).to_csv(RESULTS_DIR / "retrospective_modality_coverage.csv", index=False)

    # ── 6. Plages d'IDs ───────────────────────────────────────────────────────
    section("6. Plages d'IDs par champ (Training_retrospective)")
    print(f"  {'Champ':>6s}  {'ID min':>8s}  {'ID max':>8s}  {'N sujets':>9s}")
    sep()
    for field in FIELDS:
        fd = retro[retro["field"] == field]
        if fd.empty: continue
        print(f"  {field:>6s}  {fd['subject_id'].min():>8d}  {fd['subject_id'].max():>8d}  {fd['subject_id'].nunique():>9d}")

    print(); sep("═"); print(f"  CSV → {RESULTS_DIR}"); sep("═"); print()


if __name__ == "__main__":
    main()