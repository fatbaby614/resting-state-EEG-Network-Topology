# -*- coding: utf-8 -*-
"""Write results/tables/iad_primary.csv for the ROC figure.

Reviewer comment 8 asked that the ROC panel be drawn on the PRIMARY
(degenerate-network-excluded) sample, matching the caption. The per-subject
IAD scores for that sample were computed inside run_revision_0922.py but never
saved, so this script reproduces exactly the same call sequence
(run_revision_0922.py lines 322-326: allometry.compute_iad over the five bands
for global efficiency) and stores the resulting frame.

The script then recomputes the band-wise AUC with the identical rule used by
clinical.roc_analysis (externalizing > 1.0) and compares it against the
published roc_primary.csv. Any mismatch aborts the run, which guarantees that
the figure and Table~S6 quote the same numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(r"d:\TanHuangWork\HBN4")
TAB = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT / "code"))
from src import allometry as allomod  # noqa: E402

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CUTOFF = 1.0


def main():
    src = pd.read_csv(TAB / "subjects_features_primary.csv")
    print(f"[load] subjects_features_primary.csv: {src.shape}")
    iad = src.copy()
    for band in BANDS:
        iad = allomod.compute_iad(iad, band, "global_efficiency")
    iad_cols = [f"{b}__global_efficiency__iad" for b in BANDS]
    missing = [c for c in iad_cols if c not in iad.columns]
    if missing:
        raise SystemExit(f"[abort] missing IAD columns: {missing}")
    iad.to_csv(TAB / "iad_primary.csv", index=False, float_format="%.6g")
    print(f"[write] iad_primary.csv: {iad.shape}")

    ref = pd.read_csv(TAB / "roc_primary.csv")
    ref = ref.assign(band=ref["col"].str.split("__").str[0])
    bad = []
    for band in BANDS:
        d = iad[[f"{band}__global_efficiency__iad",
                 "externalizing"]].replace([np.inf, -np.inf], np.nan).dropna()
        y = (d["externalizing"] > CUTOFF).astype(int)
        auc = float(roc_auc_score(y, d[f"{band}__global_efficiency__iad"]))
        want = float(ref.loc[ref["band"] == band, "auc"].iloc[0])
        n_risk = int(y.sum())
        n_ref = int(ref.loc[ref["band"] == band, "n_high"].iloc[0])
        delta = abs(auc - want)
        flag = "OK " if delta < 5e-4 else "MISMATCH"
        if delta >= 5e-4 or n_risk != n_ref:
            bad.append(band)
        print(f"[check] {band:5s} fig AUC={auc:.4f} table AUC={want:.4f} "
              f"|d|={delta:.2e} n_risk={n_risk}/{n_ref} {flag}")
    if bad:
        raise SystemExit(f"[abort] AUC mismatch vs roc_primary.csv in {bad}")
    print("IAD PRIMARY OK: figure inputs agree with roc_primary.csv")


if __name__ == "__main__":
    main()
