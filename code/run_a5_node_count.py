# -*- coding: utf-8 -*-
"""A5: Node-count sensitivity (revision item #15, tier-1, zero re-run).

Question: are the regional (lobe-level) log-log exponents alpha confounded
by the number of network nodes (electrodes) in each lobe?

Design:
  1. Nominal channel counts per lobe from code/src/lobe_map.json (fixed
     montage mapping; 128 scalp channels, E129 reference excluded upstream).
  2. The 50 lobe fits from results/tables/tae_lobe_allometry.csv
     (5 lobes x 5 bands x 2 metrics). SEs are recovered exactly from the
     stored two-sided p-values: t = |alpha| / se with df = n - 2, so
     se = |alpha| / t.ppf(1 - p/2, n - 2).
  3. Meta-regression of alpha on channel count:
       (a) WLS with inverse-variance weights 1/se^2 (primary)
       (b) OLS with HC3 (robustness)
     Caveat noted in output: each lobe contributes 10 correlated fits
     (band x metric), so the effective information is coarser than 50
     independent estimates; the per-group rank correlations below
     quantify the within-group pattern.
  4. Spearman rank correlation between alpha and channel count:
       (a) pooled across all 50 fits
       (b) within each band x metric group (n = 5 lobes, descriptive)

Output: results/tables/node_count_sensitivity.csv  (50 rows, per-fit)
        results/tables/node_count_summary.csv      (meta-regression + rho)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, t as tdist

ROOT = Path(r"d:\TanHuangWork\HBN4")
TABLES = ROOT / "results" / "tables"


def main() -> int:
    # ---- 1. nominal channel counts per lobe ----
    with open(ROOT / "code" / "src" / "lobe_map.json", "r", encoding="utf-8") as f:
        lobe_map = json.load(f)
    counts = pd.Series(lobe_map).value_counts().sort_index()
    total = int(counts.sum())
    print(f"[map] channels per lobe (total={total}):")
    for lobe, k in counts.items():
        print(f"  {lobe:10s} {k}")

    # ---- 2. lobe fits + SE recovery ----
    df = pd.read_csv(TABLES / "tae_lobe_allometry.csv")
    df["n_ch"] = df["lobe"].map(counts)
    assert df["n_ch"].notna().all(), "lobe without channel count"
    df["df_resid"] = df["n"] - 2
    # floor p to avoid float underflow (p==0 -> tcrit=inf -> se=0 -> inf weights);
    # use isf on the tail probability directly (ppf(1-p/2) would round to ppf(1.0))
    p_raw = df["p"].to_numpy(float)
    n_floored = int((p_raw < 1e-300).sum())
    p_eff = np.clip(p_raw, 1e-300, None)
    tcrit = tdist.isf(p_eff / 2.0, df["df_resid"].to_numpy(float))
    df["se_rec"] = np.abs(df["alpha"].to_numpy(float)) / tcrit
    print(f"\n[fits] {len(df)} lobe fits, alpha in "
          f"[{df['alpha'].min():.3f}, {df['alpha'].max():.3f}], "
          f"se_rec in [{df['se_rec'].min():.4f}, {df['se_rec'].max():.4f}]"
          f" ({n_floored} p-values floored at 1e-300)")

    # ---- 3. meta-regressions ----
    X = sm.add_constant(df["n_ch"].to_numpy(float))
    y = df["alpha"].to_numpy(float)

    wls = sm.WLS(y, X, weights=1.0 / df["se_rec"].to_numpy(float) ** 2).fit()
    ols_hc3 = sm.OLS(y, X).fit(cov_type="HC3")

    def slope_row(name, res):
        b1, se1 = float(res.params[1]), float(res.bse[1])
        ci = res.conf_int()[1]
        return dict(analysis=name, slope_per_channel=b1, se=se1,
                    ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                    p=float(res.pvalues[1]), n_fits=len(df))

    summary = [slope_row("wls_invvar", wls), slope_row("ols_hc3", ols_hc3)]

    # ---- 4. rank correlations ----
    rho_all, p_all = spearmanr(df["n_ch"], df["alpha"])
    summary.append(dict(analysis="spearman_pooled", slope_per_channel=rho_all,
                        se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=float(p_all),
                        n_fits=len(df)))

    grp_rows = []
    for (band, metric), g in df.groupby(["band", "metric"], sort=True):
        rho, p = spearmanr(g["n_ch"], g["alpha"])
        grp_rows.append(dict(band=band, metric=metric, n_lobes=len(g),
                             spearman_rho=rho, p=p))
    grp = pd.DataFrame(grp_rows)
    n_neg = int((grp["spearman_rho"] < 0).sum())
    print(f"[rank] pooled Spearman rho={rho_all:.3f} (p={p_all:.3g}); "
          f"per-group rho<0 in {n_neg}/10 groups")

    # ---- 5. save ----
    out_cols = ["band", "lobe", "metric", "n", "n_ch", "alpha", "se_rec", "r2", "p"]
    df[out_cols].to_csv(TABLES / "node_count_sensitivity.csv",
                        index=False, float_format="%.6g")
    print(f"[save] node_count_sensitivity.csv ({len(df)} rows)")
    pd.DataFrame(summary).to_csv(TABLES / "node_count_summary.csv",
                                 index=False, float_format="%.6g")
    grp.to_csv(TABLES / "node_count_spearman_by_group.csv",
               index=False, float_format="%.6g")
    print("[save] node_count_summary.csv (3 rows) + node_count_spearman_by_group.csv (10 rows)")

    print("\n[meta-regression] alpha ~ n_channels")
    print(f"  WLS inv-var : slope={wls.params[1]:+.5f} per channel, "
          f"p={wls.pvalues[1]:.3g}")
    print(f"  OLS HC3     : slope={ols_hc3.params[1]:+.5f} per channel, "
          f"p={ols_hc3.pvalues[1]:.3g}")
    print("\n[per-group Spearman (5 lobes each)]")
    print(grp.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
