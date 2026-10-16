# -*- coding: utf-8 -*-
"""A3: Formal lobe x age interaction tests (revision item #2).

Data: lobe checkpoint (2,475 candidates) restricted to the 1,967-subject
primary analysis set. For each band x metric group (5 bands x 2 metrics:
global_efficiency, small_worldness), subjects must have non-missing,
positive values in ALL five lobes (complete-case) so that slope contrasts
are not confounded by differing sample composition across lobes.

Models (long format, no-intercept parameterisation):
  full    : log(y) ~ 0 + C(lobe) + C(lobe):log(age)      (5 lobe slopes)
  reduced : log(y) ~ 0 + C(lobe) + log(age)              (common slope)

Reported:
  1. omnibus  : F test full vs reduced (anova_lm), BH-FDR over the 10 groups
  2. slope    : per-lobe log-log slope, HC3 robust SE
  3. pairwise : all 10 lobe-pair slope contrasts per group, Wald z with HC3
                covariance, BH-FDR over the 100 contrasts (single family,
                consistent with the paper's 50-fit lobe FDR family)

Key contrast for the revision: temporal vs frontal (gamma-band sigma and
global efficiency).

Output: results/tables/lobe_interaction.csv (50 slope + 100 pairwise rows);
        results/tables/lobe_interaction_omnibus.csv (10 rows)
"""
from __future__ import annotations

import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"d:\TanHuangWork\HBN4")
TABLES = ROOT / "results" / "tables"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
METRICS = ["global_efficiency", "small_worldness"]
LOBES = ["central", "frontal", "occipital", "parietal", "temporal"]
KEY_CONTRASTS = {("gamma", "global_efficiency", "temporal", "frontal"),
                 ("gamma", "small_worldness", "temporal", "frontal")}

FORM_FULL = "y_log ~ 0 + C(lobe) + C(lobe):x_logage"
FORM_RED = "y_log ~ 0 + C(lobe) + x_logage"


def main() -> int:
    with open(TABLES / "lobe_features_checkpoint.pkl", "rb") as f:
        ckpt = pickle.load(f)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("results"), dict):
        ckpt = ckpt["results"]

    feat = pd.read_csv(TABLES / "subjects_features_with_release.csv",
                       usecols=["participant_id", "age"])
    primary_ids = set(feat["participant_id"])

    # ---- tidy long frame ----
    rec = []
    for pid, d in ckpt.items():
        if pid not in primary_ids:
            continue
        for (band, lobe, metric), v in d.items():
            rec.append((pid, band, metric, lobe, v))
    long_df = pd.DataFrame(rec, columns=["participant_id", "band", "metric", "lobe", "value"])
    long_df = long_df.merge(feat, on="participant_id", how="inner")
    print(f"[load] checkpoint rows for primary set: {len(long_df)} "
          f"({long_df['participant_id'].nunique()} subjects)")
    n_nonpos = int((long_df["value"] <= 0).sum())
    print(f"[note] non-positive lobe values (sigma=0 etc.): {n_nonpos} -> excluded per complete-case rule")

    rows = []
    omnibus_ps, pairwise_ps = [], []

    for band in BANDS:
        for metric in METRICS:
            g = long_df[(long_df["band"] == band) & (long_df["metric"] == metric)
                        & (long_df["value"] > 0)]
            # complete-case: 5 lobes present
            cnt = g.groupby("participant_id")["lobe"].nunique()
            keep_ids = cnt[cnt == len(LOBES)].index
            g = g[g["participant_id"].isin(keep_ids)].copy()
            n = int(g["participant_id"].nunique())
            g["y_log"] = np.log(g["value"].to_numpy(float))
            g["x_logage"] = np.log(g["age"].to_numpy(float))

            y_full, X_full = patsy.dmatrices(FORM_FULL, g, return_type="dataframe")
            _, X_red = patsy.dmatrices(FORM_RED, g, return_type="dataframe")
            res_full = sm.OLS(y_full.iloc[:, 0], X_full).fit()
            res_red = sm.OLS(y_full.iloc[:, 0], X_red).fit()
            anova = anova_lm(res_red, res_full)
            F = float(anova.loc[1, "F"])
            p_omni = float(anova.loc[1, "Pr(>F)"])
            omnibus_ps.append(p_omni)

            # HC3 refit for slope SEs and pairwise contrasts
            res_hc3 = sm.OLS(y_full.iloc[:, 0], X_full).fit(cov_type="HC3")
            cov = res_hc3.cov_params()
            alphas, ses = {}, {}
            for lobe in LOBES:
                key = f"C(lobe)[{lobe}]:x_logage"
                alphas[lobe] = float(res_hc3.params[key])
                ses[lobe] = float(np.sqrt(cov.loc[key, key]))

            for lobe in LOBES:
                rows.append(dict(band=band, metric=metric, analysis="slope",
                                 comparison=lobe, n=n, F=np.nan, p=np.nan,
                                 diff=np.nan, se=np.nan, z=np.nan,
                                 p_raw=np.nan, p_fdr=np.nan,
                                 alpha=alphas[lobe], alpha_se_hc3=ses[lobe]))
                pairwise_ps.append(np.nan)  # placeholder to keep index alignment

            pairs = list(combinations(LOBES, 2))
            for l1, l2 in pairs:
                k1 = f"C(lobe)[{l1}]:x_logage"
                k2 = f"C(lobe)[{l2}]:x_logage"
                diff = alphas[l1] - alphas[l2]
                var = float(cov.loc[k1, k1] + cov.loc[k2, k2] - 2 * cov.loc[k1, k2])
                se = float(np.sqrt(var))
                z = diff / se
                p = float(2 * norm.sf(abs(z)))
                rows.append(dict(band=band, metric=metric, analysis="pairwise",
                                 comparison=f"{l1} vs {l2}", n=n, F=np.nan, p=np.nan,
                                 diff=diff, se=se, z=z, p_raw=p, p_fdr=np.nan,
                                 alpha=np.nan, alpha_se_hc3=np.nan))
                pairwise_ps.append(p)

            print(f"[fit] {band}/{metric}: n={n}, omnibus F={F:.3f}, p={p_omni:.3g}")

    out = pd.DataFrame(rows)

    # FDR corrections
    out_pairs = out[out["analysis"] == "pairwise"].copy()
    fdr_pairs = multipletests(out_pairs["p_raw"].to_numpy(float), method="fdr_bh")[1]
    out.loc[out_pairs.index, "p_fdr"] = fdr_pairs

    out.to_csv(TABLES / "lobe_interaction.csv", index=False, float_format="%.6g")
    print(f"[save] lobe_interaction.csv ({len(out)} rows)")

    # omnibus FDR needs the 10 p-values in group order: recompute here
    # (kept simple: re-run the omnibus p collection from the printed order)
    # -- collect again from rows is impossible (omnibus rows not stored), so
    #    store them in a side table:
    # (omnibus_ps is in band-major group order)
    omni_fdr = multipletests(omnibus_ps, method="fdr_bh")[1]
    omni_out = []
    i = 0
    for band in BANDS:
        for metric in METRICS:
            omni_out.append(dict(band=band, metric=metric, analysis="omnibus",
                                 comparison="lobe x log(age)", n=np.nan,
                                 F=np.nan, p=omnibus_ps[i], p_fdr=omni_fdr[i]))
            i += 1
    pd.DataFrame(omni_out).to_csv(TABLES / "lobe_interaction_omnibus.csv",
                                  index=False, float_format="%.6g")
    print("[save] lobe_interaction_omnibus.csv (10 rows)")

    print("\n[omnibus, FDR-corrected]")
    for o, q in zip(omni_out, omni_fdr):
        print(f"  {o['band']:5s}/{o['metric']:17s} p={o['p']:.3g}  p_fdr={q:.3g}")

    key = out[(out["analysis"] == "pairwise")
              & out.apply(lambda r: (r["band"], r["metric"], *r["comparison"].split(" vs ")) in KEY_CONTRASTS
                          or (r["band"], r["metric"], *reversed(r["comparison"].split(" vs "))) in KEY_CONTRASTS,
                          axis=1)]
    print("\n[key contrasts: temporal vs frontal, gamma band]")
    print(key.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    sig = out[(out["analysis"] == "pairwise") & (out["p_fdr"] < 0.05)]
    print(f"\n[summary] pairwise contrasts significant at FDR q<0.05: {len(sig)}/100")
    if len(sig):
        print(sig[["band", "metric", "comparison", "diff", "p_fdr"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
