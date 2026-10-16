# -*- coding: utf-8 -*-
"""A2: Release robustness of developmental TAE scaling (revision item #3).

Part 1 (release-adjusted): for all 20 band x metric fits, refit
    log(y) ~ log(age) + C(sex) + C(release)
and report the log-age slope (alpha_adj, HC3 robust SE) and its difference
from the primary alpha (tae_summary_full.csv).

Part 2 (leave-one-release-out): refit the primary model
    log(y) ~ log(age)
dropping one of the 10 releases at a time, for all 20 fits. Summary
statistics (mean / range / 95% t-CI over the 10 omissions) are reported for
every fit; the 4 core fits (delta/theta x global_efficiency/small_worldness)
are highlighted in the console summary for the main text.

Outputs:
  results/tables/release_robustness.csv   (20 adjusted rows + 20 LORO summary rows)
  results/tables/release_loro_detail.csv  (20 fits x 10 omissions = 200 rows)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

ROOT = Path(r"d:\TanHuangWork\HBN4")
TABLES = ROOT / "results" / "tables"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
METRICS = ["global_efficiency", "local_efficiency", "modularity", "small_worldness"]
CORE = {(b, m) for b in ("delta", "theta")
        for m in ("global_efficiency", "small_worldness")}

FORMULA_ADJ = "y_log ~ x_logage + C(sex) + C(release)"
FORMULA_MAIN = "y_log ~ x_logage"


def fit_alpha(df: pd.DataFrame, formula: str, cov_type: str | None):
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    res = sm.OLS(y.iloc[:, 0], X).fit(cov_type=cov_type) if cov_type else sm.OLS(y.iloc[:, 0], X).fit()
    i = list(res.params.index).index("x_logage")
    alpha = float(res.params.iloc[i])
    se = float(res.bse.iloc[i])
    pval = float(res.pvalues.iloc[i])
    return alpha, se, pval, int(res.nobs)


def main() -> int:
    feat = pd.read_csv(TABLES / "subjects_features_with_release.csv")
    ref = pd.read_csv(TABLES / "tae_summary_full.csv").set_index(["band", "metric"])
    releases = sorted(feat["release"].unique())
    print(f"[load] {len(feat)} subjects, {len(releases)} releases: "
          + ", ".join(f"{r.split('-')[1]}={int((feat['release'] == r).sum())}" for r in releases))

    adj_rows, loro_rows, loro_detail = [], [], []
    t95_9 = 2.2621571628  # t(0.975, df=9)

    for band in BANDS:
        for metric in METRICS:
            col = f"{band}__{metric}"
            sub = feat[["age", "sex", "release", col]].dropna().copy()
            sub = sub[sub[col] > 0]
            sub["y_log"] = np.log(sub[col].to_numpy(float))
            sub["x_logage"] = np.log(sub["age"].to_numpy(float))

            # ---- Part 1: release-adjusted ----
            a_adj, se_adj, p_adj, n = fit_alpha(sub, FORMULA_ADJ, "HC3")
            a_pri = float(ref.loc[(band, metric), "alpha"])
            adj_rows.append(dict(
                band=band, metric=metric, model="adjusted", n=n,
                alpha_adj=a_adj, alpha_se_hc3=se_adj, p_adj=p_adj,
                alpha_primary=a_pri, delta_alpha=a_adj - a_pri,
            ))

            # ---- Part 2: leave-one-release-out ----
            alphas = []
            for rel in releases:
                drop_df = sub[sub["release"] != rel]
                a_r, _, _, n_r = fit_alpha(drop_df, FORMULA_MAIN, None)
                alphas.append(a_r)
                loro_detail.append(dict(
                    band=band, metric=metric, omitted_release=rel,
                    n=n_r, alpha_loro=a_r,
                ))
            arr = np.asarray(alphas)
            loro_rows.append(dict(
                band=band, metric=metric, model="loro", n=int(len(sub)),
                alpha_mean=float(arr.mean()), alpha_min=float(arr.min()),
                alpha_max=float(arr.max()),
                alpha_ci_lo=float(arr.mean() - t95_9 * arr.std(ddof=1) / np.sqrt(len(arr))),
                alpha_ci_hi=float(arr.mean() + t95_9 * arr.std(ddof=1) / np.sqrt(len(arr))),
                alpha_primary=a_pri,
                is_core=int((band, metric) in CORE),
            ))
            print(f"[fit] {band}/{metric}: adj alpha={a_adj:+.4f} (d={a_adj - a_pri:+.4f}), "
                  f"loro range=[{arr.min():+.4f}, {arr.max():+.4f}]")

    adj_df = pd.DataFrame(adj_rows)
    loro_df = pd.DataFrame(loro_rows)
    out = pd.concat([adj_df, loro_df], ignore_index=True)
    out.to_csv(TABLES / "release_robustness.csv", index=False, float_format="%.6g")
    pd.DataFrame(loro_detail).to_csv(TABLES / "release_loro_detail.csv", index=False, float_format="%.6g")
    print(f"[save] release_robustness.csv ({len(out)} rows), release_loro_detail.csv ({len(loro_detail)} rows)")

    print("\n[core fits, LORO] (delta/theta x E_glob/sigma)")
    core_df = loro_df[loro_df["is_core"] == 1]
    print(core_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    dmax = float(adj_df["delta_alpha"].abs().max())
    rmax = float((loro_df["alpha_max"] - loro_df["alpha_min"]).max())
    print(f"\n[max] |delta_alpha| (adjusted vs primary) = {dmax:.4f}")
    print(f"[max] LORO range width across 20 fits     = {rmax:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
