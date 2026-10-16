# -*- coding: utf-8 -*-
"""A4: Age-distribution sensitivity of developmental TAE scaling (revision item #4).

HBN ages are unevenly distributed (5-8: 904 / 9-12: 720 / 13-16: 284 / 17-21: 59).
For the 4 core fits (delta/theta x global_efficiency/small_worldness), compare:

  primary   : unweighted OLS log(y) ~ log(age)           (reference)
  weighted  : inverse-density weighting, w_i = 1/p_hat(age_i),
              p_hat from a Gaussian KDE (Scott rule); HC3 covariance
  bootstrap : age-bin stratified bootstrap (4 bins, B=1000, seed=42),
              resample within each bin with replacement, refit primary model,
              report mean and 2.5/97.5 percentile CI

Method D (log-log natural spline) is covered by the `gam` rows of
model_comparison.csv (A1) and is not repeated here.

Output:
  results/tables/age_balance_robustness.csv
  (4 fits x 3 analyses: primary / weighted / stratified bootstrap)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import gaussian_kde

ROOT = Path(r"d:\TanHuangWork\HBN4")
TABLES = ROOT / "results" / "tables"

BANDS = ["delta", "theta"]
METRICS = ["global_efficiency", "small_worldness"]

B_EDGES = [0.0, 9.0, 13.0, 17.0, np.inf]   # 5-8 / 9-12 / 13-16 / 17-21
B_LABELS = ["5-8", "9-12", "13-16", "17-21"]
N_BOOT = 1000
SEED = 42


def ols_alpha(y: np.ndarray, x: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def main() -> int:
    feat = pd.read_csv(TABLES / "subjects_features_with_release.csv")
    ref = pd.read_csv(TABLES / "tae_summary_full.csv").set_index(["band", "metric"])
    rng = np.random.default_rng(SEED)

    rows = []
    for band in BANDS:
        for metric in METRICS:
            col = f"{band}__{metric}"
            sub = feat[["age", col]].dropna().copy()
            sub = sub[sub[col] > 0]
            age = sub["age"].to_numpy(float)
            y_log = np.log(sub[col].to_numpy(float))
            x_logage = np.log(age)

            # ---- primary ----
            a_pri, _, _, n = (None, None, None, len(sub))
            a_pri = float(ref.loc[(band, metric), "alpha"])
            se_pri = float(ref.loc[(band, metric), "slope_se"])
            rows.append(dict(
                band=band, metric=metric, analysis="primary", n=n,
                alpha=a_pri, ci_lo=a_pri - 1.96 * se_pri, ci_hi=a_pri + 1.96 * se_pri,
                note="unweighted OLS, 1.96*SE CI",
            ))

            # ---- inverse-density weighting (KDE) ----
            kde = gaussian_kde(age)
            w = 1.0 / kde(age)
            w = w / w.mean()
            X = patsy.dmatrix("x", pd.DataFrame({"x": x_logage}))
            res_w = sm.WLS(y_log, X, weights=w).fit(cov_type="HC3")
            a_w = float(res_w.params[1])
            se_w = float(res_w.bse[1])
            eff_n = float(w.sum() ** 2 / (w ** 2).sum())
            rows.append(dict(
                band=band, metric=metric, analysis="weighted", n=n,
                alpha=a_w, ci_lo=a_w - 1.96 * se_w, ci_hi=a_w + 1.96 * se_w,
                note=f"inverse-KDE-density weights (HC3), effective n={eff_n:.0f}",
            ))

            # ---- stratified bootstrap ----
            bins = pd.cut(age, bins=B_EDGES, labels=B_LABELS, right=False)
            bin_levels = [np.where(bins == lab)[0] for lab in B_LABELS]
            counts = [len(ix) for ix in bin_levels]
            alphas = np.empty(N_BOOT)
            for b in range(N_BOOT):
                pick = np.concatenate([
                    rng.choice(ix, size=len(ix), replace=True) for ix in bin_levels
                ])
                alphas[b] = ols_alpha(y_log[pick], x_logage[pick])
            rows.append(dict(
                band=band, metric=metric, analysis="bootstrap", n=n,
                alpha=float(alphas.mean()),
                ci_lo=float(np.percentile(alphas, 2.5)),
                ci_hi=float(np.percentile(alphas, 97.5)),
                note=f"age-bin stratified bootstrap B={N_BOOT} (bins n={counts})",
            ))
            print(f"[fit] {band}/{metric}: primary={a_pri:+.4f}, weighted={a_w:+.4f}, "
                  f"boot={alphas.mean():+.4f} [{np.percentile(alphas, 2.5):+.4f}, "
                  f"{np.percentile(alphas, 97.5):+.4f}]")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "age_balance_robustness.csv", index=False, float_format="%.6g")
    print(f"[save] age_balance_robustness.csv ({len(out)} rows)")
    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
