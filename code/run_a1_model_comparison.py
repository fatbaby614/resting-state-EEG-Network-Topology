# -*- coding: utf-8 -*-
"""A1: Developmental model comparison for TAE scaling (revision item #4).

For each of the 20 band x metric fits (5 bands x 4 metrics), compare five
candidate models of the age trajectory:

  raw space    : linear        y          ~ age
                 quadratic     y          ~ age + age^2
  log-log space: power_law     log(y)     ~ log(age)
                 quadratic_log log(y)     ~ log(age) + log(age)^2
                 gam           log(y)     ~ cr(log(age), df=4)   (natural cubic spline)

Reported per model (revision item #4 spec):
  - R2, adjusted R2, AIC, BIC (comparable ONLY within the same DV space;
    raw-space and log-space models are compared across spaces via CV)
  - 5-fold CV RMSE and out-of-fold (OOF) R2 computed in BOTH spaces:
      * raw space : log-space models are exp()-back-transformed before scoring
      * log space : raw-space models are log()-transformed before scoring
                    (predictions <= 0 counted in neg_raw_preds; if any occur,
                     the log-space CV columns are left NaN for that model)

Output: results/tables/model_comparison.csv (20 fits x 5 models = 100 rows)

Sanity check: the power_law rows must reproduce tae_summary_full.csv
(alpha, r2) to numerical precision, since both are OLS fits of log(y) on
log(age) over the same n=1967 sample.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.model_selection import KFold

ROOT = Path(r"d:\TanHuangWork\HBN4")
TABLES = ROOT / "results" / "tables"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
METRICS = ["global_efficiency", "local_efficiency", "modularity", "small_worldness"]

MODELS = {
    # name: (dv_space, formula)
    "linear":        ("raw", "y_raw ~ x_age"),
    "quadratic":     ("raw", "y_raw ~ x_age + I(x_age ** 2)"),
    "power_law":     ("log", "y_log ~ x_logage"),
    "quadratic_log": ("log", "y_log ~ x_logage + I(x_logage ** 2)"),
    "gam":           ("log", "y_log ~ cr(x_logage, df=4)"),
}

KFOLD_SEED = 42
N_SPLITS = 5


def build_design(df: pd.DataFrame, formula: str):
    """Build patsy design (y, X, design_info) from the prepared frame."""
    y_mat, x_mat = patsy.dmatrices(formula, df, return_type="dataframe")
    return y_mat.iloc[:, 0], x_mat, x_mat.design_info


def fit_ols(df: pd.DataFrame, formula: str):
    y, X, _ = build_design(df, formula)
    return sm.OLS(y, X).fit()


def oof_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    """Out-of-fold R2 = 1 - SSE / SST with SST around the full-sample mean."""
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def run_one_fit(feat: pd.DataFrame, band: str, metric: str) -> list[dict]:
    col = f"{band}__{metric}"
    sub = feat[["age", col]].dropna().copy()
    n_drop_neg = int((sub[col] <= 0).sum())
    if n_drop_neg:
        print(f"  [warn] {col}: {n_drop_neg} non-positive values dropped")
    sub = sub[sub[col] > 0]
    sub["y_raw"] = sub[col].to_numpy(float)
    sub["y_log"] = np.log(sub["y_raw"].to_numpy(float))
    sub["x_age"] = sub["age"].to_numpy(float)
    sub["x_logage"] = np.log(sub["x_age"].to_numpy(float))

    n = len(sub)
    rows: list[dict] = []

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=KFOLD_SEED)
    folds = list(kfold.split(sub))

    for name, (space, formula) in MODELS.items():
        res = fit_ols(sub, formula)
        alpha = float(res.params.get("x_logage", np.nan)) if name == "power_law" else np.nan

        oof_raw = np.full(n, np.nan)
        oof_log = np.full(n, np.nan)
        neg_raw = 0
        for tr, te in folds:
            train_df = sub.iloc[tr]
            test_df = sub.iloc[te]
            y_tr, X_tr, dinfo = build_design(train_df, formula)
            res_f = sm.OLS(y_tr, X_tr).fit()
            X_te = patsy.build_design_matrices([dinfo], test_df)[0]
            pred = np.asarray(res_f.predict(X_te), dtype=float)

            if space == "raw":
                oof_raw[te] = pred
                with np.errstate(invalid="ignore", divide="ignore"):
                    plog = np.where(pred > 0, np.log(np.where(pred > 0, pred, 1.0)), np.nan)
                neg_raw += int((pred <= 0).sum())
                oof_log[te] = plog
            else:
                oof_log[te] = pred
                oof_raw[te] = np.exp(pred)

        y_raw = sub["y_raw"].to_numpy(float)
        y_log = sub["y_log"].to_numpy(float)

        valid_raw = np.isfinite(oof_raw)
        valid_log = np.isfinite(oof_log)
        cv_rmse_raw = float(np.sqrt(np.mean((y_raw[valid_raw] - oof_raw[valid_raw]) ** 2))) if valid_raw.all() else np.nan
        cv_rmse_log = float(np.sqrt(np.mean((y_log[valid_log] - oof_log[valid_log]) ** 2))) if valid_log.all() else np.nan
        oof_r2_raw = oof_r2(y_raw[valid_raw], oof_raw[valid_raw]) if valid_raw.all() else np.nan
        oof_r2_log = oof_r2(y_log[valid_log], oof_log[valid_log]) if valid_log.all() else np.nan

        rows.append(
            dict(
                band=band,
                metric=metric,
                model=name,
                dv_space=space,
                n=n,
                n_params=int(res.df_model) + 1,
                r2=float(res.rsquared),
                adj_r2=float(res.rsquared_adj),
                aic=float(res.aic),
                bic=float(res.bic),
                alpha_loglog=alpha,
                cv_rmse_raw=cv_rmse_raw,
                oof_r2_raw=oof_r2_raw,
                cv_rmse_log=cv_rmse_log,
                oof_r2_log=oof_r2_log,
                neg_raw_preds=neg_raw,
            )
        )
    return rows


def main() -> int:
    feat_path = TABLES / "subjects_features_with_release.csv"
    ref_path = TABLES / "tae_summary_full.csv"
    out_path = TABLES / "model_comparison.csv"

    feat = pd.read_csv(feat_path)
    print(f"[load] {feat_path.name}: {len(feat)} subjects")

    rows: list[dict] = []
    for band in BANDS:
        for metric in METRICS:
            print(f"[fit] {band} / {metric}")
            rows.extend(run_one_fit(feat, band, metric))

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, float_format="%.6g")
    print(f"[save] {out_path} ({len(out)} rows)")

    # ---- sanity check vs tae_summary_full.csv ----
    ref = pd.read_csv(ref_path)
    pw = out[out["model"] == "power_law"].merge(
        ref[["band", "metric", "alpha", "r2"]],
        on=["band", "metric"],
        suffixes=("_a1", "_ref"),
    )
    d_alpha = float(np.max(np.abs(pw["alpha_loglog"] - pw["alpha"])))
    d_r2 = float(np.max(np.abs(pw["r2_a1"] - pw["r2_ref"])))
    print(f"[check] power_law vs tae_summary_full: max|d_alpha|={d_alpha:.3e}, max|d_r2|={d_r2:.3e}")
    if d_alpha > 1e-6 or d_r2 > 1e-6:
        print("[check] MISMATCH — stopping.")
        return 1

    # ---- quick summary ----
    print("\n[summary] best model per fit")
    print("  (AIC comparable only within dv_space; CV RMSE raw is the cross-space referee)")
    winners = {"aic_raw": [], "aic_log": [], "cv_raw": []}
    for (b, m), g in out.groupby(["band", "metric"]):
        raw_g = g[g["dv_space"] == "raw"]
        log_g = g[g["dv_space"] == "log"]
        winners["aic_raw"].append(raw_g.loc[raw_g["aic"].idxmin(), "model"])
        winners["aic_log"].append(log_g.loc[log_g["aic"].idxmin(), "model"])
        winners["cv_raw"].append(g.loc[g["cv_rmse_raw"].idxmin(), "model"])
    for k, v in winners.items():
        vc = pd.Series(v).value_counts()
        print(f"  {k}: " + ", ".join(f"{m}={c}" for m, c in vc.items()))

    pw_cv = out[out["model"] == "power_law"][["band", "metric", "cv_rmse_raw"]]
    qu_cv = out[out["model"] == "quadratic"][["band", "metric", "cv_rmse_raw"]]
    cmp_df = pw_cv.merge(qu_cv, on=["band", "metric"], suffixes=("_pw", "_quad"))
    cmp_df["pw_minus_quad"] = cmp_df["cv_rmse_raw_pw"] - cmp_df["cv_rmse_raw_quad"]
    print("\n[power_law vs quadratic, CV RMSE raw]")
    print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    sys.exit(main())
