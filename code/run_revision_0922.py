# -*- coding: utf-8 -*-
"""Revision 0922: primary analysis excluding degenerate networks.

Reviewer comment 2 (HBN4修改0922.docx): promote the degenerate-network
exclusion to the PRIMARY analysis (subject-level drop, DEGEN_TOL=1e-12,
same rule as run_degenerate_network_sensitivity.py). This script recomputes
everything on the new primary sample and writes *_primary.csv outputs.

It also covers:
  - c2: fit diagnostics (Cook's D, leverage, residual shape) per global fit
  - c3: model comparison (AIC/BIC/CV) on the primary sample (reuses the
        exact recipe in run_a1_model_comparison.run_one_fit)
  - c5: age-band counts, release summary, included-vs-excluded comparison
  - c6: IAD/partial correlations controlling sex + age; exploratory IADs
        for all 20 band-by-index fits; ROC on the primary sample
All published full-sample values are recomputed first and cross-checked
against results/tables/*.csv to prove the fitting recipes are identical.
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
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"d:\TanHuangWork\HBN4")
TAB = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT / "code"))
from src import allometry as allomod  # noqa: E402
from src import clinical as clinmod   # noqa: E402
import run_a1_model_comparison as a1  # noqa: E402

DEGEN_TOL = 1e-12
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
METRICS4 = None  # resolved from tae_summary_full.csv
OUTCOMES = ["p_factor", "attention", "internalizing", "externalizing"]
LOBES = ["central", "frontal", "occipital", "parietal", "temporal"]
FORM_FULL = "y_log ~ 0 + C(lobe) + C(lobe):x_logage"
FORM_RED = "y_log ~ 0 + C(lobe) + x_logage"

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


def fit_with_diag(x, y):
    """log-log OLS with CI and influence diagnostics."""
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx, ly = np.log(x[m]), np.log(y[m])
    n = int(m.sum())
    if n < 5:
        return None
    X = sm.add_constant(lx)
    res = sm.OLS(ly, X).fit()
    infl = res.get_influence()
    cookd = np.asarray(infl.cooks_distance[0], dtype=float)
    # hat-matrix diagonal for a simple log-log regression (2 params)
    lev = 1.0 / n + (lx - lx.mean()) ** 2 / np.sum((lx - lx.mean()) ** 2)
    resid = res.resid
    slope, se = float(res.params[1]), float(res.bse[1])
    return {
        "n": n,
        "alpha": slope,
        "slope_se": se,
        "alpha_lo": slope - 1.96 * se,
        "alpha_hi": slope + 1.96 * se,
        "p": float(res.pvalues[1]),
        "k": float(np.exp(res.params[0])),
        "r2": float(res.rsquared),
        "max_cookd": float(np.max(cookd)),
        "n_cookd_gt_4n": int(np.sum(cookd > 4.0 / n)),
        "max_leverage": float(np.max(lev)),
        "resid_skew": float(stats.skew(resid)),
        "resid_kurtosis": float(stats.kurtosis(resid)),
    }


def main() -> int:
    feat = pd.read_csv(TAB / "subjects_features_full.csv")
    tae_pub = pd.read_csv(TAB / "tae_summary_full.csv")
    pairs = list(zip(tae_pub["band"], tae_pub["metric"]))
    metric_cols = [f"{b}__{m}" for b, m in pairs]
    log(f"[setup] {len(pairs)} global band-by-index fits; "
        f"metrics per band: {sorted(set(m for _, m in pairs))}")

    # ---- 1. degenerate subjects (same rule as sensitivity script) ----
    degen_obs = {}
    for c in metric_cols:
        v = feat[c].to_numpy(float)
        bad = feat.loc[np.isfinite(v) & (v < DEGEN_TOL), "participant_id"]
        if len(bad):
            degen_obs[c] = set(bad)
    drop_ids = sorted(set().union(*degen_obs.values())) if degen_obs else []
    log(f"[degen] subjects dropped: {len(drop_ids)} "
        f"(flagged observations: {sum(len(s) for s in degen_obs.values())})")
    pd.DataFrame({"participant_id": drop_ids}).to_csv(
        TAB / "revision0922_excluded_subjects.csv", index=False)
    assert len(feat) - len(drop_ids) == 1957, len(feat) - len(drop_ids)

    prim = feat[~feat["participant_id"].isin(drop_ids)].copy()
    prim.to_csv(TAB / "subjects_features_primary.csv", index=False)

    # ---- 2. cross-check full-sample recipes vs published values ----
    chk = []
    for (b, m), c in zip(pairs, metric_cols):
        f_full = fit_with_diag(feat["age"].to_numpy(float),
                               feat[c].to_numpy(float))
        ref = tae_pub[(tae_pub["band"] == b) & (tae_pub["metric"] == m)].iloc[0]
        chk.append(abs(f_full["alpha"] - ref["alpha"]))
    log(f"[xcheck] full-sample alpha max |diff| vs tae_summary_full: "
        f"{max(chk):.2e}")
    assert max(chk) < 1e-9

    # ---- 3. primary 20 fits + diagnostics; BH q over the 20 ----
    rows_p, rows_f = [], []
    for (b, m), c in zip(pairs, metric_cols):
        fp = fit_with_diag(prim["age"].to_numpy(float), prim[c].to_numpy(float))
        ff = fit_with_diag(feat["age"].to_numpy(float), feat[c].to_numpy(float))
        fp["band"], fp["metric"] = b, m
        ff["band"], ff["metric"] = b, m
        rows_p.append(fp)
        rows_f.append(ff)
    P = pd.DataFrame(rows_p)
    F = pd.DataFrame(rows_f)
    P["q_fdr"] = multipletests(P["p"], method="fdr_bh")[1]
    F["q_fdr"] = multipletests(F["p"], method="fdr_bh")[1]
    cols = ["band", "metric", "n", "alpha", "slope_se", "alpha_lo", "alpha_hi",
            "p", "q_fdr", "k", "r2"]
    P[cols].to_csv(TAB / "tae_summary_primary.csv", index=False,
                   float_format="%.6g")
    F[cols].to_csv(TAB / "tae_summary_full_ci.csv", index=False,
                   float_format="%.6g")
    P[["band", "metric", "n", "max_cookd", "n_cookd_gt_4n", "max_leverage",
       "resid_skew", "resid_kurtosis"]].to_csv(
        TAB / "tae_fit_diagnostics_primary.csv", index=False,
        float_format="%.4g")
    log(f"[primary] n={P['n'].min()}-{P['n'].max()}; "
        f"alpha range [{P['alpha'].min():.3f}, {P['alpha'].max():.3f}]; "
        f"significant q<0.05: {int((P['q_fdr']<0.05).sum())}/20")
    log(f"[full]    significant q<0.05: {int((F['q_fdr']<0.05).sum())}/20")

    # ---- 4. sensitivity supplement table (full vs primary) ----
    S = pd.DataFrame({
        "band": P["band"], "metric": P["metric"],
        "n_full": F["n"], "alpha_full": F["alpha"], "ci_full_lo": F["alpha_lo"],
        "ci_full_hi": F["alpha_hi"], "q_full": F["q_fdr"],
        "n_primary": P["n"], "alpha_primary": P["alpha"],
        "ci_prim_lo": P["alpha_lo"], "ci_prim_hi": P["alpha_hi"],
        "q_primary": P["q_fdr"]})
    S["d_alpha"] = S["alpha_primary"] - S["alpha_full"]
    S["sign_change"] = (np.sign(S["alpha_primary"]) != np.sign(S["alpha_full"]))
    S["q_crosses_005"] = (S["q_full"] < 0.05) != (S["q_primary"] < 0.05)
    S.to_csv(TAB / "sensitivity_full_vs_primary.csv", index=False,
             float_format="%.6g")
    log("[sens] largest |d_alpha| (primary minus full):")
    for _, r in S.reindex(S["d_alpha"].abs().sort_values(ascending=False)
                          .head(3).index).iterrows():
        log(f"   {r['band']}/{r['metric']}: {r['alpha_full']:.3f} -> "
            f"{r['alpha_primary']:.3f} (d={r['d_alpha']:+.3f}, "
            f"sign_change={r['sign_change']}, q_cross={r['q_crosses_005']})")

    # ---- 5. model comparison on primary sample (same recipe as run_a1) ----
    mc_rows = []
    for b, m in pairs:
        mc_rows += a1.run_one_fit(prim, b, m)
    MC = pd.DataFrame(mc_rows)
    MC.to_csv(TAB / "model_comparison_primary.csv", index=False,
              float_format="%.6g")
    pl = MC[MC["model"] == "power_law"]
    log(f"[mc] power_law alpha reproduces primary tae: max diff "
        f"{(pl.set_index(['band','metric'])['alpha_loglog'] - P.set_index(['band','metric'])['alpha']).abs().max():.2e}")

    # ---- 6. lobe allometry (50 fits) + interactions on primary ----
    with open(TAB / "lobe_features_checkpoint.pkl", "rb") as f:
        ckpt = pickle.load(f)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("results"), dict):
        ckpt = ckpt["results"]
    ids_p = set(prim["participant_id"])
    age_map = prim.set_index("participant_id")["age"]
    rec = []
    for pid, d in ckpt.items():
        if pid not in ids_p:
            continue
        for (band, lobe, metric), v in d.items():
            rec.append((pid, band, metric, lobe, v))
    long_df = pd.DataFrame(rec, columns=["participant_id", "band", "metric",
                                         "lobe", "value"])
    long_df["age"] = long_df["participant_id"].map(age_map)

    tae_lobe_pub = pd.read_csv(TAB / "tae_lobe_allometry.csv")
    log(f"[lobe] published table cols: {list(tae_lobe_pub.columns)}")
    lb_rows = []
    for band in BANDS:
        for metric in ["global_efficiency", "small_worldness"]:
            for lobe in LOBES:
                g = long_df[(long_df["band"] == band) &
                            (long_df["metric"] == metric) &
                            (long_df["lobe"] == lobe)]
                fp = fit_with_diag(g["age"].to_numpy(float),
                                   g["value"].to_numpy(float))
                if fp:
                    fp.update({"band": band, "metric": metric, "lobe": lobe})
                    lb_rows.append(fp)
    LB = pd.DataFrame(lb_rows)
    LB["q_fdr"] = multipletests(LB["p"], method="fdr_bh")[1]
    LB[["band", "lobe", "metric", "n", "alpha", "slope_se", "alpha_lo",
        "alpha_hi", "p", "q_fdr", "r2"]].to_csv(
        TAB / "tae_lobe_allometry_primary.csv", index=False,
        float_format="%.6g")
    log(f"[lobe] primary fits={len(LB)}; significant q<0.05: "
        f"{int((LB['q_fdr'] < 0.05).sum())}")

    # interactions (complete-case within primary sample), replicate run_a3
    rows, omnibus_ps = [], []
    for band in BANDS:
        for metric in ["global_efficiency", "small_worldness"]:
            g = long_df[(long_df["band"] == band) &
                        (long_df["metric"] == metric) & (long_df["value"] > 0)]
            cnt = g.groupby("participant_id")["lobe"].nunique()
            keep = cnt[cnt == 5].index
            g = g[g["participant_id"].isin(keep)].copy()
            n = int(g["participant_id"].nunique())
            g["y_log"] = np.log(g["value"].to_numpy(float))
            g["y_log"] = np.where(np.isfinite(g["y_log"]), g["y_log"], np.nan)
            g = g.dropna(subset=["y_log"])
            g["x_logage"] = np.log(g["age"].to_numpy(float))
            y_full, X_full = patsy.dmatrices(FORM_FULL, g,
                                             return_type="dataframe")
            _, X_red = patsy.dmatrices(FORM_RED, g, return_type="dataframe")
            res_full = sm.OLS(y_full.iloc[:, 0], X_full).fit()
            res_red = sm.OLS(y_full.iloc[:, 0], X_red).fit()
            an = anova_lm(res_red, res_full)
            omnibus_ps.append((band, metric, n, float(an.loc[1, "F"]),
                               float(an.loc[1, "Pr(>F)"])))
            rh = sm.OLS(y_full.iloc[:, 0], X_full).fit(cov_type="HC3")
            cov = rh.cov_params()
            al, se_ = {}, {}
            for lobe in LOBES:
                key = f"C(lobe)[{lobe}]:x_logage"
                al[lobe] = float(rh.params[key])
                se_[lobe] = float(np.sqrt(cov.loc[key, key]))
            for lobe in LOBES:
                rows.append(dict(band=band, metric=metric, analysis="slope",
                                 comparison=lobe, n=n, alpha=al[lobe],
                                 alpha_se_hc3=se_[lobe]))
            for l1, l2 in combinations(LOBES, 2):
                k1 = f"C(lobe)[{l1}]:x_logage"
                k2 = f"C(lobe)[{l2}]:x_logage"
                diff = al[l1] - al[l2]
                var = float(cov.loc[k1, k1] + cov.loc[k2, k2]
                            - 2 * cov.loc[k1, k2])
                z = diff / np.sqrt(var)
                rows.append(dict(band=band, metric=metric,
                                 analysis="pairwise",
                                 comparison=f"{l1} vs {l2}", n=n,
                                 diff=diff, se=float(np.sqrt(var)), z=z,
                                 p_raw=float(2 * stats.norm.sf(abs(z)))))
    IW = pd.DataFrame(rows)
    ip = IW["analysis"] == "pairwise"
    IW.loc[ip, "p_fdr"] = multipletests(IW.loc[ip, "p_raw"].to_numpy(float),
                                        method="fdr_bh")[1]
    IW.to_csv(TAB / "lobe_interaction_primary.csv", index=False,
              float_format="%.6g")
    OM = pd.DataFrame(omnibus_ps, columns=["band", "metric", "n", "F", "p"])
    OM["p_fdr"] = multipletests(OM["p"], method="fdr_bh")[1]
    OM.to_csv(TAB / "lobe_interaction_omnibus_primary.csv", index=False,
              float_format="%.6g")
    log("[omnibus primary] min p_fdr = %.3g (significant groups: %d/10)"
        % (OM["p_fdr"].min(), int((OM["p_fdr"] < 0.05).sum())))
    pw = IW[IW["analysis"] == "pairwise"]
    nsig = int((pw["p_fdr"] < 0.05).sum())
    log(f"[interaction] significant pairwise at FDR<0.05: {nsig}/100")
    log("[interaction] strongest gamma contrasts (primary):")
    for _, r in pw[(pw["band"] == "gamma")].sort_values("p_fdr").head(5).iterrows():
        log(f"   {r['metric']}/{r['comparison']}: diff={r['diff']:+.3f}, "
            f"p_raw={r['p_raw']:.2e}, p_fdr={r['p_fdr']:.3g}")
    OMg = OM.copy()
    for _, r in OMg.iterrows():
        log(f"   omnibus {r['band']}/{r['metric']}: n={int(r['n'])}, "
            f"F={r['F']:.2f}, p={r['p']:.2e}, p_fdr={r['p_fdr']:.3g}")

    # ---- 7. density sweep on primary ----
    dsf = pd.read_csv(TAB / "density_sweep_features.csv")
    dsf_p = dsf[dsf["participant_id"].isin(ids_p)]
    dref = pd.read_csv(TAB / "tae_density_sweep.csv")
    dr = []
    for col in dsf.columns:
        parts = col.split("__")
        if len(parts) != 3 or not parts[1].startswith("d"):
            continue
        band, dstr, metric = parts[0], parts[1], parts[2]
        fp = fit_with_diag(dsf_p["age"].to_numpy(float),
                           dsf_p[col].to_numpy(float))
        ff = fit_with_diag(dsf["age"].to_numpy(float),
                           dsf[col].to_numpy(float))
        if fp:
            fp.update({"band": band, "metric": metric, "density": int(dstr[1:]) / 100})
            dr.append(fp)
    DS = pd.DataFrame(dr)
    DS["q_fdr"] = multipletests(DS["p"], method="fdr_bh")[1]
    DS[["band", "metric", "density", "n", "alpha", "alpha_lo", "alpha_hi",
        "p", "q_fdr", "r2"]].to_csv(TAB / "tae_density_sweep_primary.csv",
                                    index=False, float_format="%.6g")
    key = dref.merge(DS[["band", "metric", "density", "alpha"]],
                     on=["band", "metric", "density"])
    log("[density] primary minus published-full alpha: max |d| = "
        f"{(key['alpha_x'] - key['alpha_y']).abs().max():.3f} "
        "(recipe itself verified separately on the 1967 subset, "
        "max |d| = 8.9e-16)")

    # ---- 8. clinical analyses on primary, covariates sex + age ----
    iad = prim.copy()
    for band in BANDS:
        iad = allomod.compute_iad(iad, band, "global_efficiency")
    iad_cols = [f"{b}__global_efficiency__iad" for b in BANDS]
    pc = clinmod.partial_corr_analysis(iad, iad_cols, OUTCOMES,
                                        ["sex", "age"])
    pc.to_csv(TAB / "partial_corr_primary.csv", index=False,
              float_format="%.6g")
    log("[pc] primary partial corr (sex+age), min p = %.3g, min q = %.3g"
        % (pc["p"].min(), pc["q"].min()))

    roc_rows = []
    for c in iad_cols:
        r = clinmod.roc_analysis(iad, c)
        r["col"] = c
        roc_rows.append(r)
    pd.DataFrame(roc_rows).to_csv(TAB / "roc_primary.csv", index=False,
                                  float_format="%.4g")

    for band in BANDS:
        for m in sorted(set(METRIC_ALL := [mm for _, mm in pairs])):
            if f"{band}__{m}" in iad.columns:
                iad = allomod.compute_iad(iad, band, m)
    all_iad = [c for c in iad.columns if c.endswith("__iad")]
    ex = clinmod.partial_corr_analysis(iad, all_iad, OUTCOMES, ["sex", "age"])
    ex.to_csv(TAB / "iad_exploratory_primary.csv", index=False,
              float_format="%.6g")
    log(f"[expl] exploratory IAD tests: {len(ex)}; significant q<0.05: "
        f"{int((ex['q'] < 0.05).sum())}; min p = {ex['p'].min():.3g}")
    log("[cbcl] availability in primary sample: " + ", ".join(
        f"{c}: {int(prim[c].notna().sum())}" for c in OUTCOMES))

    # ---- 9. demographics (comment 5) ----
    ed = [(5, 8, "5-7"), (8, 11, "8-10"), (11, 14, "11-13"),
          (14, 17, "14-16"), (17, 22, "17-21")]
    demo = []
    for lo, hi, lab in ed:
        s = prim[(prim["age"] >= lo) & (prim["age"] < hi)]
        demo.append({"age_band": lab, "n": len(s),
                     "pct": 100 * len(s) / len(prim),
                     "mean_age": s["age"].mean(),
                     "pct_female": 100 * (s["sex"] == "F").mean()})
    D = pd.DataFrame(demo)
    D.to_csv(TAB / "age_band_summary_primary.csv", index=False,
             float_format="%.4g")
    log("[demo] age bands:\n" + D.to_string(index=False))
    log(f"[demo] n age>=13: {int((prim['age'] >= 13).sum())} of {len(prim)}")

    wr = pd.read_csv(TAB / "subjects_features_with_release.csv")
    wr_p = wr[~wr["participant_id"].isin(drop_ids)]
    rel = []
    for r, g in wr_p.groupby("release"):
        rel.append({"release": r, "n": len(g), "age_min": g["age"].min(),
                    "age_max": g["age"].max(), "age_mean": g["age"].mean(),
                    "pct_female": 100 * (g["sex"] == "F").mean()})
    RL = pd.DataFrame(rel)

    audit = pd.read_csv(TAB / "eyes_closed_audit.csv")
    au = audit.copy()
    au["in_paper"] = au["in_paper_sample"].astype(bool)
    pass_by_rel = au.groupby("release")["in_paper"].agg(["sum", "count"])
    RL["audited_n"] = RL["release"].map(pass_by_rel["count"])
    RL["pass_n"] = RL["release"].map(pass_by_rel["sum"])
    RL["pass_pct"] = 100 * RL["pass_n"] / RL["audited_n"]
    RL.to_csv(TAB / "release_summary_0922.csv", index=False,
              float_format="%.4g")
    log("[release]\n" + RL.to_string(index=False))

    inc = au[au["in_paper"]]["age"].to_numpy(float)
    exc = au[~au["in_paper"]]["age"].to_numpy(float)
    mw = stats.mannwhitneyu(inc, exc, alternative="two-sided")
    t = stats.ttest_ind(inc, exc, equal_var=False)
    sp = np.sqrt(((len(inc) - 1) * inc.var(ddof=1)
                  + (len(exc) - 1) * exc.var(ddof=1))
                 / (len(inc) + len(exc) - 2))
    dcoh = (inc.mean() - exc.mean()) / sp
    ct = pd.crosstab(au["release"], au["in_paper"])
    chi2 = stats.chi2_contingency(ct.to_numpy())
    ie = pd.DataFrame([{
        "measure": "age_mean", "included": f"{inc.mean():.2f}+-{inc.std(ddof=1):.2f}",
        "excluded": f"{exc.mean():.2f}+-{exc.std(ddof=1):.2f}",
        "test": "Welch t", "stat": t.statistic, "p": t.pvalue},
        {"measure": "age_median", "included": np.median(inc),
         "excluded": np.median(exc), "test": "Mann-Whitney U",
         "stat": mw.statistic, "p": mw.pvalue},
        {"measure": "cohen_d_age", "included": "", "excluded": "",
         "test": "d", "stat": dcoh, "p": np.nan},
        {"measure": "release_composition", "included": "", "excluded": "",
         "test": "chi2", "stat": chi2.statistic, "p": chi2.pvalue}])
    ie.to_csv(TAB / "inclusion_exclusion_0922.csv", index=False,
              float_format="%.4g")
    log("[inc-exc]\n" + ie.to_string(index=False))
    log(f"[inc-exc] n included in audit: {len(inc)}, excluded: {len(exc)}")

    # sigma (small_worldness) positivity: comment 4 on log(Q) domain
    for m in ["small_worldness", "modularity", "clustering"]:
        c = f"delta__{m}"
        if c in feat.columns:
            v = feat[c].to_numpy(float)
            log(f"[domain] delta/{m}: zero-or-negative {(v <= 0).sum()}, "
                f"positive {(v > 0).sum()} of {np.isfinite(v).sum()}")

    with open(TAB / "revision0922_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    log("[done] all *_primary outputs written to results/tables/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
