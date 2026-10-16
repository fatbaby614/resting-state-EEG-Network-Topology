# -*- coding: utf-8 -*-
"""
run_degenerate_network_sensitivity.py

Sensitivity check for near-zero (numerically degenerate) network metrics.

Background
----------
In the analysed sample (n = 1,967) a small number of subject x metric
observations take values of order 1e-16, i.e. they are float64-level zero
(machine epsilon 2.2e-16). These correspond to completely disconnected
thresholded networks, for which efficiency-type indices collapse to 0 and
small-worldness sigma collapses with them. Because the allometric fit is
performed on log-log coordinates, log(1e-16) ~ -36.8 sits between 14 and 44
standard deviations below the bulk of the data, depending on the index, and
carries enormous leverage.

Threshold choice (NOT result-driven)
------------------------------------
The cut-off 1e-12 is placed below the degenerate cluster and above the
smallest non-degenerate value of the same index. Within every affected index
the cluster is separated from that value by 6.7-10.9 orders of magnitude, so
the threshold does not dissect the bulk of any distribution. The threshold is
therefore determined by the data structure and is identical for all 20 fits;
it was not selected by comparing fitted exponents. Only the modularity
indices, which never go degenerate, take values near the cut; these belong to
the same 10 flagged subjects and are removed with them. A looser cut such as
1e-2 would be indefensible: gamma-band global efficiency alone has 441
legitimate observations below 1e-2.

Two exclusion variants are reported
-----------------------------------
  A) per-fit  : drop only the subject x fit rows whose metric value is < 1e-12
  B) whole-subject : drop every subject that is degenerate in >= 1 fit, from
                     all 20 fits (stricter, avoids any partial-subject reuse)

Published results are NOT modified. This script only writes a companion table.

Output: results/tables/degenerate_network_sensitivity.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
FEAT = TABLES / "subjects_features_with_release.csv"
REF = TABLES / "tae_summary_full.csv"

DEGEN_TOL = 1e-12
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
METRICS = ["global_efficiency", "local_efficiency", "modularity", "small_worldness"]


def fit_loglog(df: pd.DataFrame, col: str) -> dict:
    """Same recipe that reproduces tae_summary_full.csv to 3e-16."""
    s = df[["age", col]].dropna()
    s = s[(s["age"] > 0) & (s[col] > 0)]
    r = stats.linregress(np.log(s["age"].to_numpy()), np.log(s[col].to_numpy()))
    return dict(n=len(s), alpha=r.slope, r2=r.rvalue ** 2, p=r.pvalue)


def main() -> None:
    feat = pd.read_csv(FEAT)
    feat["participant_id"] = feat["participant_id"].astype(str)
    ref = pd.read_csv(REF)

    # ---- subjects degenerate in at least one of the 20 fits -----------------
    degen_mask = pd.Series(False, index=feat.index)
    for b in BANDS:
        for m in METRICS:
            c = f"{b}__{m}"
            degen_mask |= feat[c].notna() & (feat[c].abs() < DEGEN_TOL)
    degen_ids = sorted(feat.loc[degen_mask, "participant_id"].unique())
    feat_drop = feat[~feat["participant_id"].isin(degen_ids)]
    print(f"degenerate tolerance      : {DEGEN_TOL:g}")
    print(f"subjects degenerate >=1 fit: {len(degen_ids)} "
          f"({100 * len(degen_ids) / len(feat):.2f}% of n={len(feat)})")
    for p in degen_ids:
        print("   ", p)

    rows = []
    for b in BANDS:
        for m in METRICS:
            col = f"{b}__{m}"
            full = fit_loglog(feat, col)
            # variant A: per-fit exclusion
            a = fit_loglog(feat[~((feat[col].abs() < DEGEN_TOL) & feat[col].notna())], col)
            # variant B: whole-subject exclusion
            bb = fit_loglog(feat_drop, col)
            n_degen = int(((feat[col].notna()) & (feat[col].abs() < DEGEN_TOL)).sum())
            rows.append(dict(
                band=b, metric=m,
                n_full=full["n"], n_degen_obs=n_degen,
                alpha_full=full["alpha"], r2_full=full["r2"], p_full=full["p"],
                alpha_perfit=a["alpha"], n_perfit=a["n"],
                r2_perfit=a["r2"], p_perfit=a["p"],
                d_alpha_perfit=a["alpha"] - full["alpha"],
                alpha_subjdrop=bb["alpha"], n_subjdrop=bb["n"],
                r2_subjdrop=bb["r2"], p_subjdrop=bb["p"],
                d_alpha_subjdrop=bb["alpha"] - full["alpha"],
            ))

    out = pd.DataFrame(rows)
    dest = TABLES / "degenerate_network_sensitivity.csv"
    out.to_csv(dest, index=False)
    print(f"\n[saved] {dest}")

    # ---- cross-check that alpha_full reproduces the published table ---------
    chk = out.merge(ref[["band", "metric", "alpha"]].rename(columns={"alpha": "pub"}),
                    on=["band", "metric"], how="left")
    err = (chk["alpha_full"] - chk["pub"]).abs().max()
    print(f"cross-check vs tae_summary_full.csv: "
          f"max|alpha_full - alpha_published| = {err:.3e}")
    assert err < 1e-9, "reproduction of published alphas failed -> wrong fit recipe"

    pd.set_option("display.width", 220)
    print("\n" + out[["band", "metric", "n_degen_obs", "alpha_full",
                      "alpha_perfit", "d_alpha_perfit", "alpha_subjdrop",
                      "d_alpha_subjdrop"]].to_string(
        index=False, float_format=lambda x: f"{x:+.5f}"))

    for tag in ("perfit", "subjdrop"):
        d = out[f"d_alpha_{tag}"].abs()
        i = int(d.idxmax())
        print(f"\nmax |delta alpha| ({tag}) = {d.max():.5f}  "
              f"at {out.loc[i, 'band']} / {out.loc[i, 'metric']} "
              f"({out.loc[i, 'alpha_full']:.5f} -> {out.loc[i, f'alpha_{tag}']:.5f})")
        print(f"  median |delta| = {d.median():.5f} ; "
              f"n fits with |delta| > 0.05 = {int((d > 0.05).sum())}/20")

    # significance pattern: does any fit change qualitative conclusion?
    flips = out[(out["p_full"] < 0.05) != (out["p_perfit"] < 0.05)]
    print(f"\nfits whose 0.05 significance verdict flips (per-fit variant): {len(flips)}")
    print(f"rank order of the 20 exponents preserved (perfit vs full): "
          f"{out['alpha_full'].rank().equals(out['alpha_perfit'].rank())}")


if __name__ == "__main__":
    main()
