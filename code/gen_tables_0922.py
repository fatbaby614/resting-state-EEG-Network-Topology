# -*- coding: utf-8 -*-
"""Build LaTeX fragments for the 0922 revision.

Primary analysis = subjects whose networks were numerically degenerate removed
(n = 1,957).  Full-sample results (n = 1,967) become a sensitivity analysis.

Outputs
  paper/tables/frag_t{1,2,3,4}.tex  + frag_t3head.tex  main-text tabulars
  paper/tables/s*.tex               complete appendix floats
  paper/tables/revision0922_numbers.txt  narrative numbers for the prose
"""
import math
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"d:\TanHuangWork\HBN4")
TAB = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
B = chr(92)
TXT = []


def log(s=""):
    TXT.append(str(s))


# ---------------------------------------------------------------- formatting
def fl(x):
    return int(math.floor(math.log10(x)))


RMF = B + "mathrm{full}"
RMP = B + "mathrm{prim}"


def fsci(x, d=3):
    x = float(x)
    if x <= 0:
        return f"${B}ll 10^{{-300}}$"
    if x >= 10.0 ** (-d):
        return f"${x:.{d}f}$"
    e = fl(x)
    return f"${x / 10.0 ** e:.1f}{B}times10^{{{e}}}$"


def fnum(x, d=3):
    return f"${x:.{d}f}$"


def fci(lo, hi):
    return f"$[{lo:.3f}, {hi:.3f}]$"


def apct(x):
    v = float(x) * 100.0
    return f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def acell(a, q, dag=False):
    s = f"{a:.3f}"
    core = f"{B}mathbf{{{s}}}" if q < 0.05 else s
    if dag and 0.05 <= q < 0.10:
        core += "^{" + B + "dagger}"
    return f"${core}$"


def sdash(x, d=3):
    if pd.isna(x):
        return "--"
    return f"${x:.{d}f}$"


# ---------------------------------------------------------------- constants
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BG = {b: f"${B}{b}$" for b in BANDS}
METS = ["small_worldness", "global_efficiency", "local_efficiency",
        "modularity"]
MI = {"small_worldness": f"${B}sigma$",
      "global_efficiency": f"${B}mathrm{{E}}_{{{B}rm glob}}$",
      "local_efficiency": f"${B}mathrm{{E}}_{{{B}rm loc}}$",
      "modularity": "$Q$"}
MNAME = {"small_worldness": "Small-worldness " + f"${B}sigma$",
         "global_efficiency": "Global efficiency",
         "local_efficiency": "Local efficiency",
         "modularity": "Modularity"}
LOBES = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]
ORDER = {b: i for i, b in enumerate(BANDS)}
MORD = {m: i for i, m in enumerate(METS)}
OUTC = {"p_factor": 0, "attention": 1, "internalizing": 2, "externalizing": 3}
ONM = {"p_factor": "Total problems", "attention": "Attention problems",
       "internalizing": "Internalizing", "externalizing": "Externalizing"}


def okey(df):
    d = df.copy()
    d["_b"] = d["band"].map(ORDER)
    d["_m"] = d["metric"].map(MORD)
    return d.sort_values(["_b", "_m"])


def okey_iad(df):
    d = df.copy()
    p = d["iad_col"].astype(str).str.split("__", expand=True)
    d["band"] = p[0]
    d["metric"] = p[1]
    d["_b"] = d["band"].map(ORDER)
    d["_m"] = d["metric"].map(MORD)
    d["_o"] = d["outcome"].map(OUTC)
    return d.sort_values(["_b", "_m", "_o"])


# ---------------------------------------------------------------- load
S = okey(pd.read_csv(TAB / "tae_summary_primary.csv"))
SFL = okey(pd.read_csv(TAB / "tae_summary_full_ci.csv"))
DIAG = okey(pd.read_csv(TAB / "tae_fit_diagnostics_primary.csv"))
SF = okey(pd.read_csv(TAB / "sensitivity_full_vs_primary.csv"))
MC = pd.read_csv(TAB / "model_comparison_primary.csv")
DS = pd.read_csv(TAB / "tae_density_sweep_primary.csv")
LO = okey(pd.read_csv(TAB / "tae_lobe_allometry_primary.csv"))
OM = okey(pd.read_csv(TAB / "lobe_interaction_omnibus_primary.csv"))
OFL = pd.read_csv(TAB / "lobe_interaction_omnibus.csv")
OFL = OFL[OFL["analysis"] == "omnibus"].copy()
PW = pd.read_csv(TAB / "lobe_interaction_primary.csv")
PW = PW[PW["analysis"] == "pairwise"].copy()
AGE = pd.read_csv(TAB / "age_band_summary_primary.csv")
REL = pd.read_csv(TAB / "release_summary_0922.csv")
INC = pd.read_csv(TAB / "inclusion_exclusion_0922.csv")
PC = okey_iad(pd.read_csv(TAB / "partial_corr_primary.csv"))
EX = okey_iad(pd.read_csv(TAB / "iad_exploratory_primary.csv"))
ROC = pd.read_csv(TAB / "roc_primary.csv")
ROC = ROC[ROC["auc"].notna()].copy()
EXCL = pd.read_csv(TAB / "revision0922_excluded_subjects.csv")

log("=" * 70 + "\nEXCLUDED SUBJECTS (numerically degenerate networks)")
log(", ".join(EXCL["participant_id"].astype(str)))

# ================================================================ headline
sig_n = int((S["q_fdr"] < 0.05).sum())
log("=" * 70 + "\nPRIMARY HEADLINE")
log(f"{sig_n}/20 q<0.05; alpha {S['alpha'].min():.3f} to {S['alpha'].max():.3f}"
    f"; R2 {S['r2'].min()*100:.2f}% to {S['r2'].max()*100:.2f}%")
for _, r in S[S["q_fdr"] >= 0.05].iterrows():
    log(f"  NS {r['band']}/{r['metric']}: a={r['alpha']:+.3f} "
        f"CI={r['alpha_lo']:+.3f},{r['alpha_hi']:+.3f} q={r['q_fdr']:.3g}")
for b in BANDS:
    r = S[S["band"] == b].sort_values("alpha").iloc[0]
    log(f"  steepest {b}: {r['metric']} {r['alpha']:+.3f} q={r['q_fdr']:.2g} "
        f"R2={r['r2']*100:.1f}%")
for b in BANDS:
    r = S[(S["band"] == b) & (S["metric"] == "modularity")].iloc[0]
    log(f"  modularity {b}: {r['alpha']:+.3f} q={r['q_fdr']:.2g}")
for b in ["alpha", "gamma"]:
    for m in ["global_efficiency", "small_worldness"]:
        r = S[(S["band"] == b) & (S["metric"] == m)].iloc[0]
        log(f"  {b}/{m}: {r['alpha']:+.3f} CI {r['alpha_lo']:+.3f},"
            f"{r['alpha_hi']:+.3f} q={r['q_fdr']:.2g}")

# ================================================================ main frags
rows = []
for _, r in S.sort_values("alpha").head(6).iterrows():
    rows.append(f"{BG[r['band']]} & {MNAME[r['metric']]} & {int(r['n']):,} & "
                + acell(r["alpha"], r["q_fdr"]) + f" & {r['k']:.3f} & "
                + apct(r["r2"]) + B + "% & " + fsci(r["p"]) + " & "
                + fsci(r["q_fdr"], 2) + " \\\\")
(OUT / "frag_t1.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
log("\nfrag_t1 (6 most negative):")
log("\n".join(r.replace("\\\\", "") for r in rows))

rows = [f"{BG[r['band']]} & {MI[r['metric']]} & {int(r['n']):,} & "
        + acell(r["alpha"], r["q_fdr"]) + " & "
        + fci(r["alpha_lo"], r["alpha_hi"]) + " & " + apct(r["r2"]) + B + "% & "
        + fsci(r["p"]) + " & " + fsci(r["q_fdr"], 2) + " \\\\"
        for _, r in S.iterrows()]
(OUT / "frag_t2.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

dens = sorted(DS["density"].unique())
hd = " & ".join(f"${d:.2f}$" for d in dens)
log(f"\ndensity columns: {dens}")
rows = []
for b in ["delta", "theta"]:
    for m in ["small_worldness", "global_efficiency"]:
        sub = DS[(DS["band"] == b) & (DS["metric"] == m)].set_index("density")
        vs = [float(sub.loc[d, "alpha"]) for d in dens]
        bd = dens[min(range(len(dens)), key=lambda i: vs[i])]
        cells = []
        for i, d in enumerate(dens):
            s = f"{vs[i]:.3f}"
            bold = (d == bd) or (sub.loc[d, "q_fdr"] < 0.05)
            cells.append(f"${B}mathbf{{{s}}}$" if bold else f"${s}$")
        nn = [int(sub.loc[d, "n"]) for d in dens]
        rng = f"{min(nn):,}" if min(nn) == max(nn) else f"{min(nn):,}--{max(nn):,}"
        rows.append(f"{BG[b]} & {MI[m]} ({rng}) & " + " & ".join(cells)
                    + " \\\\")
        log(f"  sweep {b}/{m}: " + ", ".join(f"{d:.2f}:{v:+.3f}"
                                             for d, v in zip(dens, vs)))
(OUT / "frag_t3.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
(OUT / "frag_t3head.tex").write_text(hd + " \\\\\n", encoding="utf-8")
log(f"  sweep max q={DS['q_fdr'].max():.3g}, all alpha<0="
    f"{bool((DS['alpha'] < 0).all())}, n {int(DS['n'].min()):,}-"
    f"{int(DS['n'].max()):,}")

rows = []
for b in BANDS:
    for m in ["global_efficiency", "small_worldness"]:
        cells = []
        for lb in LOBES:
            q = LO[(LO["band"] == b) & (LO["metric"] == m)
                   & (LO["lobe"].str.lower() == lb.lower())]
            if len(q) == 0:
                cells.append("--")
                continue
            q = q.iloc[0]
            qq = float(q["q_fdr"])
            if qq >= 1e-3:
                qs = f"{qq:.3f}"
            else:
                e = fl(qq)
                qs = (f"{qq / 10.0 ** e:.1f}{B}times10^{{{e}}}")
            cells.append(acell(q["alpha"], qq, dag=True)
                         + f"${B}scriptstyle\\,({qs})$")
        rows.append(f"{BG[b]} & {MI[m]} & " + " & ".join(cells) + " \\\\")
(OUT / "frag_t4.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
log("\nfrag_t4 (alpha; q in parentheses, dag = .05<=q<.10):")
log("\n".join(r.replace("\\\\", "") for r in rows))

# ================================================================ lobe text
log("=" * 70 + "\nLOBE (primary)")
ns = LO[LO["q_fdr"] >= 0.05]
log(f"{int((LO['q_fdr'] < 0.05).sum())}/{len(LO)} sig; n {int(LO['n'].min()):,}"
    f"-{int(LO['n'].max()):,}; sig-positive alphas="
    f"{int(((LO['alpha'] > 0) & (LO['q_fdr'] < 0.05)).sum())}")
for _, r in ns.iterrows():
    log(f"  NS {r['band']}/{r['metric']}/{r['lobe']}: {r['alpha']:+.3f} "
        f"q={r['q_fdr']:.3g} n={int(r['n'])}")
for b in BANDS:
    for m in ["global_efficiency", "small_worldness"]:
        s = LO[(LO["band"] == b) & (LO["metric"] == m)].sort_values("alpha")
        log(f"  {b}/{m}: steepest {s.iloc[0]['lobe']} {s.iloc[0]['alpha']:+.3f}"
            f" (q={s.iloc[0]['q_fdr']:.2g}); 2nd {s.iloc[1]['lobe']} "
            f"{s.iloc[1]['alpha']:+.3f}; flattest {s.iloc[-1]['lobe']} "
            f"{s.iloc[-1]['alpha']:+.3f}")
log("  alpha-band E_glob per lobe:")
for lb in LOBES:
    r = LO[(LO["band"] == "alpha") & (LO["metric"] == "global_efficiency")
           & (LO["lobe"].str.lower() == lb.lower())].iloc[0]
    log(f"    {lb}: {r['alpha']:+.3f} q={r['q_fdr']:.2g} n={int(r['n'])}")
log("  gamma-band sigma per lobe:")
for lb in LOBES:
    r = LO[(LO["band"] == "gamma") & (LO["metric"] == "small_worldness")
           & (LO["lobe"].str.lower() == lb.lower())].iloc[0]
    log(f"    {lb}: {r['alpha']:+.3f} q={r['q_fdr']:.2g} n={int(r['n'])}")

# ================================================================ interaction
log("=" * 70 + "\nREGION x AGE INTERACTION (primary)")
for _, r in OM.iterrows():
    log(f"  {r['band']}/{r['metric']}: n={int(r['n'])} F={r['F']:.2f} "
        f"p={r['p']:.3g} q={r['p_fdr']:.3g}"
        + ("  SIG" if r["p_fdr"] < 0.05 else ""))
log(f"  omnibus {(OM['p_fdr'] < 0.05).sum()}/{len(OM)} sig; "
    f"pairwise {(PW['p_fdr'] < 0.05).sum()}/{len(PW)} sig")
for _, r in PW.sort_values("p_fdr").head(14).iterrows():
    log(f"  {r['band']}/{r['metric']} {r['comparison']}: diff={r['diff']:+.3f} "
        f"p={r['p_raw']:.2g} q={r['p_fdr']:.3g}")
log(f"  [full-sample published] p {OFL['p'].min():.3f}-{OFL['p'].max():.3f}, "
    f"min q={OFL['p_fdr'].min():.3g}, sig={(OFL['p_fdr'] < 0.05).sum()}"
    f"/{len(OFL)}")

# ================================================================ sensitivity
log("=" * 70 + "\nSENSITIVITY full vs primary")
mx = SF.loc[SF["d_alpha"].abs().idxmax()]
log(f"max |dalpha| {abs(mx['d_alpha']):.3f} ({mx['band']}/{mx['metric']} "
    f"{mx['alpha_full']:+.3f}->{mx['alpha_primary']:+.3f}); median "
    f"{SF['d_alpha'].abs().median():.3f}")
for _, r in SF[SF["sign_change"]].iterrows():
    log(f"  SIGN {r['band']}/{r['metric']}: {r['alpha_full']:+.3f}->"
        f"{r['alpha_primary']:+.3f} (q {r['q_full']:.3g}->"
        f"{r['q_primary']:.3g})")
for _, r in SF[SF["q_crosses_005"]].iterrows():
    log(f"  QCROSS {r['band']}/{r['metric']}: q {r['q_full']:.3g}->"
        f"{r['q_primary']:.3g}")
log(f"  full-sample: alpha {SFL['alpha'].min():.3f} to {SFL['alpha'].max():.3f}"
    f", all neg={bool((SFL['alpha'] < 0).all())}, "
    f"sig={(SFL['q_fdr'] < 0.05).sum()}/{len(SFL)}")
log(f"  primary n max={int(SF['n_primary'].max())}, "
    f"full n max={int(SF['n_full'].max())}")

# ================================================================ model comp
LOGM, RAWM = ["power_law", "quadratic_log", "gam"], ["linear", "quadratic"]
G = {(b, m): g.set_index("model")
     for (b, m), g in MC.groupby(["band", "metric"])}


def has(gs, ms):
    return all(m in gs.index for m in ms)


log("=" * 70 + "\nMODEL COMPARISON (primary)")
win = {"aic": Counter(), "bic": Counter(), "cvr": Counter(), "cvl": Counter()}
for k, gs in G.items():
    lg = [m for m in LOGM if m in gs.index]
    win["aic"][gs.loc[lg, "aic"].idxmin()] += 1
    win["bic"][gs.loc[lg, "bic"].idxmin()] += 1
    win["cvr"][gs["cv_rmse_raw"].idxmin()] += 1
    win["cvl"][gs.loc[lg, "cv_rmse_log"].idxmin()] += 1
for key, name in [("aic", "AIC(log)"), ("bic", "BIC(log)"),
                  ("cvr", "CV raw-space"), ("cvl", "CV log-space")]:
    log(f"  {name}: {dict(win[key])}")
pl_lin = sum(1 for gs in G.values()
             if has(gs, ["power_law", "linear"])
             and gs.loc["power_law", "cv_rmse_raw"]
             < gs.loc["linear", "cv_rmse_raw"])
ql_pl = sum(1 for gs in G.values()
            if has(gs, ["quadratic_log", "power_law"])
            and gs.loc["quadratic_log", "cv_rmse_log"]
            < gs.loc["power_law", "cv_rmse_log"])
gam_pl = sum(1 for gs in G.values()
             if has(gs, ["gam", "power_law"])
             and gs.loc["gam", "cv_rmse_log"]
             < gs.loc["power_law", "cv_rmse_log"])
log(f"  power law better than linear on raw CV: {pl_lin}/20")
log(f"  log-quadratic better than power law on log CV: {ql_pl}/20")
log(f"  GAM better than power law on log CV: {gam_pl}/20")
PLR = MC[MC["model"] == "power_law"]
log(f"  power-law R2 {PLR['r2'].min()*100:.1f}-{PLR['r2'].max()*100:.1f}%, "
    f"adjR2 {PLR['adj_r2'].min()*100:.1f}-{PLR['adj_r2'].max()*100:.1f}%")
log(f"  models present: {sorted(set(MC['model']))}")
log(f"  total raw negative predictions {int(MC['neg_raw_preds'].sum())}")

# ================================================================ clinical
log("=" * 70 + "\nCLINICAL (primary)")
log(f"planned (20, cov sex+age): n={int(PC['n'].max())} "
    f"min p={PC['p'].min():.4f} min q={PC['q'].min():.3f} "
    f"sig={(PC['q'] < 0.05).sum()}")
for _, r in PC.sort_values("p").head(5).iterrows():
    log(f"  {r['iad_col']} ~ {r['outcome']}: r={r['r']:+.3f} p={r['p']:.4f} "
        f"q={r['q']:.3f} n={int(r['n'])}")
log(f"exploratory ({len(EX)} tests): min p={EX['p'].min():.4f} "
    f"min q={EX['q'].min():.3f} q-sig={(EX['q'] < 0.05).sum()} "
    f"p-sig(unadj)={int((EX['p'] < 0.05).sum())} n={int(EX['n'].max())}")
log("  roc columns: " + str(list(ROC.columns)))
for _, r in ROC.iterrows():
    log(f"  ROC {r['col']}: AUC={r['auc']:.3f} [{r['ci_lo']:.3f},"
        f"{r['ci_hi']:.3f}] n_high={int(r['n_high'])} "
        f"n_low={int(r['n_low'])}")

# ================================================================ sample
log("=" * 70 + "\nSAMPLE")
log(AGE.to_string(index=False))
PRIM = pd.read_csv(TAB / "subjects_features_primary.csv")
AG = PRIM["age"].astype(float)
log("  exact age>=11: %d; >=13: %d; >=14: %d; >=17: %d; max=%.2f" % (
    int((AG >= 11).sum()), int((AG >= 13).sum()), int((AG >= 14).sum()),
    int((AG >= 17).sum()), float(AG.max())))
log(REL.to_string(index=False))
log(f"  sums: n={int(REL['n'].sum())} pass={int(REL['pass_n'].sum())} "
    f"audited={int(REL['audited_n'].sum())}")
log(INC.to_string(index=False))

# ================================================================ appendix
def tab_block(rows, spec, size, colsep, hdr=1):
    lines = [B + size if colsep is None else
             f"{B}{size} {B}setlength{{{B}tabcolsep}}{{{colsep}}}",
             B + "begin{tabular}{" + spec + "}", B + "toprule"]
    for i, r in enumerate(rows):
        if i == hdr:
            lines.append(B + "midrule")
        if r.startswith(B + "cmidrule"):
            lines.append(r)
        elif r.endswith("\\\\"):
            lines.append(r)
        else:
            lines.append(r + " \\\\")
    lines += [B + "bottomrule", B + "end{tabular}"]
    return "\n".join(lines) + "\n"


def make_float(name, label, caption, blocks, wide=True):
    env = "table*" if wide else "table"
    parts = [B + f"begin{{{env}}}[pos=t]", B + "centering",
             B + "caption{" + caption + "}" + B + "label{tab:" + label + "}",
             ""]
    for blk in blocks:
        rows, spec, size, colsep = blk[:4]
        hdr = blk[4] if len(blk) > 4 else 1
        parts.append(tab_block(rows, spec, size, colsep, hdr))
        parts.append("")
    parts.append(B + f"end{{{env}}}")
    (OUT / f"{name}.tex").write_text("\n".join(parts), encoding="utf-8")
    log(f"wrote {name}.tex")


LBL = lambda r: f"{r['band']} / {MI[r['metric']]}"


def iadlbl(r):
    return f"{BG[r['band']]} & {MI[r['metric']]}"


QL = B + "quad "

# ---- S1a information criteria
s = [f"{B}multicolumn{{2}}{{l}}{{Specification}} & "
     f"{B}multicolumn{{3}}{{c}}{{AIC (log space)}} & "
     f"{B}multicolumn{{3}}{{c}}{{BIC (log space)}} & "
     f"{B}multicolumn{{2}}{{c}}{{AIC (raw space)}} & "
     f"{B}multicolumn{{2}}{{c}}{{BIC (raw space)}}",
     f"{B}cmidrule(lr){{3-5}} {B}cmidrule(lr){{6-8}} "
     f"{B}cmidrule(lr){{9-10}} {B}cmidrule(lr){{11-12}}",
     f"Band & Index & $P$ & $Q2$ & GAM & $P$ & $Q2$ & GAM "
     f"& Linear & Quad. & Linear & Quad."]
def ic(gs, m, col):
    return f"{gs.loc[m, col]:.1f}" if m in gs.index else "--"


for k in sorted(G, key=lambda x: (ORDER[x[0]], MORD[x[1]])):
    gs = G[k]
    s.append(f"{k[0]} & {MI[k[1]]} & " + " & ".join(
        ic(gs, m, "aic") for m in LOGM) + " & " + " & ".join(
        ic(gs, m, "bic") for m in LOGM) + " & " + " & ".join(
        ic(gs, m, "aic") for m in RAWM) + " & " + " & ".join(
        ic(gs, m, "bic") for m in RAWM) + " \\\\")
make_float("s1a_ic", "S1a",
           f"Information criteria for the 20 global fits (primary sample). "
           f"$P$ = power law, $Q2$ = quadratic in ${B}log_{{10}}$(age), "
           f"GAM = penalized cubic spline. AIC and BIC are computed within a "
           f"prediction space and are therefore comparable only inside the "
           f"corresponding block, never between the log-space and raw-space "
           f"blocks.",
           [(s, "llrrrrrrrrrr", "scriptsize", "3pt", 3)])

# ---- S1b cross-validation
s = [f"Band & Index & {B}multicolumn{{5}}{{c}}{{Raw-space CV RMSE}} & "
     f"{B}multicolumn{{3}}{{c}}{{Log-space CV RMSE}}",
     f"{B}cmidrule(lr){{3-7}} {B}cmidrule(lr){{8-10}}",
     " & & Linear & Quad. & $P$ & $Q2$ & GAM & $P$ & $Q2$ & GAM"]
best_raw = {}
for k in sorted(G, key=lambda x: (ORDER[x[0]], MORD[x[1]])):
    gs = G[k]
    ms = [m for m in RAWM + LOGM if m in gs.index]
    vals = [gs.loc[m, "cv_rmse_raw"] for m in ms]
    best_raw[k] = ms[min(range(len(vals)), key=lambda i: vals[i])]
    lg = [m for m in LOGM if m in gs.index]
    s.append(f"{k[0]} & {MI[k[1]]} & " + " & ".join(
        (f"{gs.loc[m, 'cv_rmse_raw']:.4f}" if m in gs.index else "--")
        for m in RAWM + LOGM) + " & "
        + " & ".join(f"{gs.loc[m, 'cv_rmse_log']:.4f}" for m in lg) + " \\\\")
log("  best raw-CV model per fit: " + str(Counter(best_raw.values())))
make_float("s1b_cv", "S1b",
           f"Five-fold cross-validated prediction error for the 20 global "
           f"fits (primary sample; identical folds and seed for every "
           f"model). Raw-space error is comparable across all five "
           f"specifications; log-space error only across the three "
           f"log-space specifications.",
           [(s, "llrrrrrrrr", "scriptsize", "3pt", 3)])

# ---- S2 sensitivity
s = [f"Band / index & $n_{{{RMF}}}$ & ${B}alpha_{{{RMF}}}$ & 95" + B + "% CI & "
     f"$q_{{{RMF}}}$ & $n_{{{RMP}}}$ & ${B}alpha_{{{RMP}}}$ & 95" + B + "% CI & "
     f"$q_{{{RMP}}}$ & ${B}Delta {B}alpha$"]
for _, r in SF.iterrows():
    d = float(r["d_alpha"])
    ds = f"${'-' if d < 0 else '+'}{abs(d):.3f}$"
    s.append(f"{LBL(r)} & {int(r['n_full']):,} & {fnum(r['alpha_full'])} & "
             + fci(r["ci_full_lo"], r["ci_full_hi"]) + " & "
             + fsci(r["q_full"], 2) + " & "
             + f"{int(r['n_primary']):,} & "
             + acell(r["alpha_primary"], r["q_primary"]) + " & "
             + fci(r["ci_prim_lo"], r["ci_prim_hi"]) + " & "
             + fsci(r["q_primary"], 2) + " & " + ds + " \\\\")
make_float("s2_sens", "S2",
           f"Sensitivity analysis: the full audited sample ($n=1{{,}}967$) "
           f"against the primary sample that excludes the ten subjects with "
           f"numerically degenerate networks ($n=1{{,}}957$). Bold face "
           f"marks fits significant at FDR $q<0.05$ in the primary "
           f"analysis.",
           [(s, "llrrrrrrrl", "scriptsize", "3pt")])

# ---- S3 diagnostics
s = ["Band / index & $n$ & Max Cook's $D$ & Obs. with $D>4/n$ & "
     "Max leverage & Resid. skewness & Resid. kurtosis"]
for _, r in DIAG.iterrows():
    s.append(f"{LBL(r)} & {int(r['n']):,} & {r['max_cookd']:.3f} & "
             f"{int(r['n_cookd_gt_4n'])} & {r['max_leverage']:.4f} & "
             f"${r['resid_skew']:+.2f}$ & {r['resid_kurtosis']:.1f} \\\\")
make_float("s3_diag", "S3",
           f"Influence and residual diagnostics for the 20 primary log-log "
           f"power-law fits. No observation exceeded Cook's $D=0.5$, and "
           f"maximum leverage is identical across fits because the "
           f"regressor (${B}log_{{10}}$ of age) is common to all of them. "
           f"Residuals remain right-skewed for the most variable indices, "
           f"so interval estimates rest on the usual asymptotic argument "
           f"rather than on a distributional assumption.",
           [(s, "lrrrrrr", "scriptsize", "4pt")])

# ---- S4 lobe
a = [f"Band / index & Lobe & $n$ & ${B}alpha$ & 95" + B + "% CI & $q$"]
b = list(a)
for _, r in LO.iterrows():
    line = (f"{LBL(r)} & {r['lobe']} & {int(r['n']):,} & "
            + acell(r["alpha"], r["q_fdr"]) + " & "
            + fci(r["alpha_lo"], r["alpha_hi"]) + " & "
            + fsci(r["q_fdr"], 2) + " \\\\")
    (a if r["metric"] == "global_efficiency" else b).append(line)
make_float("s4_lobe", "S4",
           f"All 50 lobe-resolved power-law fits for the primary sample. $q$ "
           f"values are BH-FDR adjusted across the family of 50 fits; bold "
           f"face marks $q<0.05$.",
           [(a, "lclrrr", "scriptsize", "5pt"),
            (b, "lclrrr", "scriptsize", "5pt")])

# ---- S5 sample
a = ["Age band (years) & $n$ & " + B + "% of sample & Mean age & "
     + B + "% female"]
for _, r in AGE.iterrows():
    a.append(f"{str(r['age_band']).replace('-', '--')} & {int(r['n']):,} & "
             f"{r['pct']:.1f} & {r['mean_age']:.1f} & "
             f"{r['pct_female']:.1f} \\\\")
b = ["Data release & $n$ & Age range & Mean age & " + B + "% female & "
     "Audit $n$ & Audit pass & Pass " + B + "%"]
RS = REL.copy()
RS["_k"] = RS["release"].astype(str).str.extract(r"Release(\d+)").astype(int)
for _, r in RS.sort_values("_k").iterrows():
    rel = str(r["release"]).replace("HBN-Release", "Release ").replace(
        "-ds", ", ds")
    b.append(f"{rel} & {int(r['n']):,} & {r['age_min']:.1f}--"
             f"{r['age_max']:.1f} & {r['age_mean']:.1f} & "
             f"{r['pct_female']:.1f} & {int(r['audited_n']):,} & "
             f"{int(r['pass_n']):,} & {r['pass_pct']:.1f} \\\\")
make_float("s5_sample", "S5",
           f"Sample representativeness of the primary analysis. Upper "
           f"block: age and sex composition within each age band. Lower "
           f"block: per-release composition and audit-QC pass rate. "
           f"Audit-pass counts include the ten subjects removed later for "
           f"numerically degenerate networks, so they can slightly exceed "
           f"the analysed $n$.",
           [(a, "lrrrr", "footnotesize", "6pt"),
            (b, "lrrrrrrr", "footnotesize", "4pt")])

# ---- S5b included vs excluded
INCNM = {"age_mean": "Mean age (years, SD)",
         "age_median": "Median age (years)",
         "cohen_d_age": "Age difference (Cohen $d$)",
         "release_composition": "Release composition"}
TESTNM = {"Welch t": "Welch " + B + "textit{t}-test",
          "Mann-Whitney U": "Mann-Whitney $U$",
          "d": "$d$",
          "chi2": "$" + B + "chi^{2}$"}


def fbig(x):
    x = float(x)
    if abs(x) >= 1000.0:
        e = fl(abs(x))
        return f"${x / 10.0 ** e:.2f}{B}times10^{{{e}}}$"
    return f"${x:.2f}$"


def fmt_inc(v):
    v = str(v).strip()
    if v == "" or v.lower() == "nan":
        return "--"
    if "+-" in v:
        p1, p2 = v.split("+-")
        return f"${float(p1):.2f} {B}pm {float(p2):.2f}$"
    try:
        x = float(v)
    except ValueError:
        return TESTNM.get(v, v)
    if abs(x) >= 1000:
        return fbig(x)
    return f"${x:.2f}$"


a = ["Measure & Included & Excluded & Test & Statistic & $p$"]
for _, r in INC.iterrows():
    m = str(r["measure"])
    pv = "--" if pd.isna(r["p"]) else fsci(r["p"], 3)
    name = INCNM.get(m, m)
    cell_i = fmt_inc(r["included"]) if "age" in m or "release" in m else "--"
    cell_e = fmt_inc(r["excluded"]) if "age" in m or "release" in m else "--"
    if m in ("cohen_d_age", "release_composition"):
        cell_i = cell_e = "--"
    a.append(f"{name} & {cell_i} & {cell_e} & {fmt_inc(r['test'])} & "
             f"{fmt_inc(r['stat'])} & {pv} \\\\")
make_float("s5b_incl", "S5b",
           f"Participants retained for the primary analysis compared with "
           f"those excluded by the audit-based QC stage. Sex and CBCL "
           f"scores are not available for excluded participants, so "
           f"comparisons are limited to age and release composition.",
           [(a, "llllll", "footnotesize", "5pt")])

# ---- S6 clinical
a = [f"Band & Index & CBCL outcome & $n$ & Partial ${B}rho$ & $p$ & "
     f"$q$ (20 tests)"]
for _, r in PC.iterrows():
    a.append(f"{iadlbl(r)} & {ONM[r['outcome']]} & {int(r['n']):,} & "
             f"${r['r']:+.3f}$ & {r['p']:.4f} & {r['q']:.3f} \\\\")
b = ["Band & Index & AUC & 95" + B + "% CI & Risk $n$ & Non-risk $n$"]
for _, r in ROC.iterrows():
    pc = str(r["col"]).split("__")
    b.append(f"{BG[pc[0]]} & {MI[pc[1]]} & {r['auc']:.3f} & "
             + fci(r["ci_lo"], r["ci_hi"])
             + f" & {int(r['n_high'])} & {int(r['n_low']):,} \\\\")
make_float("s6_clinical", "S6",
           f"Clinical associations in the primary sample. Upper block: "
           f"partial Spearman associations between the band-level "
           f"individual allometric deviation (IAD) of global efficiency and "
           f"the four CBCL problem scores, adjusted for sex and age, with "
           f"BH-FDR over the 20 planned tests. Lower block: discrimination "
           f"of externalizing risk ($>+1$ SD, $n=211$) by the same IADs.",
           [(a, "lllrrrr", "footnotesize", "5pt"),
            (b, "llrrrr", "footnotesize", "5pt")])

# ---- S6b exploratory
a = [f"Band & Index & CBCL outcome & $n$ & ${B}rho$ & $p$ & "
     f"$q$ (80 tests)"]
for _, r in EX.sort_values("p").iterrows():
    a.append(f"{iadlbl(r)} & {ONM[r['outcome']]} & {int(r['n']):,} & "
             f"${r['r']:+.3f}$ & {r['p']:.4f} & {r['q']:.3f} \\\\")
h = (len(a) - 1) // 2 + 1
make_float("s6b_explor", "S6b",
           f"Exploratory extension of the clinical analysis to all 20 "
           f"band-by-index IADs and the four CBCL problem "
           f"scores (80 tests, adjusted for sex and age; BH-FDR over the "
           f"full family). No association survived correction. Blocks are "
           f"ordered by unadjusted $p$.",
           [(a[:h], "llllrrr", "scriptsize", "3pt"),
            ([a[0]] + a[h:], "llllrrr", "scriptsize", "3pt")])

# ---- S7 interaction
a = ["Band / index & $n$ & $F$ & $p$ & $q$ (10 tests)"]
for _, r in OM.iterrows():
    a.append(f"{LBL(r)} & {int(r['n']):,} & {r['F']:.2f} & "
             f"{fsci(r['p'], 3)} & {fsci(r['p_fdr'], 3)} \\\\")
p1 = [f"Band / index & Contrast & ${B}Delta {B}alpha$ & $p$ & "
      "$q$ (100 tests)"]
for _, r in PW.sort_values("p_fdr").iterrows():
    p1.append(f"{LBL(r)} & {r['comparison']} & ${r['diff']:+.3f}$ & "
              f"{fsci(r['p_raw'], 3)} & {fsci(r['p_fdr'], 3)} \\\\")
hp = (len(p1) - 1) // 2 + 1
make_float("s7_inter", "S7",
           f"Formal tests of regional differences in developmental slope "
           f"(primary sample). Upper block: nested-model comparison of a "
           f"lobewise-slope model against a common-slope model for each "
           f"band and index, BH-FDR over 10 tests. Remaining blocks: all "
           f"100 pairwise lobe contrasts (HC3 covariance), ordered by "
           f"BH-FDR adjusted $p$.",
           [(a, "lrrrr", "footnotesize", "5pt"),
            (p1[:hp], "lcccc", "scriptsize", "3pt"),
            ([p1[0]] + p1[hp:], "lcccc", "scriptsize", "3pt")])

(OUT / "revision0922_numbers.txt").write_text("\n".join(TXT) + "\n",
                                              encoding="utf-8")
print("DONE. see revision0922_numbers.txt")
