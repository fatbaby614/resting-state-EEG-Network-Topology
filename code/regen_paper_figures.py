# -*- coding: utf-8 -*-
"""
regen_paper_figures.py — 论文投稿图统一重绘（Python/matplotlib 后端）

契约：
1. 所有图按 *最终排版物理尺寸* 创建（单栏 3.3 in / 通栏 6.8 in），
   保证图内文字排版后 >= 7.5 pt，避免缩放导致的字号缩水。
2. 数据全部来自 results/tables/ 真实结果表，不重新拟合管线。
3. 脑地形图放弃 128 电极插值（旧图含电极黑点、伪插值、无标签色标），
   改为按 lobe 级真实 α 值绘制五脑区示意地形图。
4. 导出 600 dpi PNG（白底），覆盖 paper/figures/ 旧文件。

输出（全部为 primary / 剔除近零退化网络样本，n = 1,957；见 HBN4修改0922 意见 2、8）：
  paper/figures/fig1_concept.png              （示意图，模拟数据）
  paper/figures/fig3_loglog_grid_primary.png
  paper/figures/fig4_trajectory_theta.png
  paper/figures/fig12_roc_primary.png
  paper/figures/fig_density_sweep.png
  paper/figures/fig_lobe_grid_primary.png      （5 频段 x 2 指标 合成网格）
  paper/figures/fig_lobe_pair_gamma.png       （gamma 双面板）
  paper/figures/fig_lobe_pair_alpha.png       （alpha 双面板）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Circle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# 全局样式：论文字号契约（最小 7.5 pt）
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ROOT = Path(r"d:\TanHuangWork\HBN4")
TAB = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 600

# 统一频带配色（全论文一致）
BAND_COLORS = {
    "delta": "#C44E52",
    "theta": "#DD8452",
    "alpha": "#55A868",
    "beta":  "#4C72B0",
    "gamma": "#8172B3",
}
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_GREEK = {"delta": r"$\delta$", "theta": r"$\theta$", "alpha": r"$\alpha$",
              "beta": r"$\beta$", "gamma": r"$\gamma$"}


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white",
                pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] {p.name}")


# ===========================================================================
# Fig 1 — 概念示意图（跨栏 figure*，两行排布：A/B/C + D/E；模拟数据标注）
# ===========================================================================
def fig1_concept():
    """Figure 1 — 五面板研究框架图（per revision #20；matplotlib 拼装）.

    两行排布：第一行 A/B/C，第二行 D/E（居中）。
    A cohort / B network construction / C developmental scaling /
    D regional analysis / E individual deviation.
    """
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 6, wspace=0.85, hspace=0.30,
                          left=0.06, right=0.985, top=0.905, bottom=0.065)
    box_kw = dict(boxstyle="round,pad=0.30", fc="#EFF3F8", ec="#4C72B0", lw=0.8)
    flow_kw = dict(arrowstyle="-|>", color="0.30", lw=0.9, shrinkA=0, shrinkB=0)
    note_kw = dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.5,
                   alpha=0.92)

    def _flow(ax, title, items, fs=8.5):
        ax.set_title(title, fontsize=9.5, pad=5, loc="left")
        n = len(items)
        ys = np.linspace(0.90, 0.06, n)
        gap = (ys[0] - ys[-1]) / (n - 1) if n > 1 else 0.2
        pad = min(0.075, 0.42 * gap)
        for i, (t, yy) in enumerate(zip(items, ys)):
            ax.text(0.5, yy, t, ha="center", va="center", fontsize=fs,
                    transform=ax.transAxes, bbox=box_kw)
            if i:
                ax.annotate("", xy=(0.5, ys[i] + pad),
                            xytext=(0.5, ys[i - 1] - pad),
                            xycoords="axes fraction",
                            textcoords="axes fraction", arrowprops=flow_kw)
        ax.axis("off")

    # Panel A — cohort（第一行左）
    axA = fig.add_subplot(gs[0, 0:2])
    _flow(axA, "A  Developmental cohort",
          ["5\u201321 years", "1,967 participants",
           "128-ch resting-state\nEEG"], fs=8.5)

    # Panel B — pipeline（第一行中；长文本换行避免裁切）
    axB = fig.add_subplot(gs[0, 2:4])
    _flow(axB, "B  Network construction",
          ["EEG", "Frequency decomposition",
           "Covariance\n(Ledoit\u2013Wolf)",
           "Functional connectivity", "Graph topology"], fs=8.5)

    # Panel C — scaling（第一行右；模拟数据示意）
    axC = fig.add_subplot(gs[0, 4:6])
    rng = np.random.default_rng(42)
    n = 800
    ages = rng.uniform(5, 21, size=n)
    log_age = np.log(ages)
    slope, intercept = -1.06, np.log(3.08)
    log_y = slope * log_age + intercept + rng.normal(0, 1.30, size=n)
    axC.scatter(ages, np.exp(log_y), s=5, alpha=0.20, color="#7F7F7F",
                edgecolors="none", rasterized=True)
    res = stats.linregress(log_age, log_y)
    xx = np.linspace(log_age.min(), log_age.max(), 100)
    axC.plot(np.exp(xx), np.exp(res.intercept + res.slope * xx),
             color="#C44E52", lw=1.7)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlim(4.4, 23)
    axC.set_ylim(2e-3, 6e1)
    axC.set_xlabel("Age (years)", fontsize=8.5, labelpad=2)
    axC.set_ylabel(r"Index $y$", fontsize=8.5, labelpad=1)
    axC.tick_params(labelsize=8, pad=2)
    axC.grid(True, which="major", ls="--", lw=0.35, alpha=0.35)
    axC.text(0.05, 0.96, r"$y = k \cdot \mathrm{age}^{\alpha}$"
             + ",   " + rf"$\alpha = {res.slope:.2f}$",
             transform=axC.transAxes, fontsize=8.5, va="top", color="#C44E52",
             bbox=note_kw)
    axC.set_title("C  Developmental scaling", fontsize=9.5, pad=5, loc="left")

    # Panel D — regional（第二行左中）
    axD = fig.add_subplot(gs[1, 1:3])
    _flow(axD, "D  Regional analysis",
          ["Global network", "Frontal", "Central", "Temporal", "Parietal",
           "Occipital"], fs=8.5)

    # Panel E — individual deviation（第二行右中；示意）
    axE = fig.add_subplot(gs[1, 3:5])
    demo_ages = np.array([6.5, 9.5, 14.0, 19.0])
    yhat_demo = np.exp(res.intercept + res.slope * np.log(demo_ages))
    demo_y = yhat_demo * np.array([2.4, 0.42, 2.1, 0.50])
    axE.plot(np.exp(xx), np.exp(res.intercept + res.slope * xx),
             color="#4C72B0", lw=1.7)
    axE.scatter(demo_ages, demo_y, s=16, color="#C44E52", zorder=5)
    for xa, ya, yh in zip(demo_ages, demo_y, yhat_demo):
        axE.annotate("", xy=(xa, yh), xytext=(xa, ya),
                     arrowprops=dict(arrowstyle="<->", color="#C44E52",
                                     lw=0.9, shrinkA=0, shrinkB=0))
    axE.set_xscale("log")
    axE.set_yscale("log")
    axE.set_xlim(4.4, 23)
    axE.set_ylim(0.04, 6.0)
    axE.set_xlabel("Age (years)", fontsize=8.5, labelpad=2)
    axE.set_ylabel(r"Index $y$", fontsize=8.5, labelpad=1)
    axE.tick_params(labelsize=8, pad=2)
    axE.grid(True, which="major", ls="--", lw=0.35, alpha=0.35)
    axE.text(0.05, 0.96, r"IAD $= \log y - \widehat{\log y}$"
             ";  IAD\u2013CBCL association: null",
             transform=axE.transAxes, fontsize=8.5, va="top", bbox=note_kw)
    axE.set_title("E  Individual deviation", fontsize=9.5, pad=5, loc="left")

    _save(fig, "fig1_concept.png")


# ===========================================================================
# Fig 3 — 2x2 log-log 网格（通栏 6.8 in）
# ===========================================================================
def _filter_iqr(x, y, k=4.0, y_floor=None):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (y >= lo) & (y <= hi)
    if y_floor is not None:
        mask &= y >= y_floor
    return x[mask], y[mask]


def _loglog_panel(ax, df, band, metric, ylabel_on):
    col = f"{band}__{metric}"
    sub = df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub["age"] > 0) & (sub[col] > 0)]
    ages, yv = sub["age"].to_numpy(), sub[col].to_numpy()

    floor = 1e-6 if metric == "global_efficiency" else 1e-4
    af, yf = _filter_iqr(ages, yv, 4.0, floor)

    res = stats.linregress(np.log(ages), np.log(yv))
    xl = np.linspace(np.log(ages).min(), np.log(ages).max(), 200)
    yhat = res.intercept + res.slope * xl
    resid = np.log(yv) - (res.intercept + res.slope * np.log(ages))
    nn = len(ages)
    se = np.sqrt(np.sum(resid ** 2) / (nn - 2)) * np.sqrt(
        1.0 / nn + (xl - np.log(ages).mean()) ** 2
        / np.sum((np.log(ages) - np.log(ages).mean()) ** 2))
    tcrit = stats.t.ppf(0.975, nn - 2)

    ax.scatter(af, yf, s=3.5, alpha=0.18, color="#7F7F7F",
               edgecolors="none", rasterized=True, label=f"Subjects (n = {len(af)})")
    c = BAND_COLORS[band]
    ax.plot(np.exp(xl), np.exp(yhat), color=c, lw=1.8,
            label=rf"$\alpha={res.slope:.3f}$, $R^2={res.rvalue**2:.3f}$")
    ax.fill_between(np.exp(xl), np.exp(yhat - tcrit * se), np.exp(yhat + tcrit * se),
                    color=c, alpha=0.20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = np.percentile(yf, 0.1), np.percentile(yf, 99.9)
    pad = (np.log10(hi) - np.log10(lo)) * 0.3
    ax.set_ylim(10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad))
    ax.set_xlabel("Age (years)")
    if ylabel_on:
        ax.set_ylabel(r"Global efficiency $E_{\rm glob}$" if metric == "global_efficiency"
                      else r"Small-worldness $\sigma$")
    mlabel = ("Global efficiency" if metric == "global_efficiency"
              else "Small-worldness")
    ax.set_title(rf"{BAND_GREEK[band]}-band | {mlabel}", fontsize=8.5)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.35)
    # 图例放左下角（负斜率数据带下方空区）：右上角在高龄端散点密集，会遮挡
    ax.legend(loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="0.7", fontsize=7.5)


def fig3_grid():
    df = pd.read_csv(TAB / "subjects_features_primary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.05))
    panels = [("delta", "global_efficiency"), ("theta", "global_efficiency"),
              ("delta", "small_worldness"), ("theta", "small_worldness")]
    for (band, metric), ax in zip(panels, axes.ravel()):
        _loglog_panel(ax, df, band, metric, ylabel_on=(ax in axes[:, 0]))
    fig.tight_layout(w_pad=1.4, h_pad=1.6)
    _save(fig, "fig3_loglog_grid_primary.png")


# ===========================================================================
# Fig 4 — theta 全局效率的连续发育轨迹（单栏 3.3 in；per revision #22）
#   individual points + age-bin mean ± 95% CI + power-law prediction + CI
# ===========================================================================
def fig4_trajectory():
    df = pd.read_csv(TAB / "subjects_features_primary.csv")
    col = "theta__global_efficiency"
    sub = df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub["age"] > 0) & (sub[col] > 0)]
    ages, yv = sub["age"].to_numpy(), sub[col].to_numpy()

    res = stats.linregress(np.log(ages), np.log(yv))
    xx = np.linspace(np.log(ages).min(), np.log(ages).max(), 200)
    yhat = res.intercept + res.slope * xx
    resid = np.log(yv) - (res.intercept + res.slope * np.log(ages))
    nn = len(ages)
    se = np.sqrt(np.sum(resid ** 2) / (nn - 2)) * np.sqrt(
        1.0 / nn + (xx - np.log(ages).mean()) ** 2
        / np.sum((np.log(ages) - np.log(ages).mean()) ** 2))
    tcrit = stats.t.ppf(0.975, nn - 2)

    # 高度 2.75 -> 3.05：为绘图区下方（轴外）图例预留空间（宽度不变，字号契约不受影响）
    fig, ax = plt.subplots(figsize=(3.3, 3.05))
    ax.scatter(ages, yv, s=3.5, alpha=0.16, color="#7F7F7F",
               edgecolors="none", rasterized=True,
               label=f"Individual subjects (n = {nn})")
    ax.fill_between(np.exp(xx), np.exp(yhat - tcrit * se),
                    np.exp(yhat + tcrit * se), color="#C44E52", alpha=0.18,
                    label="Power-law fit (95% CI)")
    ax.plot(np.exp(xx), np.exp(yhat), color="#C44E52", lw=1.8,
            label=rf"$\alpha = {res.slope:.2f}$")
    # age-bin means（2 岁一箱）± 95% CI
    edges = np.arange(5, 23, 2)
    centers, means, cis = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = yv[(ages >= lo) & (ages < hi)]
        if len(v) < 10:
            continue
        m = v.mean()
        sem = v.std(ddof=1) / np.sqrt(len(v))
        centers.append(0.5 * (lo + hi))
        means.append(m)
        cis.append(stats.t.ppf(0.975, len(v) - 1) * sem)
    centers, means, cis = map(np.asarray, (centers, means, cis))
    ax.errorbar(centers, means, yerr=cis, fmt="o", ms=4.5, color="#1F5E8C",
                ecolor="#1F5E8C", elinewidth=1.0, capsize=2.2, lw=1.0,
                label="Age-bin mean $\pm$ 95% CI", zorder=6)
    ax.set_yscale("log")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$E_{\rm glob}$ (theta band)")
    lo, hi = np.percentile(yv, 0.1), np.percentile(yv, 99.9)
    pad = (np.log10(hi) - np.log10(lo)) * 0.25
    ax.set_ylim(10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad))
    ax.set_xlim(4.5, 21.5)
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.4)
    # 图例移到绘图区外底部，横排 2 列两行（与 Fig 12 同一约定）：数据带被 6 个
    # 近零离群值（E_glob ~1e-16）压成轴顶的一条窄带，轴内任何位置都会遮挡
    # 拟合线与右侧 age-bin 均值点
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              frameon=False, fontsize=7.5, handlelength=1.5,
              columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    _save(fig, "fig4_trajectory_theta.png")


# ===========================================================================
# Fig 12 — ROC（单栏 3.3 in）
# ===========================================================================
def fig12_roc():
    # per-subject IAD for the primary sample, written by
    # gen_iad_primary_0922.py and cross-checked against roc_primary.csv
    iad = pd.read_csv(TAB / "iad_primary.csv")
    # 高度 2.85 -> 3.15：为轴外底部图例预留空间（宽度不变，字号契约不受影响）
    fig, ax = plt.subplots(figsize=(3.3, 3.15))
    for band in BANDS:
        col = f"{band}__global_efficiency__iad"
        d = iad[[col, "externalizing"]].replace([np.inf, -np.inf], np.nan).dropna()
        y = (d["externalizing"] > 1.0).astype(int)
        if y.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y, d[col])
        auc = roc_auc_score(y, d[col])
        ax.plot(fpr, tpr, lw=1.4, color=BAND_COLORS[band],
                label=f"{band} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(ls="--", lw=0.35, alpha=0.35)
    # 图例移到绘图区外底部，横排 3 列两行：彻底避开右下角的曲线汇聚区
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, fontsize=7.5, handlelength=1.2,
              columnspacing=0.4, handletextpad=0.3)
    fig.tight_layout()
    _save(fig, "fig12_roc_primary.png")


# ===========================================================================
# Fig 5 — 密度阈值敏感性（通栏 6.8 in）
# ===========================================================================
def fig_density():
    tae = pd.read_csv(TAB / "tae_density_sweep_primary.csv")
    metrics = [("global_efficiency", "Global efficiency $E_{\\rm glob}$"),
               ("small_worldness", "Small-worldness $\\sigma$")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), sharey=False)
    for ax, (metric, mlabel) in zip(axes, metrics):
        lo, hi = np.inf, -np.inf
        for band in ("delta", "theta"):
            d = tae[(tae["metric"] == metric) & (tae["band"] == band)].sort_values("density")
            x = (d["density"] * 100).to_numpy()
            a = d["alpha"].to_numpy()
            r2, nn = d["r2"].to_numpy(), d["n"].to_numpy()
            tstat = np.sqrt(r2 * (nn - 2) / np.clip(1 - r2, 1e-12, None))
            se = np.abs(a) / np.where(np.isfinite(tstat) & (tstat > 0), tstat, 1.0)
            tcrit = stats.t.ppf(0.975, nn - 2)
            lo = min(lo, float((a - tcrit * se).min()))
            hi = max(hi, float((a + tcrit * se).max()))
            ax.fill_between(x, a - tcrit * se, a + tcrit * se,
                            color=BAND_COLORS[band], alpha=0.13, linewidth=0)
            ax.plot(x, a, "o-", ms=5, lw=1.6, mfc=BAND_COLORS[band],
                    color=BAND_COLORS[band], label=band)
        ax.axhline(0, ls="--", c="0.35", lw=0.9)
        ax.set_xlabel("Network density (%)")
        ax.set_ylabel(r"Allometric exponent $\alpha$")
        ax.set_title(mlabel, fontsize=8.5)
        ax.set_xticks(np.arange(10, 31, 5))
        ax.grid(True, ls="--", lw=0.4, alpha=0.4)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92,
                  edgecolor="0.7", fontsize=7.5)
        # 纵轴范围随数据变化：换用 primary 样本后不再沿用 full 样本的手调上限
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, max(hi + pad, 0.05))
    fig.tight_layout(w_pad=1.6)
    _save(fig, "fig_density_sweep.png")


# ===========================================================================
# 脑区示意地形图（五脑区，真实 lobe 级 α；无电极点/插值）
# ===========================================================================
# 单位头坐标（鼻在 +y），多边形在圆内，最后按单位圆裁剪
LOBES = {
    "frontal": [(-0.66, 0.40), (-0.42, 0.70), (0.0, 0.80), (0.42, 0.70),
                (0.66, 0.40), (0.58, 0.14), (0.30, 0.10), (-0.30, 0.10),
                (-0.58, 0.14)],
    "central": [(-0.58, 0.14), (-0.30, 0.10), (0.30, 0.10), (0.58, 0.14),
                (0.60, -0.10), (0.30, -0.16), (-0.30, -0.16), (-0.60, -0.10)],
    "temporal_l": [(-0.66, 0.28), (-0.58, 0.14), (-0.60, -0.10),
                   (-0.62, -0.32), (-0.80, -0.28), (-0.90, -0.02), (-0.80, 0.20)],
    "temporal_r": [(0.66, 0.28), (0.58, 0.14), (0.60, -0.10),
                   (0.62, -0.32), (0.80, -0.28), (0.90, -0.02), (0.80, 0.20)],
    "parietal": [(-0.60, -0.10), (-0.30, -0.16), (0.30, -0.16), (0.60, -0.10),
                 (0.52, -0.50), (0.28, -0.56), (-0.28, -0.56), (-0.52, -0.50)],
    "occipital": [(-0.52, -0.50), (-0.28, -0.56), (0.28, -0.56), (0.52, -0.50),
                  (0.42, -0.74), (0.15, -0.80), (-0.15, -0.80), (-0.42, -0.74)],
}
CENTROIDS = {"frontal": (0.0, 0.36), "central": (0.0, -0.03),
             "temporal_l": (-0.74, -0.04), "temporal_r": (0.74, -0.04),
             "parietal": (0.0, -0.35), "occipital": (0.0, -0.66)}
# 结果表里使用的五个脑区键（左右颞叶在表中合并为 temporal）
LOBES_KEYS = ("frontal", "central", "temporal", "parietal", "occipital")

CMAP = plt.get_cmap("RdBu_r")
# vmax 由 0.1 放宽到 0.2：primary 样本里最强的正斜率（beta 带额叶 sigma，
# +0.109）刚好越过原上限，会被裁成同一种颜色
NORM = mcolors.TwoSlopeNorm(vmin=-0.9, vcenter=0.0, vmax=0.2)


def _text_color(rgb):
    """深底白字、浅底黑字"""
    r, g, b = rgb[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "black"


def draw_lobe_head(ax, alpha_by_lobe, sig_by_lobe):
    head = Circle((0, 0), 1.0, fc="0.96", ec="none", zorder=0)
    ax.add_patch(head)
    for name, verts in LOBES.items():
        lobe = "temporal" if name.startswith("temporal") else name
        a = alpha_by_lobe[lobe]
        poly = Polygon(verts, closed=True, fc=CMAP(NORM(a)),
                       ec="white", lw=0.7, zorder=2)
        ax.add_patch(poly)
        poly.set_clip_path(head)
    ax.add_patch(Circle((0, 0), 1.0, fc="none", ec="0.25", lw=1.1, zorder=4))
    # 鼻
    ax.add_patch(Polygon([(-0.11, 0.95), (0.11, 0.95), (0.0, 1.10)],
                         closed=True, fc="0.96", ec="0.25", lw=1.0, zorder=4))
    for name, (x, y) in CENTROIDS.items():
        lobe = "temporal" if name.startswith("temporal") else name
        a = alpha_by_lobe[lobe]
        rot = 90 if name.startswith("temporal") else 0
        ax.text(x, y, f"{a:.2f}", ha="center", va="center",
                fontsize=7.5, zorder=5, rotation=rot,
                fontweight="bold" if sig_by_lobe.get(lobe, False) else "normal",
                color=_text_color(CMAP(NORM(a))))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.02, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def _lope_tables(tae):
    """返回 {(band,metric): (alpha dict, sig dict)}

    列名兼容：primary 表把 FDR 校正后的 q 记作 ``q_fdr``（与
    gen_tables_0922.py 读取的 tae_lobe_allometry_primary.csv 一致），旧表记作
    ``p_fdr``；脑区名统一转小写以匹配 LOBES/CENTROIDS 的键。
    """
    pcol = "q_fdr" if "q_fdr" in tae.columns else "p_fdr"
    out = {}
    for band in BANDS:
        for metric in ("global_efficiency", "small_worldness"):
            d = tae[(tae["band"] == band) & (tae["metric"] == metric)].copy()
            d["_lobe"] = d["lobe"].astype(str).str.lower()
            out[(band, metric)] = (
                dict(zip(d["_lobe"], d["alpha"])),
                dict(zip(d["_lobe"], d[pcol] < 0.05)),
            )
    return out


def fig_lobe_grid(tables):
    fig = plt.figure(figsize=(6.8, 3.25))
    gs = fig.add_gridspec(2, 6, width_ratios=[0.10, 1, 1, 1, 1, 1],
                          hspace=0.28, wspace=0.06,
                          left=0.005, right=0.995, top=0.90, bottom=0.16)
    for r, metric in enumerate(("global_efficiency", "small_worldness")):
        la = fig.add_subplot(gs[r, 0])
        la.axis("off")
        la.text(0.5, 0.5,
                "Global efficiency $E_{\\rm glob}$" if metric == "global_efficiency"
                else "Small-worldness $\\sigma$",
                rotation=90, ha="center", va="center", fontsize=8)
        for c, band in enumerate(BANDS):
            ax = fig.add_subplot(gs[r, c + 1])
            a, s = tables[(band, metric)]
            draw_lobe_head(ax, a, s)
            if r == 0:
                ax.set_title(BAND_GREEK[band], fontsize=9, pad=2)
    cax = fig.add_axes([0.42, 0.055, 0.16, 0.035])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                      ticks=[-0.8, -0.4, 0.0])
    cb.set_label(r"Allometric exponent $\alpha$ (bold: FDR $q<0.05$)",
                 fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_linewidth(0.7)
    _save(fig, "fig_lobe_grid_primary.png")


def fig_lobe_pair(tables, band, name):
    fig = plt.figure(figsize=(3.3, 1.75))
    gs = fig.add_gridspec(1, 2, wspace=0.05, top=0.86, bottom=0.14,
                          left=0.01, right=0.99)
    for c, metric in enumerate(("global_efficiency", "small_worldness")):
        ax = fig.add_subplot(gs[0, c])
        a, s = tables[(band, metric)]
        draw_lobe_head(ax, a, s)
        ax.set_title(r"$E_{\rm glob}$" if metric == "global_efficiency"
                     else r"$\sigma$", fontsize=8.5, pad=2)
    cax = fig.add_axes([0.36, 0.06, 0.28, 0.055])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                      ticks=[-0.8, -0.4, 0.0])
    cb.set_label(r"$\alpha$", fontsize=7.5, labelpad=1)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_linewidth(0.7)
    _save(fig, name)


def main():
    fig1_concept()
    fig3_grid()
    fig4_trajectory()
    fig12_roc()
    fig_density()
    # lobe 级 α 与 Table 4 / Table S4 同源：primary 样本的 50 个拟合
    tae = pd.read_csv(TAB / "tae_lobe_allometry_primary.csv")
    have = set(zip(tae["band"], tae["metric"], tae["lobe"].astype(str).str.lower()))
    need = [(b, m, lb) for b in BANDS
            for m in ("global_efficiency", "small_worldness") for lb in LOBES_KEYS]
    missing = [k for k in need if k not in have]
    if missing:
        raise SystemExit(f"[abort] tae_lobe_allometry_primary.csv missing {missing}")
    lo_a, hi_a = float(tae["alpha"].min()), float(tae["alpha"].max())
    if not (NORM.vmin <= lo_a and hi_a <= NORM.vmax):
        raise SystemExit(f"[abort] alpha range [{lo_a:.3f}, {hi_a:.3f}] exceeds "
                         f"color scale [{NORM.vmin}, {NORM.vmax}]")
    print(f"[lobe] alpha range {lo_a:.3f} .. {hi_a:.3f} (color scale ok)")
    tables = _lope_tables(tae)
    fig_lobe_grid(tables)
    fig_lobe_pair(tables, "gamma", "fig_lobe_pair_gamma.png")
    fig_lobe_pair(tables, "alpha", "fig_lobe_pair_alpha.png")
    print("ALL FIGURES REGENERATED")


if __name__ == "__main__":
    main()
