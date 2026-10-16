"""
visualize.py — 图表生成（对应课题方案 Figure 3/4/6/9/12）
- 图3: 全局效率 log-log 散点 + 幂律拟合线
- 图4: 模块化指数随年龄组的小提琴图
- 图6: 频带特异性 TAE 雷达图
- 图9: 小世界属性随年龄曲线
- 图12: IAD 判别高风险组的 ROC 曲线
"""

import logging

import matplotlib

matplotlib.use("Agg")  # 无界面后端，服务器可用
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import allometry as allomod
from .utils import FIGURES_ROOT

logger = logging.getLogger(__name__)

# 中文字体配置（Windows 微软雅黑 / Linux Noto）
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                       "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:  # noqa: BLE001
    pass


def _fig(name, tag):
    fig = plt.figure(figsize=(6, 4.5), dpi=150)
    fname = FIGURES_ROOT / f"{name}_{tag}.png"
    return fig, fname


def fig3_loglog(features, band_names, tae_df, tag):
    """图3: 各频带全局效率 log-log 拟合"""
    for band in band_names:
        col = f"{band}__global_efficiency"
        if col not in features.columns:
            continue
        sub = features[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3:
            continue
        sub = sub[(sub["age"] > 0) & (sub[col] > 0)]
        if len(sub) < 3:
            continue
        fig, fname = _fig(f"fig3_loglog_{band}", tag)
        lx, ly = np.log(sub["age"]), np.log(sub[col])
        ax = fig.add_subplot(111)
        ax.scatter(lx, ly, s=8, alpha=0.4, c="#4472C4")
        res = stats.linregress(lx, ly)
        xs = np.linspace(lx.min(), lx.max(), 100)
        ax.plot(xs, res.intercept + res.slope * xs, "r-", lw=2)
        ax.set_xlabel("log(Age)"); ax.set_ylabel("log(Global Efficiency)")
        ax.set_title(f"Band: {band}   alpha={res.slope:.3f}, R^2={res.rvalue**2:.3f}")
        fig.tight_layout(); fig.savefig(fname); plt.close(fig)
        logger.info("图3保存: %s", fname)


def fig4_violin(features, band_names, age_edges, age_labels, tag):
    """图4: 模块化指数随年龄组小提琴图"""
    for band in band_names:
        col = f"{band}__modularity"
        if col not in features.columns:
            continue
        rows = allomod.age_group_stats(features, band, "modularity",
                                       age_edges, age_labels)
        if not rows:
            continue
        fig, fname = _fig(f"fig4_violin_{band}", tag)
        ax = fig.add_subplot(111)
        positions, data, labels = [], [], []
        for i, r in enumerate(rows):
            if len(r["values"]) == 0:
                continue
            positions.append(i)
            data.append(r["values"])
            labels.append(r["age_group"])
        if not data:
            plt.close(fig)
            continue
        parts = ax.violinplot(data, positions=positions, showmeans=True,
                              showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Modularity Q"); ax.set_title(f"Band: {band}")
        fig.tight_layout(); fig.savefig(fname); plt.close(fig)
        logger.info("图4保存: %s", fname)


def fig6_radar(tae_df, tag):
    """图6: 各频带全局效率 TAE 雷达图"""
    sub = tae_df[tae_df["metric"] == "global_efficiency"]
    if len(sub) < 3:
        return
    sub = sub.sort_values("band")
    bands = sub["band"].tolist()
    vals = sub["alpha"].to_numpy()
    fig, fname = _fig("fig6_radar", tag)
    angles = np.linspace(0, 2 * np.pi, len(bands), endpoint=False).tolist()
    # 雷达图中心化处理：以 0 为基准偏移，使负值也能正常显示
    v_min = min(vals.min(), 0.0)
    offset = -v_min + 0.01 if v_min < 0 else 0.0
    vals_plot = vals + offset
    vals_c = np.concatenate([vals_plot, [vals_plot[0]]])
    angles_c = angles + [angles[0]]
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_c, vals_c, "o-", lw=2, c="#C00000")
    ax.fill(angles_c, vals_c, alpha=0.25, c="#C00000")
    ax.set_xticks(angles); ax.set_xticklabels(bands)
    ax.set_yticklabels([])
    ax.set_title("Global Efficiency Allometric Exponents by Band\n"
                 f"(centered at {offset:.2f})")
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    logger.info("图6保存: %s", fname)


def fig9_smallworld(features, band_names, tag):
    """图9: 小世界属性随年龄曲线（均值+误差带）"""
    for band in band_names:
        col = f"{band}__small_worldness"
        if col not in features.columns:
            continue
        sub = features[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 10:
            continue
        fig, fname = _fig(f"fig9_smallworld_{band}", tag)
        ax = fig.add_subplot(111)
        # 年龄分箱（每年一箱），确保均值和标准误分组一致
        bins = np.arange(np.floor(sub["age"].min()),
                         np.ceil(sub["age"].max()) + 2, 1)
        grp_mean = sub.groupby(pd.cut(sub["age"], bins, right=False))[[col]].mean()
        grp_sem = sub.groupby(pd.cut(sub["age"], bins, right=False))[[col]].sem()
        grp_mean = grp_mean.dropna()
        # sem 对齐 mean 的分组
        grp_sem = grp_sem.loc[grp_mean.index]
        centers = [iv.mid for iv in grp_mean.index]
        means = grp_mean[col].to_numpy()
        sems = grp_sem[col].fillna(0).to_numpy()
        # 排除无数据的分箱
        mask = ~np.isnan(means)
        if mask.sum() < 3:
            plt.close(fig)
            continue
        centers = [c for c, m in zip(centers, mask) if m]
        means = means[mask]
        sems = sems[mask]
        ax.errorbar(centers, means, yerr=sems, fmt="-o", c="#2E75B6",
                    capsize=3, lw=1.5, ms=4)
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_xlabel("Age (years)"); ax.set_ylabel("Small-worldness sigma")
        ax.set_title(f"Band: {band}")
        fig.tight_layout(); fig.savefig(fname); plt.close(fig)
        logger.info("图9保存: %s", fname)


def fig12_roc(iad_df, tag):
    """图12: IAD 判别高风险组 ROC"""
    from sklearn.metrics import roc_auc_score, roc_curve
    iad_cols = [c for c in iad_df.columns if c.endswith("__iad")]
    fig, fname = _fig("fig12_roc", tag)
    ax = fig.add_subplot(111)
    plotted = False
    for col in iad_cols:
        df = iad_df[[col, "externalizing"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 20:
            continue
        y = (df["externalizing"] > 1.0).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, df[col])
        auc = roc_auc_score(y, df[col])
        band = col.split("__")[0]
        ax.plot(fpr, tpr, lw=1.5, label=f"{band} (AUC={auc:.2f})")
        plotted = True
    if plotted:
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC: IAD vs High Externalizing (z>1)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fname)
    plt.close(fig)
    if plotted:
        logger.info("图12保存: %s", fname)


def make_all_figures(cfg, features, iad_df, band_names, tae_df, tag="full"):
    """生成全部图表"""
    am = cfg["allometry"]
    fig3_loglog(features, band_names, tae_df, tag)
    fig4_violin(features, band_names, am["age_group_edges"],
                am["age_group_labels"], tag)
    if len(tae_df):
        fig6_radar(tae_df, tag)
    fig9_smallworld(features, band_names, tag)
    fig12_roc(iad_df, tag)
    logger.info("全部图表生成完毕 -> %s", FIGURES_ROOT)
