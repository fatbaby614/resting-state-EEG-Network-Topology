"""
allometry.py — 拓扑异速生长建模
- 群体 TAE：log(y) = alpha * log(age) + log(k) 的线性回归斜率
- 个体异速生长偏差 IAD：实际值 - 群体预测值
- 年龄分组统计（用于小提琴图）
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def log_log_regression(x, y):
    """
    双对数线性回归：log(y) ~ log(x)。
    返回 dict: alpha(斜率), k(截距exp), r2, p, n, ci。
    y 需 > 0。
    """
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    res = stats.linregress(lx, ly)
    return {
        "n": int(mask.sum()),
        "alpha": float(res.slope),
        "k": float(np.exp(res.intercept)),
        "r2": float(res.rvalue ** 2),
        "p": float(res.pvalue),
        "slope_se": float(res.stderr),
        "intercept_se": float(res.intercept_stderr),
    }


def fit_group_tae(features_df, band, metric="global_efficiency"):
    """
    对单个频带+指标，拟合群体 TAE。
    features_df 需含列: age, f"{band}__{metric}"
    返回 dict 或 None。
    """
    col = f"{band}__{metric}"
    if col not in features_df.columns:
        return None
    x = features_df["age"].to_numpy(float)
    y = features_df[col].to_numpy(float)
    return log_log_regression(x, y)


def compute_iad(features_df, band, metric="global_efficiency"):
    """
    计算个体异速生长偏差 IAD（残差）：
    IAD = log(y_actual) - log(y_predicted) = log(y) - (alpha*log(age)+log(k))
    返回新列 {band}__{metric}__iad 并加回 df。
    """
    col = f"{band}__{metric}"
    iad_col = f"{band}__{metric}__iad"
    df = features_df.copy()
    mask = (df["age"] > 0) & (df[col] > 0) & np.isfinite(df[col])
    df[iad_col] = np.nan
    if mask.sum() < 3:
        return df
    fit = log_log_regression(df.loc[mask, "age"].to_numpy(),
                             df.loc[mask, col].to_numpy())
    if fit is None:
        return df
    lx = np.log(df.loc[mask, "age"])
    ly_pred = fit["alpha"] * lx + np.log(fit["k"])
    ly_act = np.log(df.loc[mask, col])
    df.loc[mask, iad_col] = ly_act - ly_pred
    return df


def assign_age_group(age, edges, labels):
    """按边界给年龄分组，返回组标签"""
    for i in range(len(edges) - 1):
        if edges[i] <= age < edges[i + 1]:
            return labels[i]
    return labels[-1]


def age_group_stats(features_df, band, metric, age_edges, age_labels):
    """
    按年龄组汇总某个指标的描述统计（用于图4小提琴图）。
    返回 DataFrame: age_group, count, mean, std, values(list)。
    """
    col = f"{band}__{metric}"
    rows = []
    df = features_df.dropna(subset=[col]).copy()
    df["age_group"] = df["age"].apply(
        lambda a: assign_age_group(a, age_edges, age_labels))
    for g in age_labels:
        sub = df[df["age_group"] == g]
        vals = sub[col].to_numpy()
        rows.append({
            "age_group": g,
            "count": int(len(vals)),
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "std": float(np.std(vals)) if len(vals) else np.nan,
            "values": vals,
        })
    return rows
