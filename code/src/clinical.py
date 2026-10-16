"""
clinical.py — 临床关联分析
- IAD 与 CBCL 因子的偏相关（控制性别等协变量）
- ROC 分析：IAD 判别高风险组（externalizing > 1SD）
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def _partial_corr(x, y, covs):
    """
    偏相关系数：控制 covs 后 x 与 y 的相关。
    方法：分别对 covs 回归 x 和 y，取残差后计算 Pearson 相关。
    covs: (n, k) 矩阵。返回 (r, p)。
    """
    if len(covs) == 0 or covs.shape[1] == 0:
        r, p = stats.pearsonr(x, y)
        return r, p
    # 对 covs 中心化
    covs_c = covs - covs.mean(axis=0)
    x_c = x - x.mean()
    y_c = y - y.mean()
    # 分别用 covs 回归 x 和 y，取残差（关键：不将 x/y 自身放入设计矩阵）
    try:
        beta_x = np.linalg.lstsq(covs_c, x_c, rcond=None)[0]
        resid_x = x_c - covs_c @ beta_x
        beta_y = np.linalg.lstsq(covs_c, y_c, rcond=None)[0]
        resid_y = y_c - covs_c @ beta_y
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p


def partial_corr_analysis(features_df, iad_cols, outcomes, covariates,
                          fdr_alpha=0.05):
    """
    对每个 (IAD列, 结局) 计算偏相关。
    返回 DataFrame: iad_col, outcome, r, p, n, significant。
    """
    # 数值化协变量（在 features_df 副本上直接修改，避免 concat 重复列）
    df = features_df.copy()
    for c in covariates:
        if c not in df.columns:
            logger.warning("协变量 %s 不在特征表中，跳过", c)
            continue
        if df[c].dtype == object:
            df[c] = df[c].map({"M": 1, "F": 0, "m": 1, "f": 0,
                               "Male": 1, "Female": 0}).fillna(0)
    covars_exist = [c for c in covariates if c in df.columns]

    rows = []
    for iad in iad_cols:
        for outc in outcomes:
            if outc not in df.columns:
                continue
            sub = df[[iad, outc] + covars_exist].replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 10:
                continue
            x = sub[iad].to_numpy(float)
            y = sub[outc].to_numpy(float)
            covs = sub[covars_exist].to_numpy(float) if covars_exist else np.zeros(
                (len(sub), 0))
            r, p = _partial_corr(x, y, covs)
            rows.append({"iad_col": iad, "outcome": outc,
                         "r": r, "p": p, "n": len(sub)})

    res = pd.DataFrame(rows)
    if len(res):
        try:
            qvals = stats.false_discovery_control(res["p"].to_numpy(),
                                                  method="bh")
            # false_discovery_control 直接返回与 p 等长的 q 值一维数组
            # （旧 bug：误取 [0] 标量，导致整列被同一个值填充）
            res["q"] = np.asarray(qvals, dtype=float)
        except (TypeError, AttributeError, ValueError):
            res["q"] = res["p"]
        res["significant"] = res["q"] < fdr_alpha
    return res


def roc_analysis(features_df, iad_col, risk_col="externalizing", cutoff=1.0):
    """
    ROC：IAD 判别高风险组（标准化分数 > cutoff）。
    返回 dict: auc, ci, n_high, n_low, fpr, tpr, thresholds, col。
    """
    df = features_df[[iad_col, risk_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 20 or df[risk_col].nunique() < 2:
        return None
    y = (df[risk_col] > cutoff).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    x = df[iad_col].to_numpy(float)
    auc = roc_auc_score(y, x)
    # DeLong 简化 95% CI（正态近似）
    n1, n0 = int(y.sum()), int((1 - y).sum())
    se = np.sqrt((auc * (1 - auc)) / min(n1, n0)) if 0 < auc < 1 else 0.1
    lo, hi = max(0, auc - 1.96 * se), min(1, auc + 1.96 * se)
    fpr, tpr, thr = roc_curve(y, x)
    return {
        "col": iad_col, "auc": float(auc), "ci_lo": float(lo),
        "ci_hi": float(hi), "n_high": int(n1), "n_low": int(n0),
        "fpr": fpr, "tpr": tpr, "thresholds": thr,
    }
