"""
covariance.py — 协方差矩阵估计与 SPD 网络构建
- Ledoit-Wolf 收缩协方差（对每个滑动窗口）
- 窗口间黎曼均值(log-Euclidean) 或算术均值
- 转换为相关矩阵，密度阈值化，输出加权邻接矩阵
"""

import logging
import warnings

import numpy as np
from scipy.linalg import expm, logm
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def _log_euclidean_mean(mats):
    """黎曼(log-Euclidean)均值：exp(mean(log(mat)))"""
    n = mats.shape[0]
    with warnings.catch_warnings():
        # logm 的近似误差警告(1e-13量级)为正常数值噪声，屏蔽避免日志刷屏
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="logm result may be inaccurate")
        logs = np.stack([logm(m) for m in mats], axis=0)
        return expm(logs.mean(axis=0))


def estimate_covariance(seg_data, win_len_samp, win_overlap, method="ledoit_wolf",
                        riemann_mean=True):
    """
    对给定频带的全部闭眼段数据估计 128x128 SPD 协方差矩阵。

    seg_data: (n_segments, n_channels, n_samples)
    返回: (n_channels, n_channels) SPD 协方差矩阵
    """
    n_chan = seg_data.shape[1]
    win_step = max(1, int(win_len_samp * (1 - win_overlap)))

    covs = []
    for seg in seg_data:
        n_samples = seg.shape[1]
        starts = range(0, n_samples - win_len_samp + 1, win_step)
        for s in starts:
            win = seg[:, s:s + win_len_samp]
            if method == "ledoit_wolf":
                lw = LedoitWolf(store_precision=False, assume_centered=False)
                lw.fit(win.T)  # (samples, channels)
                cov = lw.covariance_
            else:
                cov = np.cov(win)
            # 数值稳定：加微小正则确保正定
            cov = cov + 1e-10 * np.eye(n_chan)
            covs.append(cov)

    if len(covs) == 0:
        raise ValueError("窗口数为0，无法估计协方差")

    covs = np.stack(covs, axis=0)
    if riemann_mean and len(covs) > 1:
        try:
            return _log_euclidean_mean(covs)
        except Exception as e:  # noqa: BLE001
            logger.warning("黎曼均值失败(%s)，回退算术均值", e)
            return covs.mean(axis=0)
    return covs.mean(axis=0)


def cov_to_correlation(cov):
    """协方差 -> 相关矩阵（对角元素为1）"""
    d = np.sqrt(np.clip(np.diag(cov), a_min=1e-12, a_max=None))
    corr = cov / np.outer(d, d)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    return corr


def density_threshold(corr, density, use_abs=True):
    """
    密度阈值化：保留强度最高的 density 比例连接，返回加权邻接矩阵。
    对角线置0。use_abs=True 时按绝对值比较强度。
    """
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    iu = np.triu_indices(n, k=1)
    weights = corr[iu]
    if use_abs:
        strengths = np.abs(weights)
    else:
        strengths = weights
    n_keep = max(1, int(round(len(weights) * density)))
    idx = np.argsort(strengths)[::-1][:n_keep]
    adj[iu[0][idx], iu[1][idx]] = weights[idx]
    adj = adj + adj.T
    return adj


def mst_threshold(corr):
    """
    最小生成树阈值化（可选方案）：保留所有节点的树连接。
    返回加权邻接矩阵（MST 边数 = n-1）。
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    # 将相关性转为距离：d = 1 - |corr|，MST 取最小距离
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    mst = minimum_spanning_tree(dist).toarray()
    mst = mst + mst.T
    # 恢复权重为相关性
    adj = np.where(mst > 0, corr, 0.0)
    return adj


def build_network(cov, threshold_mode="density", density=0.2,
                  use_abs=True, edge_weight="correlation"):
    """由 SPD 协方差构建加权邻接矩阵"""
    if edge_weight == "correlation":
        mat = cov_to_correlation(cov)
    else:
        mat = cov.copy()
    if threshold_mode == "density":
        return density_threshold(mat, density, use_abs)
    elif threshold_mode == "mst":
        return mst_threshold(mat)
    raise ValueError(f"未知阈值化方式: {threshold_mode}")
