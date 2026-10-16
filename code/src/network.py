"""
network.py — 图论拓扑指标计算
- 加权邻接矩阵 -> networkx 图
- 全局效率、局部效率、模块化指数 Q、小世界属性 sigma
效率类指标基于 scipy 加权最短路径（networkx 的 efficiency 不支持加权）；
模块化、聚类、随机化基准基于 networkx。
"""

import logging
import warnings

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity as nx_modularity
from scipy.sparse.csgraph import shortest_path

logger = logging.getLogger(__name__)


def _to_graph(adj, abs_weight=False):
    """
    加权邻接矩阵 -> 加权无向图。
    边权重 = 连接强度（绝对值，因为模块化/聚类关心强度而非符号）。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.abs(adj) if abs_weight else adj
    G = nx.from_numpy_array(w, parallel_edges=False, create_using=nx.Graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _to_distance_matrix(adj):
    """连接强度 -> 距离矩阵（d = 1/|w|，无连接=inf，对角线=0）"""
    n = adj.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.abs(adj)
    d = np.where(adj != 0, inv, np.inf)
    np.fill_diagonal(d, 0.0)
    return d


def global_efficiency_w(adj):
    """加权全局效率 = mean(1/d_ij)，d 为加权最短路径（scipy shortest_path），排除对角线"""
    n = adj.shape[0]
    if n <= 1:
        return 0.0
    d = _to_distance_matrix(adj)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = shortest_path(d, method="auto", directed=False)
    off_diag = sp[~np.eye(n, dtype=bool)]
    off_diag = off_diag[np.isfinite(off_diag)]
    if len(off_diag) == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(1.0 / off_diag))


def local_efficiency_w(adj):
    """
    加权局部效率：对每个节点 i，取其邻居子图（排除 i 后），
    计算子图内两两节点间的最短路径效率的平均值（排除子图对角线）。
    """
    n = adj.shape[0]
    if n <= 2:
        return 0.0
    d = _to_distance_matrix(adj)
    effs = []
    for i in range(n):
        neigh = np.where(adj[i] != 0)[0]
        if len(neigh) < 2:
            continue
        sub = d[np.ix_(neigh, neigh)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            with np.errstate(divide="ignore", invalid="ignore"):
                sp = shortest_path(sub, method="auto", directed=False)
        m = len(neigh)
        off_diag = sp[~np.eye(m, dtype=bool)]
        off_diag = off_diag[np.isfinite(off_diag)]
        if len(off_diag) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            effs.append(np.mean(1.0 / off_diag))
    return float(np.mean(effs)) if effs else 0.0


def modularity_q(adj):
    """模块化指数 Q（Louvain 贪心社区发现 + Newman 模块度），权重取绝对值"""
    G = _to_graph(adj, abs_weight=True)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    try:
        comms = greedy_modularity_communities(G, weight="weight")
        return float(nx_modularity(G, comms, weight="weight"))
    except Exception as e:  # noqa: BLE001
        logger.debug("模块化计算失败: %s", e)
        return 0.0


def _weighted_clustering_coeff(adj):
    """加权聚类系数（Onnela 定义），无向加权图，权重取绝对值"""
    G = _to_graph(adj, abs_weight=True)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    return float(nx.average_clustering(G, weight="weight"))


def _weighted_avg_path_length(adj):
    """加权平均最短路径长度（基于距离图，只统计连通且非对角的节点对）"""
    n = adj.shape[0]
    if n <= 1:
        return np.nan
    d = _to_distance_matrix(adj)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = shortest_path(d, method="auto", directed=False)
    off_diag = sp[~np.eye(n, dtype=bool)]
    vals = off_diag[np.isfinite(off_diag)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def small_worldness(adj, n_random=20, rng_seed=42):
    """
    小世界属性 sigma = (C/C_rand) / (L/L_rand)
    随机基准：度保持随机化（double_edge_swap），n_random 次取平均。
    """
    C = _weighted_clustering_coeff(adj)
    L = _weighted_avg_path_length(adj)

    G = _to_graph(adj, abs_weight=True)
    if G.number_of_edges() < 3:
        return 0.0, C, L

    n_edges = G.number_of_edges()
    # nswap 与 n_edges 匹配，保证充分随机化
    nswap = max(20, n_edges)
    max_tries = nswap * 20

    rng = np.random.default_rng(rng_seed)
    C_rands, L_rands = [], []
    base = G.copy()
    for _ in range(n_random):
        G_rand = base.copy()
        try:
            # nx 3.6.1 不支持 random_state/seed 参数，直接调用即可
            # 注：nx>=3.4 将 swap 尝试耗尽包装为 NetworkXAlgorithmError
            #（与 NetworkXError 平级），须用父类 NetworkXException 兜底，
            # 否则异常逃逸导致单被试处理失败
            nx.double_edge_swap(G_rand, nswap=nswap, max_tries=max_tries)
        except nx.NetworkXException:
            try:
                nx.double_edge_swap(G_rand, nswap=max(5, nswap // 4),
                                    max_tries=max_tries)
            except nx.NetworkXException:
                break
        adj_rand = nx.to_numpy_array(G_rand, nodelist=sorted(G_rand.nodes()),
                                     weight="weight")
        C_rands.append(_weighted_clustering_coeff(adj_rand))
        L_rands.append(_weighted_avg_path_length(adj_rand))

    if not C_rands or not L_rands:
        return 0.0, C, L

    C_rand = float(np.mean(C_rands))
    L_rand = float(np.mean([x for x in L_rands if np.isfinite(x)])) if any(
        np.isfinite(x) for x in L_rands) else np.nan

    if C_rand <= 0 or not np.isfinite(L_rand) or L_rand <= 0 or L <= 0 or C <= 0:
        return 0.0, C, L

    sigma = (C / C_rand) / (L / L_rand)
    return float(sigma), float(C), float(L)


def compute_all_metrics(adj, n_random=20, rng_seed=42):
    """计算全部拓扑指标，返回 dict"""
    return {
        "global_efficiency": global_efficiency_w(adj),
        "local_efficiency": local_efficiency_w(adj),
        "modularity": modularity_q(adj),
        "small_worldness": small_worldness(adj, n_random, rng_seed)[0],
        "clustering": _weighted_clustering_coeff(adj),
    }
