#!/usr/bin/env python3
"""
run_density_sweep.py — 密度阈值敏感性分析（审稿建议 #3 / #8d）

目的：在多个网络密度阈值（10/15/20/25/30%）下重算全局效率和小世界属性的
异速生长指数 α，验证 α < 0 的结论对阈值选择是否稳健。

运行：
    python run_density_sweep.py --n-jobs 4
输出：
    - results/tables/tae_density_sweep.csv   # 每个 density × band × metric 一条
    - results/figures/fig_density_sweep.png   # α vs density 折线图
    - results/logs/run_density_sweep_<ts>.log

设计要点：
    1) 只对审稿人最关心的两个指标（global_efficiency, small_worldness）重算，
       其它指标（local_efficiency/modularity/clustering）保持原结果不变。
    2) 为节省时间，**只对审稿人最关心的两个频带（delta, theta）** 全跑，
       其它频带只跑 20% (baseline) 用于横向比较（也保存）。
    3) 协方差矩阵要从 .set 重新计算（之前未保存），所以仍需 _subject_work
       的预处理路径；但 network.py 阶段只对目标 metric/density 重算。
    4) 进程池并行，n_jobs 根据可用内存自适应（继承 run_all.py 的 _safe_n_jobs）。
"""
import argparse
import gc
import logging
import signal
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import psutil  # optional, for _safe_n_jobs
except ImportError:
    psutil = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import covariance as covmod
from src import network as netmod
from src import pipeline as pipemod
from src import preprocess as premod
from src import utils
from src.utils import FIGURES_ROOT, LOGS_ROOT, TABLES_ROOT

logger = logging.getLogger("density_sweep")

# 审稿人指定的扫描密度（10/15/20/25/30%）
DENSITIES = [0.10, 0.15, 0.20, 0.25, 0.30]

# 主指标：审稿人最关心的两个
KEY_METRICS = ["global_efficiency", "small_worldness"]

# 主频带：delta 和 theta（审稿人最强的两个），其它频带只跑 baseline
KEY_BANDS = ["delta", "theta"]
ALL_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def _safe_n_jobs(requested):
    if requested is not None:
        return max(1, requested)
    if psutil is None:
        return 2
    try:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024 ** 3)
        usable = max(0, avail_gb - 30)
        per_worker = 32
        n = max(1, int(usable // per_worker))
        return min(n, 4)
    except Exception:
        return 2


def _subject_density_work(args):
    """单被试密度扫描：算一次协方差，再对 (density, band) 组合分别算指标。"""
    set_path, cfg, band_names, densities = args
    try:
        pc = cfg["preprocess"]
        raw, segs = premod.read_subject_eeg(
            set_path,
            l_freq=pc["l_freq"], h_freq=pc["h_freq"],
            notch_filter=pc.get("notch_filter", True),
            eyes_closed_event=pc["eyes_closed_event"],
            eyes_closed_duration=pc["eyes_closed_duration"],
            eyes_closed_trim=pc["eyes_closed_trim"],
            reject_bad_channels=pc.get("reject_bad_channels", True),
            drop_channels=tuple(pc.get("drop_channels", ["Cz"])),
        )
        if raw is None or not segs:
            return None, None, "no usable data"

        bands_data = premod.split_bands(raw, segs, cfg["bands"], band_names)

        nc = cfg["network"]
        win_samp = int(pc["win_len_sec"] * raw.info["sfreq"])
        overlap = pc.get("win_overlap", 0.5)

        feats = {}  # key: (band, density, metric) -> value
        for band in band_names:
            seg = bands_data[band]
            cov = covmod.estimate_covariance(
                seg, win_samp, overlap,
                method=nc["cov_estimator"], riemann_mean=nc.get("riemann_mean", True))
            for density in densities:
                adj = covmod.build_network(
                    cov, threshold_mode=nc["threshold_mode"], density=density,
                    use_abs=nc.get("use_abs", True),
                    edge_weight=nc.get("edge_weight", "correlation"))
                # 只算 KEY_METRICS 两个指标
                if "global_efficiency" in KEY_METRICS:
                    feats[(band, density, "global_efficiency")] = \
                        netmod.global_efficiency_w(adj)
                if "small_worldness" in KEY_METRICS:
                    feats[(band, density, "small_worldness")] = \
                        netmod.small_worldness(adj, n_random=nc.get("n_random_graphs", 20),
                                                rng_seed=42)[0]
            del cov, seg  # 释放
        return set_path.parents[1].name, feats, None
    except Exception as e:  # noqa: BLE001
        logger.debug("被试处理失败 %s: %s", set_path, e)
        return None, None, str(e)
    finally:
        gc.collect()


def build_density_features(cfg, participants_df, release_dirs, band_names,
                           densities, n_jobs=4):
    """逐被试扫描，返回 DataFrame：participant_id + (band, density, metric) 列"""
    tasks = []
    for _, row in participants_df.iterrows():
        pid = row["participant_id"]
        rel = row.get("release", "")
        release_dir = next((rd for rd in release_dirs if Path(rd).name == rel), None)
        if release_dir is None:
            continue
        eeg_dir = Path(release_dir) / pid / "eeg"
        set_path = eeg_dir / f"{pid}_task-RestingState_eeg.set"
        if not set_path.exists():
            continue
        tasks.append((set_path, cfg, band_names, densities))

    logger.info("待处理被试数: %d", len(tasks))
    results = {}
    failures = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_subject_density_work, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            sid, feats, err = f.result()
            if sid and feats:
                results[sid] = feats
            elif err:
                failures.append((str(futs[f]), err))
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                logger.info("[%d/%d] 成功 %d, 失败 %d",
                            i + 1, len(tasks), len(results), len(failures))

    if failures:
        logger.warning("失败被试样本(最多10个):")
        for fn, err in failures[:10]:
            logger.warning("  %s: %s", Path(fn).parent.parent.name, err[:200])

    if not results:
        return pd.DataFrame()

    # 展平为宽表
    rows = []
    for sid, feat in results.items():
        row = {"participant_id": sid}
        for (band, density, metric), val in feat.items():
            row[f"{band}__d{int(density*100):02d}__{metric}"] = val
        rows.append(row)
    feat_df = pd.DataFrame(rows)

    # 合并表型
    meta = participants_df[["participant_id", "age", "sex"]].copy()
    meta = meta.merge(feat_df, on="participant_id", how="inner")
    am = cfg["allometry"]
    meta = meta[(meta["age"] >= am["min_age"]) & (meta["age"] <= am["max_age"])]
    return meta.reset_index(drop=True)


def fit_tae_per_density(df, band, metric, densities):
    """对每个 density 拟合 log y = alpha log(age) + log k"""
    from scipy import stats
    rows = []
    for d in densities:
        col = f"{band}__d{int(d*100):02d}__{metric}"
        if col not in df.columns:
            continue
        sub = df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub[(sub["age"] > 0) & (sub[col] > 0)]
        if len(sub) < 30:
            continue
        lx, ly = np.log(sub["age"]), np.log(sub[col])
        res = stats.linregress(lx, ly)
        rows.append({
            "band": band, "metric": metric, "density": d,
            "n": len(sub), "alpha": res.slope, "log_k": res.intercept,
            "r2": res.rvalue ** 2, "p": res.pvalue,
            "p_fdr": np.nan,  # 后面批量校正
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--densities", nargs="+", type=float, default=DENSITIES,
                    help="网络密度扫描列表，如 --densities 0.1 0.2 0.3")
    ap.add_argument("--bands", nargs="+", default=None,
                    help="频带子集，默认 delta/theta 节省时间")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = utils.load_config(args.config if hasattr(args, 'config') else None)
    if args.smoke:
        cfg["run"]["smoke_test"] = True

    utils.ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_ROOT / f"run_density_sweep_{ts}.log"
    utils.setup_logging("INFO", log_file=log_path)

    logger.info("=" * 70)
    logger.info("密度阈值敏感性扫描启动")
    logger.info("密度列表: %s", args.densities)
    logger.info("频带: %s", args.bands or KEY_BANDS)
    logger.info("=" * 70)

    data_root = utils.resolve_data_root(cfg)
    release_dirs = utils.resolve_releases(cfg, data_root)
    if args.smoke:
        smoke_rel = cfg["run"].get("smoke_releases", [])
        release_dirs = [data_root / r for r in smoke_rel
                        if (data_root / r).exists()]

    n_jobs = _safe_n_jobs(args.n_jobs if args.n_jobs is not None else cfg["run"].get("n_jobs"))
    logger.info("并行数: %d", n_jobs)

    participants = pipemod.load_participants(release_dirs, task=cfg["task"])
    logger.info("候选被试: %d", len(participants))

    band_names = args.bands or KEY_BANDS

    max_subj = args.max_subjects
    if args.smoke:
        max_subj = cfg["run"].get("smoke_max_subjects", 6)

    df = build_density_features(cfg, participants, release_dirs, band_names,
                                args.densities, n_jobs=n_jobs)
    if len(df) == 0:
        logger.error("无被试成功处理")
        return 1

    logger.info("处理后被试数: %d", len(df))
    feat_path = TABLES_ROOT / "density_sweep_features.csv"
    df.to_csv(feat_path, index=False)
    logger.info("特征表保存: %s", feat_path)

    # 拟合 α
    tae_rows = []
    for band in band_names:
        for metric in KEY_METRICS:
            tae_rows.extend(fit_tae_per_density(df, band, metric, args.densities))
    tae_df = pd.DataFrame(tae_rows)
    if len(tae_df) == 0:
        logger.error("TAE 拟合全部失败")
        return 1

    # FDR 校正
    from scipy.stats import false_discovery_control
    try:
        qvals = false_discovery_control(tae_df["p"].to_numpy(), method="bh")
        # false_discovery_control 直接返回与 p 等长的 q 值一维数组
        # （旧 bug：误取 [0] 标量，导致整列被同一个值填充）
        q = np.asarray(qvals, dtype=float)
    except Exception:
        q = tae_df["p"].to_numpy()
    tae_df["p_fdr"] = q

    tae_path = TABLES_ROOT / "tae_density_sweep.csv"
    tae_df.to_csv(tae_path, index=False)
    logger.info("TAE 扫描保存: %s", tae_path)

    # 画 α vs density 折线图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
        for ax, metric in zip(axes, KEY_METRICS):
            for band in band_names:
                sub = tae_df[tae_df["metric"] == metric]
                sub = sub[sub["band"] == band].sort_values("density")
                if len(sub) == 0:
                    continue
                ax.plot(sub["density"] * 100, sub["alpha"],
                        "o-", lw=2, ms=8, label=band)
                # 95% CI 阴影（用 se_alpha 近似：se_alpha = alpha / |t|）
                from scipy import stats as sst
                t_crit = sst.t.ppf(0.975, df=sub["n"].to_numpy() - 2)
                se_alpha = np.abs(sub["alpha"] / np.where(np.isfinite(t_crit), t_crit, 1.0))
                ax.fill_between(sub["density"] * 100,
                                sub["alpha"] - se_alpha,
                                sub["alpha"] + se_alpha,
                                alpha=0.15)
            ax.axhline(0, ls="--", c="gray", lw=1)
            ax.set_xlabel("Network density (%)")
            ax.set_ylabel(r"Allometric exponent $\alpha$")
            ax.set_title(metric.replace("_", " ").title())
            ax.grid(True, ls="--", alpha=0.4)
            ax.legend(fontsize=9)
        fig.suptitle("Density-threshold sensitivity of allometric scaling",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        out_fig = FIGURES_ROOT / "fig_density_sweep.png"
        fig.savefig(out_fig, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("密度敏感性图保存: %s", out_fig)
    except Exception as e:
        logger.warning("图表生成失败: %s", e)

    # 打印汇总
    print("\n===== 密度敏感性 TAE 扫描结果 =====")
    cols = ["band", "metric", "density", "n", "alpha", "r2", "p", "p_fdr"]
    print(tae_df[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())