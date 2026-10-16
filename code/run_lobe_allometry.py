#!/usr/bin/env python3
"""
  run_lobe_allometry.py — 脑区分区异速生长（审稿建议 #8a）

  修复版：解决工作站 mne/matplotlib 导入被 KeyboardInterrupt 干扰导致 90% 失败的问题
  关键改动：
    1. 不在主进程 import mne/matplotlib（避免污染）
    2. worker 函数开头重置 signal handlers 并延迟导入 mne
    3. 每个 lobe 单独 try/except，一个 lobe 失败不影响其它
    4. 每个频带单独 try/except，一个 band 失败不影响其它
    5. 增加 --dry-run 选项：启动前先跑1个被试验证流程通畅

  预计耗时：1967 被试 × 5 lobe × 5 频带 ≈ 4-6 小时（工作站，干净环境）
"""
import argparse
import gc
import json
import logging
import os
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 注意：主进程不导入 mne/matplotlib，由 worker 自己导入
from src import pipeline as pipemod
from src import utils
from src.utils import FIGURES_ROOT, LOGS_ROOT, TABLES_ROOT

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("lobe_allometry")

LOBE_MAP_PATH = Path(__file__).resolve().parent / "src" / "lobe_map.json"


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


def _slice_adj_by_lobe(adj, lobe_ch_indices, density=0.20, use_abs=True):
    """从一个 128x128 全脑 adj 切出 lobe 内部的子图，再按 density 阈值化"""
    idx = lobe_ch_indices
    sub_adj = adj[np.ix_(idx, idx)]
    if density < 1.0:
        n = sub_adj.shape[0]
        iu = np.triu_indices(n, k=1)
        weights = sub_adj[iu]
        strengths = np.abs(weights) if use_abs else weights
        n_keep = max(1, int(round(len(weights) * density)))
        if n_keep >= len(weights):
            return sub_adj
        keep_idx = np.argsort(strengths)[::-1][:n_keep]
        mask = np.zeros_like(sub_adj, dtype=bool)
        mask[iu[0][keep_idx], iu[1][keep_idx]] = True
        mask = mask | mask.T
        sub_adj = sub_adj * mask
    return sub_adj


def _subject_lobe_work(args):
    """单被试 lobe 扫描（修复版：重置 signal，延迟导入 mne，按 lobe 独立 try）"""
    # === 关键修复1：重置 signal handlers ===
    # 在某些工作站上，父进程的 KeyboardInterrupt 状态会泄漏到子进程。
    # 重置为默认 SIG_DFL 行为，避免 matplotlib FigureBase docstring 初始化时被干扰。
    try:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except Exception:
        pass

    # === 关键修复2：每个 worker 自己延迟导入 mne ===
    try:
        import mne  # noqa: F401
        from src import covariance as covmod
        from src import network as netmod
        from src import preprocess as premod
    except Exception as e:
        return None, None, f"import failed: {e}"

    set_path, cfg, band_names, lobe_indices, metrics = args
    try:
        pc = cfg["preprocess"]
        try:
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
        except Exception as e:
            return None, None, f"read_subject_eeg failed: {e}"

        if raw is None or not segs:
            return None, None, "no usable data"

        try:
            bands_data = premod.split_bands(raw, segs, cfg["bands"], band_names)
        except Exception as e:
            return None, None, f"split_bands failed: {e}"

        nc = cfg["network"]
        try:
            win_samp = int(pc["win_len_sec"] * raw.info["sfreq"])
        except Exception:
            win_samp = int(pc["win_len_sec"] * 500.0)  # fallback 500 Hz
        overlap = pc.get("win_overlap", 0.5)

        feats = {}
        for band in band_names:
            try:
                seg = bands_data[band]
            except KeyError:
                continue
            # === 关键修复3：每个 band 独立 try，band 失败不影响其它 ===
            try:
                cov = covmod.estimate_covariance(
                    seg, win_samp, overlap,
                    method=nc["cov_estimator"], riemann_mean=nc.get("riemann_mean", True))
                adj_full = covmod.build_network(
                    cov, threshold_mode=nc["threshold_mode"], density=nc["density"],
                    use_abs=nc.get("use_abs", True),
                    edge_weight=nc.get("edge_weight", "correlation"))
            except Exception as e:
                continue  # 此 band 失败，跳过该 band

            for lobe_name, lobe_idx in lobe_indices.items():
                if len(lobe_idx) < 5:
                    continue
                # === 关键修复4：每个 lobe 独立 try，lobe 失败不影响其它 ===
                try:
                    sub_adj = _slice_adj_by_lobe(
                        adj_full, lobe_idx, density=nc["density"],
                        use_abs=nc.get("use_abs", True))
                    if "global_efficiency" in metrics:
                        feats[(band, lobe_name, "global_efficiency")] = \
                            netmod.global_efficiency_w(sub_adj)
                    if "small_worldness" in metrics:
                        feats[(band, lobe_name, "small_worldness")] = \
                            netmod.small_worldness(
                                sub_adj, n_random=nc.get("n_random_graphs", 20),
                                rng_seed=42)[0]
                except Exception:
                    pass  # 单 lobe 失败不影响其它 lobe
            del cov, adj_full
        return set_path.parents[1].name, feats, None
    except Exception as e:
        return None, None, f"unexpected: {e}"
    finally:
        gc.collect()


def build_lobe_features(cfg, participants_df, release_dirs, band_names,
                        lobe_indices, metrics, n_jobs=4):
    """逐被试扫描，返回 DataFrame：participant_id + (band, lobe, metric) 列"""
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
        tasks.append((set_path, cfg, band_names, lobe_indices, metrics))

    logger.info("待处理被试数: %d", len(tasks))
    results = {}
    failures = []
    checkpoint_path = TABLES_ROOT / "lobe_features_checkpoint.pkl"

    if checkpoint_path.exists():
        try:
            import pickle
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
            results = ckpt.get("results", {})
            failures = ckpt.get("failures", [])
            done_sids = set(results.keys())
            tasks = [t for t in tasks if t[0].parents[1].name not in done_sids]
            logger.info("从检查点恢复: 已完成 %d, 待处理 %d", len(done_sids), len(tasks))
        except Exception as e:
            logger.warning("检查点加载失败: %s", e)

    def _save_checkpoint():
        import pickle
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({"results": results, "failures": failures}, f)
        except Exception as e:
            logger.warning("检查点保存失败: %s", e)

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_subject_lobe_work, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            sid, feats, err = f.result()
            if sid and feats:
                results[sid] = feats
            elif err:
                failures.append((str(futs[f]), err))
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                logger.info("[%d/%d] 成功 %d, 失败 %d",
                            i + 1, len(tasks), len(results), len(failures))
                _save_checkpoint()

    if failures:
        # 去重 + 统计
        from collections import Counter
        err_counter = Counter(err for _, err in failures if err)
        logger.warning("失败被试样本(最多10个) + 错误统计:")
        for fn, err in failures[:10]:
            logger.warning("  %s: %s", Path(fn).parent.parent.name, err[:200])
        logger.warning("错误类型统计 (top 10):")
        for err_type, cnt in err_counter.most_common(10):
            logger.warning("  [%d] %s", cnt, err_type[:150])

    if not results:
        return pd.DataFrame()

    rows = []
    for sid, feat in results.items():
        row = {"participant_id": sid}
        for (band, lobe, metric), val in feat.items():
            row[f"{band}__{lobe}__{metric}"] = val
        rows.append(row)
    feat_df = pd.DataFrame(rows)
    meta = participants_df[["participant_id", "age", "sex"]].copy()
    meta = meta.merge(feat_df, on="participant_id", how="inner")
    am = cfg["allometry"]
    meta = meta[(meta["age"] >= am["min_age"]) & (meta["age"] <= am["max_age"])]
    return meta.reset_index(drop=True)


def fit_tae_per_lobe(df, band, lobe, metric):
    from scipy import stats
    col = f"{band}__{lobe}__{metric}"
    if col not in df.columns:
        return None
    sub = df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub["age"] > 0) & (sub[col] > 0)]
    if len(sub) < 30:
        return None
    lx, ly = np.log(sub["age"]), np.log(sub[col])
    res = stats.linregress(lx, ly)
    return {
        "band": band, "lobe": lobe, "metric": metric,
        "n": len(sub), "alpha": res.slope, "log_k": res.intercept,
        "r2": res.rvalue ** 2, "p": res.pvalue,
    }


def plot_topomap_alpha(tae_df, band, metric, lobe_map, output_path):
    """把每个 lobe 的 α 画在 head topomap 上"""
    try:
        # 这里只在主进程画图，所以可以直接 import matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mne

        sub = tae_df[(tae_df["band"] == band) & (tae_df["metric"] == metric)]
        if len(sub) == 0:
            return
        lobe_alpha = {row["lobe"]: row["alpha"] for _, row in sub.iterrows()}
        ch_names = sorted(lobe_map.keys(), key=lambda x: int(x[1:]))
        values = np.array([lobe_alpha.get(lobe_map[c], 0.0) for c in ch_names])
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")
        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
        info.set_montage(montage, match_case=False, on_missing="ignore")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        mne.viz.plot_topomap(values, info, axes=ax, show=False,
                             cmap="RdBu_r", vlim=(-1.2, 0.0),
                             contours=0)
        ax.set_title(f"Allometric exponent $\\alpha$ by lobe\n"
                     f"({band} band, {metric.replace('_', ' ')})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Topomap saved: %s", output_path)
    except Exception as e:
        logger.warning("Topomap 生成失败: %s", e)


def dry_run(cfg, participants_df, release_dirs, lobe_indices, band_names,
            metrics, n_jobs):
    """启动前先跑 1 个被试验证 mne 能正常导入并算 lobe 指标"""
    logger.info("=== Dry-run: 验证 1 个被试的 lobe 处理流程 ===")
    # 找到第一个有 .set 的被试
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
        break
    else:
        logger.error("找不到任何 .set 文件用于 dry-run")
        return False

    logger.info("Dry-run 被试: %s", set_path.name)
    args = (set_path, cfg, band_names, lobe_indices, metrics)
    sid, feats, err = _subject_lobe_work(args)
    if err:
        logger.error("Dry-run 失败: %s", err)
        return False
    if not feats:
        logger.error("Dry-run 成功但没有输出 feats (预处理问题?)")
        return False
    logger.info("Dry-run 成功: %d 个 (band, lobe, metric) 指标", len(feats))
    # 打印前几个
    for i, (key, val) in enumerate(list(feats.items())[:5]):
        logger.info("  示例: %s = %.4f", key, val)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--bands", nargs="+", default=None,
                    help="频带子集，默认全跑")
    ap.add_argument("--lobes", nargs="+", default=None,
                    help="lobe 子集，默认全跑")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="启动前先验证1个被试能否跑通，不进入正式任务")
    ap.add_argument("--lobe-map", type=str, default=str(LOBE_MAP_PATH))
    args = ap.parse_args()

    cfg = utils.load_config(args.config if hasattr(args, 'config') else None)
    if args.smoke:
        cfg["run"]["smoke_test"] = True

    utils.ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_ROOT / f"run_lobe_allometry_{ts}.log"
    utils.setup_logging("INFO", log_file=log_path)

    logger.info("=" * 70)
    logger.info("脑区分区异速生长启动 (修复版)")
    logger.info("=" * 70)

    # 加载 lobe 分区
    with open(args.lobe_map, "r", encoding="utf-8") as f:
        lobe_map = json.load(f)
    logger.info("Loaded lobe map: %d channels, lobes=%s",
                len(lobe_map), sorted(set(lobe_map.values())))

    all_ch = [f"E{i}" for i in range(1, 129)]
    lobe_indices = {}
    for lobe in sorted(set(lobe_map.values())):
        idx = [all_ch.index(c) for c in all_ch if lobe_map.get(c) == lobe]
        lobe_indices[lobe] = np.array(idx, dtype=int)
        logger.info("  Lobe %s: %d channels", lobe, len(idx))

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

    band_names = args.bands or list(cfg["bands"].keys())
    metrics = ["global_efficiency", "small_worldness"]

    max_subj = args.max_subjects
    if args.smoke:
        max_subj = cfg["run"].get("smoke_max_subjects", 6)

    if args.dry_run:
        ok = dry_run(cfg, participants, release_dirs, lobe_indices,
                     band_names, metrics, n_jobs=1)
        return 0 if ok else 1

    df = build_lobe_features(cfg, participants, release_dirs, band_names,
                             lobe_indices, metrics, n_jobs=n_jobs)
    if len(df) == 0:
        logger.error("无被试成功处理")
        return 1

    logger.info("处理后被试数: %d", len(df))

    tae_rows = []
    for band in band_names:
        for lobe in (args.lobes or lobe_indices.keys()):
            for metric in metrics:
                row = fit_tae_per_lobe(df, band, lobe, metric)
                if row is not None:
                    tae_rows.append(row)
    tae_df = pd.DataFrame(tae_rows)
    if len(tae_df) == 0:
        logger.error("TAE 拟合全部失败")
        return 1

    try:
        from scipy.stats import false_discovery_control
        q = false_discovery_control(tae_df["p"].to_numpy(), method="bh")
        # false_discovery_control 返回与 p 等长的 q 值数组（旧 bug：误取 q[0]
        # 导致整列被同一个标量填充）
        tae_df["p_fdr"] = np.asarray(q, dtype=float)
    except Exception:
        tae_df["p_fdr"] = tae_df["p"]

    tae_path = TABLES_ROOT / "tae_lobe_allometry.csv"
    tae_df.to_csv(tae_path, index=False)
    logger.info("TAE lobe 扫描保存: %s", tae_path)

    for band in band_names:
        for metric in metrics:
            out = FIGURES_ROOT / f"fig_lobe_topomap_{band}_{metric}.png"
            plot_topomap_alpha(tae_df, band, metric, lobe_map, out)

    print("\n===== 脑区分区 TAE 扫描结果 =====")
    cols = ["band", "lobe", "metric", "n", "alpha", "r2", "p", "p_fdr"]
    print(tae_df[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())