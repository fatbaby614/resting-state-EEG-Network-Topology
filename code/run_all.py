#!/usr/bin/env python3
"""
run_all.py — HBN 课题4 一键启动脚本（Windows/Linux 通用）

按顺序执行完整实验流水线：
  1. 数据准备：合并 participants、筛选被试
  2. 特征提取：EEG -> 频带 -> 协方差 -> 网络 -> 拓扑指标
  3. 异速生长建模：群体 TAE + 个体 IAD
  4. 临床关联：CBCL 偏相关 + ROC
  5. 图表输出

用法：
  python run_all.py                # 完整运行
  python run_all.py --smoke        # 冒烟测试（小样本、少量频带）
  python run_all.py --n-jobs 8     # 指定并行度
  python run_all.py --band alpha beta   # 只跑指定频带

所有控制台输出同时写入 results/logs/run_<时间戳>.log
"""

import argparse
import logging
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import psutil
import pandas as pd

# 保证可导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import allometry as allomod  # noqa: E402
from src import clinical as clinmod  # noqa: E402
from src import pipeline as pipemod  # noqa: E402
from src import utils  # noqa: E402

logger = logging.getLogger("run_all")


def _safe_n_jobs(requested):
    """根据可用内存自动调整并行数，避免 OOM kill。
    每个被试进程峰值约 30 GB（128 通道 EEG PSD + 协方差 + 网络）。
    """
    if requested is not None:
        return max(1, requested)
    try:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024 ** 3)
        # 保留 30 GB 余量给系统/OS + worker 自身
        usable = max(0, avail_gb - 30)
        per_worker = 32  # 128 通道 EEG 协方差 + 网络峰值
        n = max(1, int(usable // per_worker))
        return min(n, 4)
    except Exception:
        return 2


def _shutdown_handler(signum, frame):
    """优雅退出：通知所有子进程终止，避免孤儿进程。"""
    logger.warning("收到信号 %d，正在终止所有子进程...", signum)
    try:
        # 通知所有 child 进程（父进程 PID）
        my_pid = os.getpid()
        for pdir in os.listdir("/proc"):
            if not pdir.isdigit():
                continue
            try:
                with open(f"/proc/{pdir}/stat") as f:
                    stat = f.read().split()
                    if len(stat) > 3 and int(stat[3]) == my_pid:
                        child_pid = int(pdir)
                        try:
                            os.kill(child_pid, signal.SIGTERM)
                            logger.info("  终止子进程 %d", child_pid)
                        except OSError:
                            pass
            except (OSError, IndexError, ValueError):
                pass
    except Exception:
        pass
    logger.warning("退出完成，已终止所有子进程。")
    sys.exit(0)


def install_shutdown_handler():
    """安装信号处理器（仅 Linux）"""
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)


def parse_args():
    ap = argparse.ArgumentParser(description="HBN 课题4 一键实验")
    ap.add_argument("--config", type=str, default=None, help="配置文件路径")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试模式")
    ap.add_argument("--n-jobs", type=int, default=None, help="并行被试数")
    ap.add_argument("--band", nargs="+", default=None,
                    help="频带子集，如 --band alpha beta")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="最多处理被试数（调试用）")
    ap.add_argument("--skip-features", action="store_true",
                    help="跳过特征提取(若已有特征表)")
    ap.add_argument("--skip-clinical", action="store_true", help="跳过临床分析")
    return ap.parse_args()


def main():
    args = parse_args()

    # 安装信号处理器：确保 Ctrl+C / 终端断开时所有子进程被终止
    install_shutdown_handler()

    cfg = utils.load_config(args.config)

    # 冒烟模式合并配置
    if args.smoke:
        cfg["run"]["smoke_test"] = True

    # 时间戳日志文件
    utils.ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = utils.LOGS_ROOT / f"run_{ts}.log"
    utils.setup_logging(cfg["run"].get("log_level", "INFO"), log_file=log_path)

    run_cfg = cfg["run"]
    smoke = run_cfg.get("smoke_test", False)

    logger.info("=" * 70)
    logger.info("HBN 课题4 实验流水线启动  平台: %s  模式: %s",
                utils.get_platform_name(), "SMOKE" if smoke else "FULL")
    logger.info("控制台日志: %s", log_path)
    logger.info("=" * 70)

    # ---- 1. 数据准备 ----
    data_root = utils.resolve_data_root(cfg)
    release_dirs = utils.resolve_releases(cfg, data_root)

    if smoke:
        # 冒烟模式：只用一个 release，限制被试数
        smoke_rel = run_cfg.get("smoke_releases", [])
        if smoke_rel:
            release_dirs = [data_root / r for r in smoke_rel
                            if (data_root / r).exists()]
        if not release_dirs:
            logger.error("冒烟测试找不到指定 release，检查 data_root 与 smoke_releases")
            return 1

    logger.info("数据根目录: %s", data_root)
    logger.info("Release 列表: %s", [p.name for p in release_dirs])

    # ---- 2. 特征提取 ----
    band_names = args.band or list(cfg["bands"].keys())
    if smoke and args.band is None:
        band_names = run_cfg.get("smoke_bands", band_names)
    n_jobs = _safe_n_jobs(args.n_jobs if args.n_jobs is not None else run_cfg.get("n_jobs"))
    logger.info("并行数: %d (基于可用内存自动选择)", n_jobs)
    try:
        vm = psutil.virtual_memory()
        logger.info("系统可用内存: %.1f GB / 总计: %.1f GB", vm.available / 1024**3, vm.total / 1024**3)
    except Exception:
        pass
    max_subj = args.max_subjects
    if smoke:
        max_subj = run_cfg.get("smoke_max_subjects", 6)

    tag = "smoke" if smoke else "full"

    if not args.skip_features:
        participants = pipemod.load_participants(release_dirs, task=cfg["task"])
        logger.info("候选被试(表型+任务可用): %d", len(participants))
        if len(participants) == 0:
            logger.error("无可处理被试，终止。")
            return 1

        features = pipemod.build_subject_features(
            cfg, participants, release_dirs, band_names,
            max_subjects=max_subj, n_jobs=n_jobs)
        if len(features) == 0:
            logger.error("特征提取失败：0 个有效被试，终止。")
            return 1
        feat_path = pipemod.save_subject_features(features, tag=tag)
    else:
        # 复用已有特征表
        from src.utils import TABLES_ROOT
        feat_path = TABLES_ROOT / f"subjects_features_{tag}.csv"
        if not feat_path.exists():
            logger.error("跳过特征提取但找不到 %s", feat_path)
            return 1
        features = pd.read_csv(feat_path)
        logger.info("加载已有特征表: %s (n=%d)", feat_path, len(features))

    # ---- 3. 异速生长建模 (TAE + IAD) ----
    am = cfg["allometry"]
    metrics = ["global_efficiency", "local_efficiency", "modularity",
               "small_worldness"]
    tae_rows, iad_df = [], features.copy()

    for band in band_names:
        for metric in metrics:
            fit = allomod.fit_group_tae(features, band, metric)
            if fit is None:
                continue
            tae_rows.append({
                "band": band, "metric": metric, **fit,
                "interpretation": ("加速" if fit["alpha"] > 1 else
                                   ("减速" if fit["alpha"] < 1 else "匀速")),
            })
        iad_df = allomod.compute_iad(iad_df, band, "global_efficiency")

    tae_df = pd.DataFrame(tae_rows)
    if len(tae_df):
        tae_path = utils.TABLES_ROOT / f"tae_summary_{tag}.csv"
        tae_df.to_csv(tae_path, index=False)
        logger.info("TAE 汇总已保存: %s", tae_path)
        print("\n===== 群体 TAE (alpha) 汇总 =====")
        print(tae_df[["band", "metric", "alpha", "r2", "p", "n"]].to_string(
            index=False))
    else:
        logger.warning("TAE 拟合全部失败（样本太少或指标无有效值）")

    iad_cols = [c for c in iad_df.columns if c.endswith("__iad")]
    if iad_cols:
        iad_path = utils.TABLES_ROOT / f"iad_{tag}.csv"
        iad_df.to_csv(iad_path, index=False)
        logger.info("IAD 表已保存: %s (%d 列)", iad_path, len(iad_cols))

    # ---- 4. 临床关联 ----
    if not args.skip_clinical and iad_cols:
        cc = cfg["clinical"]
        pc = clinmod.partial_corr_analysis(
            iad_df, iad_cols, cc["outcomes"], cc["covariates"],
            fdr_alpha=am.get("fdr_alpha", 0.05))
        if len(pc):
            pc_path = utils.TABLES_ROOT / f"partial_corr_{tag}.csv"
            pc.to_csv(pc_path, index=False)
            logger.info("偏相关结果已保存: %s", pc_path)
            sig = pc[pc["significant"]]
            logger.info("FDR 显著关联 %d 项", len(sig))

        roc_rows = []
        for iad in iad_cols:
            roc = clinmod.roc_analysis(iad_df, iad, cc["high_risk_col"],
                                       cc["high_risk_cutoff"])
            if roc:
                roc_rows.append({k: v for k, v in roc.items()
                                 if k not in ("fpr", "tpr", "thresholds")})
        if roc_rows:
            roc_df = pd.DataFrame(roc_rows)
            roc_path = utils.TABLES_ROOT / f"roc_{tag}.csv"
            roc_df.to_csv(roc_path, index=False)
            logger.info("ROC 结果已保存: %s", roc_path)

    # ---- 5. 图表 ----
    try:
        from src import visualize as viz
        viz.make_all_figures(cfg, features, iad_df, band_names, tae_df,
                             tag=tag)
    except Exception as e:  # noqa: BLE001
        logger.warning("图表生成跳过: %s", e)

    logger.info("流水线完成。结果目录: %s", utils.RESULTS_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
