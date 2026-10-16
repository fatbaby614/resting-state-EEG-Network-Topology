"""
pipeline.py — 主流水线：被试清单 -> 特征表
- 合并各 release 的 participants.tsv
- 筛选有 RestingState 数据的被试
- 逐被试：读EEG -> 分频带 -> 协方差 -> 网络 -> 拓扑指标
- 输出被试级特征表 subjects_features.csv
"""

import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import covariance as covmod
from . import network as netmod
from . import preprocess as premod
from .utils import FIGURES_ROOT, TABLES_ROOT

logger = logging.getLogger(__name__)

PHENO_COLS = ["age", "sex", "ehq_total", "p_factor", "attention",
              "internalizing", "externalizing"]


def load_participants(release_dirs, task="RestingState"):
    """
    读取所有 release 的 participants.tsv，合并、清洗。
    同一被试若跨 release 出现，只保留 release 编号最大的版本（最新采集）。
    返回 DataFrame：仅保留任务可用的被试。
    """
    frames = []
    for rd in release_dirs:
        pf = Path(rd) / "participants.tsv"
        if not pf.exists():
            continue
        df = pd.read_csv(pf, sep="\t", dtype={"participant_id": str})
        df["release"] = Path(rd).name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)

    # 任务可用性过滤
    if task in allp.columns:
        allp = allp[allp[task] == "available"]
    # 年龄有效
    allp["age"] = pd.to_numeric(allp.get("age"), errors="coerce")
    allp = allp[allp["age"].notna()]
    # 所需表型列保留
    for c in PHENO_COLS:
        if c not in allp.columns:
            allp[c] = np.nan

    # 同一被试跨 release：保留 release 编号最大者
    rel_num = allp["release"].str.extract(r"HBN-Release(\d+)", expand=False)
    allp["_rel_num"] = pd.to_numeric(rel_num, errors="coerce")
    allp = allp.sort_values("_rel_num", ascending=False).drop_duplicates(
        subset=["participant_id"], keep="first")
    allp = allp.drop(columns=["_rel_num"])
    return allp.reset_index(drop=True)


def _subject_work(args):
    """
    单被试处理（进程池可序列化）：返回 (participant_id, dict, err)。
    """
    set_path, cfg, band_names = args
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

        feats = {}
        for band in band_names:
            seg = bands_data[band]
            cov = covmod.estimate_covariance(
                seg, win_samp, overlap,
                method=nc["cov_estimator"], riemann_mean=nc.get("riemann_mean", True))
            adj = covmod.build_network(
                cov, threshold_mode=nc["threshold_mode"], density=nc["density"],
                use_abs=nc.get("use_abs", True), edge_weight=nc.get("edge_weight", "correlation"))
            m = netmod.compute_all_metrics(
                adj, n_random=nc.get("n_random_graphs", 20), rng_seed=42)
            for k, v in m.items():
                feats[f"{band}__{k}"] = v
        return set_path.parents[1].name, feats, None
    except Exception as e:  # noqa: BLE001
        logger.debug("被试处理失败 %s: %s", set_path, e)
        return None, None, str(e)
    finally:
        # 每个被试结束后强制 GC + 释放，防止 mne raw 驻留内存 → OOM
        gc.collect()


def build_subject_features(cfg, participants_df, release_dirs, band_names,
                           max_subjects=None, n_jobs=4):
    """
    逐被试计算拓扑特征，生成宽表。
    返回 DataFrame：participant_id, age, sex, ...pheno, 各频带指标。
    """
    # 构造任务列表：找到每个被试的 RestingState .set 文件
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
        tasks.append((set_path, cfg, band_names))
        if max_subjects and len(tasks) >= max_subjects:
            break

    logger.info("待处理被试数: %d", len(tasks))

    results = {}
    failures = []
    n_total = len(tasks)

    if n_jobs <= 1 or len(tasks) <= 1:
        for i, t in enumerate(tasks):
            sid, feats, err = _subject_work(t)
            if sid and feats:
                results[sid] = feats
            elif err:
                failures.append((str(t[0]), err))
            if (i + 1) % 10 == 0 or (i + 1) == n_total:
                logger.info("[%d/%d] 已处理 %d, 成功 %d, 失败 %d",
                            i + 1, n_total, i + 1, len(results), len(failures))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_subject_work, t): t for t in tasks}
            for i, f in enumerate(as_completed(futs)):
                sid, feats, err = f.result()
                if sid and feats:
                    results[sid] = feats
                elif err:
                    failures.append((str(futs[f]), err))
                if (i + 1) % 10 == 0 or (i + 1) == n_total:
                    logger.info("[%d/%d] 已处理 %d, 成功 %d, 失败 %d",
                                i + 1, n_total, i + 1, len(results), len(failures))

    # 输出失败原因（最多前10个），便于诊断
    if failures:
        logger.warning("失败被试样本(最多10个):")
        for fn, err in failures[:10]:
            logger.warning("  %s: %s", Path(fn).parent.parent.name, err[:200])

    logger.info("成功处理被试数: %d", len(results))

    if not results:
        return pd.DataFrame()

    feat_df = pd.DataFrame.from_dict(results, orient="index")
    feat_df.index.name = "participant_id"
    feat_df = feat_df.reset_index()

    # 合并表型
    meta = participants_df[["participant_id"] + PHENO_COLS].copy()
    meta = meta.merge(feat_df, on="participant_id", how="inner")

    # 年龄范围过滤
    am = cfg["allometry"]
    meta = meta[(meta["age"] >= am["min_age"]) & (meta["age"] <= am["max_age"])]
    return meta.reset_index(drop=True)


def save_subject_features(df, tag=""):
    """保存特征表到 results/tables/"""
    fname = TABLES_ROOT / f"subjects_features{('_' + tag) if tag else ''}.csv"
    df.to_csv(fname, index=False)
    logger.info("特征表已保存: %s (n=%d)", fname, len(df))
    return fname
