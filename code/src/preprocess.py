"""
preprocess.py — EEG 数据预处理
- 读取单个被试 RestingState 的 .set (EEGLAB/BIDS) 文件
- 提取闭眼段（兼容多种 HBN 事件协议）
- 陷波 + 带通滤波
- 按频带切分，输出 (频带 -> 闭眼段时间序列) 字典

HBN 数据集不同 Release 闭眼协议不同（事件名称见下），
本模块统一返回闭眼段列表，保证后续分析一致。
"""

import logging
import warnings
from pathlib import Path

import mne
import numpy as np

logger = logging.getLogger(__name__)

# HBN-RestingState 不同版本的事件名集合：
#   - 早期 Release：使用 instructed_toCloseEyes(闭眼开始) + instructed_toOpenEyes(睁眼开始)
#   - 较新 Release：只有 resting_start(任务开始)，闭眼段需要用 alpha 功率特征推断
# 优先级：先尝试显式闭眼事件，没有再退化到 alpha 功率推断。
CLOSED_EYE_EVENTS = ["instructed_toCloseEyes", "closeEyes",
                     "eyes_closed", "EC", "30"]
RESTING_START_EVENTS = ["resting_start", "rs_start", "restingStart",
                        "90", "resting"]

MIN_CLOSED_EYE_DURATION = 5.0  # 闭眼段最短时长（秒），低于此视为无效


def _match_event(event_id, candidates):
    """在 event_id 字典中查找任一候选事件名（精确匹配优先，子串匹配兜底）"""
    for name in candidates:
        if name in event_id:
            return event_id[name], name
    for name, code in event_id.items():
        n = str(name).strip()
        for cand in candidates:
            if cand.lower() in n.lower() or n.lower() in cand.lower():
                return code, name
    return None, None


def _alpha_psd_windows(raw, window_sec=40):
    """
    按 window_sec 秒分窗，计算每窗的 alpha(8-13Hz) 功率均值。
    返回 (window_times_s, alpha_psds)。用于推断闭眼段。
    内存优化：仅取 top-2 候选窗口后立即删除中间数据，限制峰值 RAM。
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = raw.get_data(picks="eeg")
            sfreq = raw.info["sfreq"]
    except Exception as e:  # noqa: BLE001
        logger.debug("获取数据失败: %s", e)
        return [], []

    n = int(data.shape[1])
    win_s = int(window_sec * sfreq)
    if n < win_s:
        return [], []

    nwin = n // win_s
    alpha_vals = []
    win_times = []
    nperseg = min(512, win_s)
    for i in range(nwin):
        seg = data[:, i*win_s:(i+1)*win_s]
        try:
            psd, ff = mne.time_frequency.psd_array_welch(
                seg, sfreq, n_fft=nperseg, n_overlap=nperseg//2)
            ia = np.where((ff >= 8) & (ff <= 13))[0]
            if len(ia) == 0:
                alpha_vals.append(0.0)
            else:
                alpha_vals.append(float(psd[:, ia].mean().mean()))
            win_times.append(i * window_sec)
            # 立即释放单窗口 PSD（否则上百个窗口会驻留内存）
            del psd, ff, seg
        except Exception:  # noqa: BLE001
            alpha_vals.append(0.0)
            win_times.append(i * window_sec)

    # 关键：释放 data 引用，避免跨被试堆积
    del data
    return win_times, alpha_vals


def _find_eyes_closed_by_alpha(raw, duration_sec, trim_sec):
    """
    无显式事件时：用 alpha 功率最高窗口近似闭眼段。
    闭眼时 alpha(8-13Hz) 功率显著高于睁眼，这是可靠的生理代理指标。
    返回 1-2 个闭眼段（取 alpha 最高的 top-2 窗口）。
    """
    win_times, alpha_vals = _alpha_psd_windows(raw, window_sec=40)
    if len(alpha_vals) == 0:
        return []
    sfreq = raw.info["sfreq"]
    trim_samp = int(trim_sec * sfreq)
    dur_samp = int(duration_sec * sfreq)

    # 排序：alpha 功率从高到低
    order = np.argsort(alpha_vals)[::-1]
    segs = []
    for rank in range(min(2, len(order))):
        i = order[rank]
        start = int(win_times[i] * sfreq) + trim_samp
        end = start + dur_samp
        if end <= int(raw.n_times):
            segs.append((start, end))
    if segs:
        segs = sorted(segs, key=lambda x: x[0])  # 按时间排序，供 crop 使用
        logger.info("无显式闭眼事件，用 alpha 功率 top-%d 窗口推断闭眼段",
                    len(segs))
    return segs


def _find_eyes_closed_segments(raw, eyes_closed_event,
                                duration_sec, trim_sec):
    """
    提取闭眼段，返回 [(start_samp, end_samp), ...]。

    协议优先级：
      1. 显式闭眼事件（instructed_toCloseEyes 等）
      2. alpha 功率特征推断（用于只有 resting_start 的新 Release）
      3. 无闭眼事件可用 → 跳过该被试
    """
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    except Exception as e:  # noqa: BLE001
        logger.debug("events_from_annotations 失败: %s", e)
        return []

    avail = list(event_id.keys())
    sfreq = raw.info["sfreq"]
    n_total = int(raw.n_times)
    segs = []
    min_dur_samp = int(MIN_CLOSED_EYE_DURATION * sfreq)

    # 协议1：显式闭眼事件
    code, matched = _match_event(event_id,
                                 [eyes_closed_event] + CLOSED_EYE_EVENTS)
    if code is not None:
        trim_samp = int(trim_sec * sfreq)
        dur_samp = int(duration_sec * sfreq)
        for ev in events:
            if ev[2] == code:
                start = ev[0] + trim_samp
                end = start + dur_samp
                if 0 <= start < end <= n_total:
                    segs.append((start, end))
        if segs:
            logger.debug("协议1(%s)闭眼段 %d 段", matched, len(segs))
            return segs
        logger.debug("事件 %s 存在但无可用段，尝试协议2", matched)

    # 协议2：用 alpha 功率特征推断（无显式闭眼事件）
    segs = _find_eyes_closed_by_alpha(raw, duration_sec, trim_sec)
    if segs:
        return segs

    logger.warning("无闭眼事件且无法用 alpha 推断，跳过。事件: %s", avail)
    return []


def read_subject_eeg(set_path, l_freq, h_freq, notch_filter,
                     eyes_closed_event, eyes_closed_duration, eyes_closed_trim,
                     reject_bad_channels=True, drop_channels=("Cz",)):
    """
    读取单个被试的 .set 文件，返回：预处理后的 raw 和闭眼段样本索引列表。
    　　
    默认剔除参考通道 Cz（数据为 128 EEG + Cz 参考，需 128x128 网络）。
    若读取/事件提取失败，返回 (None, [])。

    内存优化：先用 preload=False 读取元数据定位闭眼段，
    再用 preload=True + crop() 只加载闭眼段数据（97MB → ~20MB）。
    """
    # 1. 元数据读取（不加载数据）
    try:
        raw_meta = mne.io.read_raw_eeglab(set_path, preload=False,
                                          verbose="ERROR")
    except Exception as e:  # noqa: BLE001
        logger.error("读取失败 %s: %s", Path(set_path).name, e)
        return None, []

    # 2. 剔除参考通道（元数据级别即可）
    drop = [c for c in drop_channels if c in raw_meta.ch_names]
    if drop:
        raw_meta.drop_channels(drop)

    # 3. 定位闭眼段
    segs_meta = _find_eyes_closed_segments(raw_meta, eyes_closed_event,
                                           eyes_closed_duration,
                                           eyes_closed_trim)
    if not segs_meta:
        logger.warning("无闭眼段: %s", Path(set_path).name)
        return None, []

    # 4. 加载闭眼段（crop 到闭眼段范围，含 2 秒缓冲）
    buf_samp = int(2.0 * raw_meta.info["sfreq"])
    crop_start = max(0, segs_meta[0][0] - buf_samp)
    crop_end = min(int(raw_meta.n_times),
                   segs_meta[-1][1] + buf_samp)
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        raw.drop_channels([c for c in drop_channels if c in raw.ch_names])
    except Exception as e:  # noqa: BLE001
        logger.error("加载失败 %s: %s", Path(set_path).name, e)
        return None, []

    # 用实际加载的 raw.n_times 重新 clamp，避免 raw_meta 与 raw 的 n_times 微差导致 tmax 越界
    # 用 sfreq 计算 max time 而非 n_times/sfreq，避免浮点误差
    max_t = (raw.n_times - 1) / raw.info["sfreq"]
    crop_end = min(int(raw.n_times) - 1, crop_end)
    crop_start = max(0, crop_start)
    raw.crop(tmin=crop_start / raw.info["sfreq"],
             tmax=min(crop_end / raw.info["sfreq"], max_t - 1e-3))
    del raw_meta

    # 5. 把元数据级样本索引折算到 crop 后的 raw 上
    shift = -crop_start
    segs = [(s + shift, e + shift) for s, e in segs_meta]
    n = int(raw.n_times)
    segs = [(s, e) for s, e in segs if 0 <= s < n and 0 < e <= n]
    if not segs:
        return None, []

    # 6. 坏通道
    if reject_bad_channels:
        try:
            raw = _drop_bad_channels(raw)
        except Exception as e:  # noqa: BLE001
            logger.debug("坏通道处理失败: %s", e)

    # 7. 陷波(工频60Hz, 美标数据)
    if notch_filter:
        try:
            raw.notch_filter(freqs=[60.0], picks="eeg", verbose="ERROR")
        except Exception:  # noqa: BLE001
            logger.debug("60Hz陷波失败，跳过")

    # 8. 带通
    try:
        raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg",
                   fir_design="firwin", verbose="ERROR")
    except Exception as e:  # noqa: BLE001
        logger.error("滤波失败 %s: %s", Path(set_path).name, e)
        return None, []

    return raw, segs


def _drop_bad_channels(raw):
    """
    基于幅度方差剔除坏通道：通道标准差 > median + 3*MAD 视为坏通道。
    """
    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        return raw
    data = raw.get_data(picks=picks)
    std = np.std(data, axis=1)
    med = np.median(std)
    mad = np.median(np.abs(std - med))
    thr = med + 3 * (1.4826 * mad) if mad > 0 else np.inf
    bad = np.where(std > thr)[0]
    if len(bad) > 0 and len(bad) < len(picks) * 0.3:
        bad_names = [raw.ch_names[picks[i]] for i in bad]
        raw.info["bads"].extend(bad_names)
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw


def split_bands(raw, segs, bands, band_names):
    """
    按频带滤波并提取闭眼段数据。
    返回 {band_name: np.ndarray (n_segments, n_channels, n_samples)}。
    segs 内各段长度可能不同，用最小长度对齐。
    """
    if not segs:
        return {}
    seg_lens = [e - s for s, e in segs]
    min_len = min(seg_lens)
    segs = [(s, s + min_len) for s, _ in segs]

    out = {}
    for name in band_names:
        if name not in bands:
            continue
        l_f, h_f = bands[name]
        raw_b = raw.copy().filter(l_freq=l_f, h_freq=h_f, picks="eeg",
                                  fir_design="firwin", verbose="ERROR")
        data = raw_b.get_data(picks="eeg")
        seg_data = np.stack([data[:, s:e] for s, e in segs], axis=0)
        out[name] = seg_data
    return out