"""
utils.py — 公共工具：配置加载、平台路径、日志、目录创建
"""

import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # code/
CODE_ROOT = PROJECT_ROOT
WORKSPACE_ROOT = PROJECT_ROOT.parent  # d:\TanHuangWork\HBN4
RESULTS_ROOT = WORKSPACE_ROOT / "results"
DATA_PROCESSED_ROOT = WORKSPACE_ROOT / "data" / "processed"
FIGURES_ROOT = RESULTS_ROOT / "figures"
TABLES_ROOT = RESULTS_ROOT / "tables"
LOGS_ROOT = RESULTS_ROOT / "logs"


def load_config(config_path=None):
    """加载 config.yaml"""
    if config_path is None:
        config_path = CODE_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_data_root(cfg):
    """根据操作系统选择数据根目录，并验证存在"""
    if os.name == "nt":
        root = Path(cfg["platform"]["windows_data_root"])
    else:
        root = Path(cfg["platform"]["linux_data_root"])
    if not root.exists():
        logging.warning("数据根目录不存在: %s (本机冒烟测试可能失败)", root)
    return root


def discover_releases(data_root):
    """发现数据根目录下所有 HBN-Release* 目录，返回排序列表"""
    if not data_root.exists():
        return []
    releases = sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("HBN-Release")]
    )
    return releases


def resolve_releases(cfg, data_root):
    """解析要处理的 release 列表：配置优先，否则自动发现"""
    rels = cfg.get("releases") or []
    if rels:
        return [data_root / r for r in rels]
    return discover_releases(data_root)


def setup_logging(log_level="INFO", log_file=None):
    """初始化日志：控制台 + 可选文件"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def ensure_dirs():
    """确保结果目录存在"""
    for d in [RESULTS_ROOT, FIGURES_ROOT, TABLES_ROOT, LOGS_ROOT, DATA_PROCESSED_ROOT]:
        d.mkdir(parents=True, exist_ok=True)


def get_platform_name():
    return "windows" if os.name == "nt" else "linux"
