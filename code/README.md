# HBN 课题4 — 实验代码说明

## 概述

基于黎曼几何的脑网络拓扑异速生长（Riemannian Topological Allometry）实验流水线。
数据：HBN-EEG 静息态（Release 1-10，128通道，500Hz）。
支持 Windows（本机）与 Linux（工作站）双平台，一套代码直接运行。

## 目录结构

```
code/
├── run_all.py              # 一键启动脚本（完整流水线）
├── config/config.yaml      # 实验配置（双平台路径自动切换）
├── requirements.txt        # 依赖清单
└── src/
    ├── utils.py            # 配置加载、日志、路径解析
    ├── preprocess.py       # EEG读取(.set)、闭眼段提取、滤波
    ├── covariance.py       # Ledoit-Wolf协方差、黎曼均值、密度阈值化
    ├── network.py          # 图论指标：全局/局部效率、模块化、小世界
    ├── allometry.py        # 群体TAE(幂律拟合)、个体IAD
    ├── clinical.py         # CBCL偏相关、ROC分析
    ├── pipeline.py         # 主流水线：被试特征提取(可并行)
    └── visualize.py        # 图表生成（图3/4/6/9/12）
```

## 快速开始

```bash
# 完整运行（工作站）
python run_all.py

# 冒烟测试（本机小样本快速验证，约几分钟）
python run_all.py --smoke

# 自定义并行度 / 频带
python run_all.py --n-jobs 8 --band alpha beta

# 复用已有特征表（跳过耗时的特征提取）
python run_all.py --skip-features
```

## 配置说明（config/config.yaml）

| 项 | 说明 |
|----|------|
| `platform.windows_data_root` | 本机数据路径 |
| `platform.linux_data_root` | 工作站数据路径（自动检测系统切换） |
| `releases` | 留空=自动发现全部 HBN-Release* |
| `preprocess` | 滤波范围、闭眼段时长、窗口参数 |
| `bands` | 5个频带定义 |
| `network` | 协方差估计法、密度阈值、黎曼均值开关 |
| `allometry` | 年龄范围、分组边界、FDR阈值 |
| `clinical` | CBCL结局列、协变量、ROC截断 |
| `run` | 并行度、冒烟测试参数 |

## 输出

所有结果输出到 `results/`：

```
results/
├── logs/        run_<时间戳>.log（控制台日志，含时间戳）
├── tables/      subjects_features_*.csv, tae_summary_*.csv, iad_*.csv, partial_corr_*.csv, roc_*.csv
└── figures/     fig3_loglog_*.png, fig4_violin_*.png, fig6_radar.png, fig9_smallworld_*.png, fig12_roc.png
```

## 冒烟测试

```bash
python run_all.py --smoke
```

冒烟模式使用 `run.smoke_releases`（默认 Release1）+ `smoke_max_subjects`（默认6人）+ `smoke_bands`（默认 alpha/beta），
几分钟内验证全流水线（读取→滤波→协方差→网络→TAE→临床→图表）可运行。

## 工作站部署提示

1. 拷贝整个 `code/` 目录到工作站
2. 工作站数据放 `/mnt/data1/home/tanhuang/datasets/HBN`（或改 config.yaml 的 `linux_data_root`）
3. `pip install -r requirements.txt`
4. 建议用 `nohup python run_all.py > /dev/null 2>&1 &` 后台运行（日志已自动保存到 results/logs/）

## 注意

- 首次运行特征提取较慢（2500+被试 × 5频带 × 图论指标），建议工作站用 `--n-jobs` 开满核心
- `participants.tsv` 中 `ehq_total` 为共情-系统化商数，非 IQ；若需 IQ 需另找数据源（见文献笔记）
- 小世界属性计算含度保持随机化基准，每个被试每个频带约 n_random_graphs 次随机图

## 兼容性说明

**HBN 闭眼协议自适应**（`preprocess.py`）：自动兼容以下三种事件协议：
1. **早期 Release（R1-R5 等）**：使用 `instructed_toCloseEyes` 显式事件，5段闭眼各约40秒
2. **中期 Release**：同时存在 `resting_start` 和 `instructed_toCloseEyes`，优先匹配后者
3. **新 Release（R9-R10 等）**：只有 `resting_start`（睁眼开始），闭眼段自动取开头到 `resting_start` 前的38秒

**内存优化**：`preprocess.py` 读取时使用 `preload=False` + `crop()` 切出闭眼段，
避免每被试加载完整的 97MB EEG 文件，工作站 2480 被试内存压力大幅降低。

## 性能预估（工作站 16 核）

- 单被试单频带（5段闭眼）：约 5-6 秒
- 单被试全 5 频带：约 25-30 秒
- 2480 被试 × 5 频带（16 并行）：约 1-2 小时
- 2480 被试 × 5 频带（4 并行）：约 5-6 小时

进度日志每 10 个被试输出一次（`results/logs/`），可实时观察处理进度。
