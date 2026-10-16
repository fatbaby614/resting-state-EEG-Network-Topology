# Allometric scaling of resting-state EEG network topology across childhood and adolescence

Analysis code for *"Allometric scaling of resting-state EEG network topology across childhood and adolescence: a descriptive power-law analysis"* (submitted to *Brain Topography*, Springer Nature).

This repository implements a descriptive power-law analysis of graph-theoretic
brain network topology derived from resting-state EEG, across childhood and
adolescence. It covers the full pipeline from raw EEG to allometric scaling
exponents, together with the robustness and sensitivity analyses reported in
the manuscript.

---

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Data access](#data-access)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [Analysis scripts](#analysis-scripts)
- [Method parameters](#method-parameters)
- [Outputs](#outputs)
- [Figures](#figures)
- [Results at a glance](#results-at-a-glance)
- [Citation](#citation)
- [How to cite the data](#how-to-cite-the-data)
- [License](#license)

---

## Overview

The analysis asks a simple question with an unusually explicit answer: **how
does EEG network topology scale with age between 5 and 21 years, and is a
power law a sufficient description?**

The pipeline builds a brain network for every subject in every frequency band,
computes four graph-theoretic indices, and fits an allometric power law of
each index on age. Robustness checks cover release composition, age-distribution
balance, threshold density, node count, and the treatment of numerically
degenerate networks.

**Key design points**

- **Riemannian covariance pooling.** Sliding-window covariances are pooled with
  a log-Euclidean (Riemannian) mean rather than an arithmetic mean, which is
  markedly less sensitive to window-level outliers.
- **Shrinkage covariance.** Each window uses a Ledoit–Wolf regularised
  covariance estimate, which is well-conditioned at 128 channels and 2-second
  windows.
- **Primary sample excludes degenerate networks.** Ten subjects produced indices
  that are numerically zero under thresholding. The primary analysis drops these
  subjects at the subject level; sensitivity analyses retaining them are also
  provided, and the conclusions do not change.
- **Honest model comparison.** Power-law fits are compared against four
  alternatives (linear and quadratic in raw space, quadratic and a penalised
  spline GAM in log space) by AIC, BIC, and five-fold cross-validated RMSE.
  The power law is reported as *parsimonious*, not *uniquely superior* — raw-space
  cross-validation never preferred it, which the manuscript states explicitly.

---

## Repository layout

```
.
├── run_all.py                          end-to-end pipeline
├── run_revision_0922.py                primary-sample recomputation + cross-checks
├── run_a1_model_comparison.py          AIC / BIC / cross-validation
├── run_a2_release_robustness.py        release covariates + leave-one-release-out
├── run_a3_lobe_interaction.py          lobe x age tests
├── run_a4_age_balance.py               age-distribution reweighting
├── run_a5_node_count.py                node-count sensitivity
├── run_density_sweep.py                density-threshold refits
├── run_lobe_allometry.py               lobe-resolved fits
├── run_degenerate_network_sensitivity.py
├── gen_iad_primary_0922.py
├── gen_tables_0922.py
├── regen_paper_figures.py              publication figures at 600 dpi
├── config/config.yaml                  all analysis parameters
├── src/
│   ├── utils.py                        config, platform paths, logging
│   ├── preprocess.py                   EEG reading, closed-eye extraction, filtering
│   ├── covariance.py                   Ledoit-Wolf, Riemannian mean, thresholding
│   ├── network.py                      graph indices
│   ├── allometry.py                    power-law fits, individual deviations
│   ├── clinical.py                     partial correlation, ROC
│   ├── pipeline.py                     per-subject feature extraction
│   ├── visualize.py                    exploratory plots
│   └── lobe_map.json                   128 electrodes to 5 lobes
└── requirements.txt
```

`run_a*.py` and `check_lobe_progress.py` write to `results/` relative to the
parent directory of this folder, matching the layout used during development.

---

## Data access

**This repository contains source code only. No data are redistributed.**

The analyses use resting-state EEG from the Healthy Brain Network (HBN-EEG),
Releases 1–10 (OpenNeuro accessions `ds005505`–`ds005512`, `ds005514`,
`ds005515`); Release 11 was not included. Each release is expected as a
directory named `HBN-Release<N>-ds<accession>` containing per-subject `.set`
files and a `participants.tsv`.

HBN data are available under a Data Use Agreement. To request access, apply
through the HBN portal:

<https://www.hbnconsortium.org/>

The expected directory layout is:

```
<data_root>/
├── HBN-Release1-ds005505/
│   ├── sub-XXXX/ses-YYYY/eeg/*.set
│   └── participants.tsv
├── HBN-Release2-ds005506/
└── ...
```

Point `platform.windows_data_root` / `platform.linux_data_root` in
`config/config.yaml` at the directory that contains the release folders.

---

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/fatbaby614/resting-state-EEG-Network-Topology.git
cd resting-state-EEG-Network-Topology
pip install -r requirements.txt
```

---

## Quick start

Run the smoke test first. It exercises the entire pipeline on a handful of
subjects and two bands, and finishes in a few minutes:

```bash
python run_all.py --smoke
```

Then run the full analysis:

```bash
# full pipeline, automatic parallelism
python run_all.py

# explicit control
python run_all.py --n-jobs 8
python run_all.py --band alpha beta

# reuse an existing feature table and skip the expensive extraction step
python run_all.py --skip-features
```

For long runs on a workstation, background the process; logs are written
automatically:

```bash
nohup python run_all.py > /dev/null 2>&1 &
```

---

## Pipeline

```
EEG (.set)
  └─ closed-eye segment extraction      src/preprocess.py
       band-pass 0.5–45 Hz, notch, 2 s windows at 50% overlap
       (adaptive to three HBN eyes-closed annotation protocols)
  └─ per-window covariance              src/covariance.py
       Ledoit–Wolf shrinkage → log-Euclidean Riemannian mean
  └─ threshold to 20% strongest edges
  └─ graph-theoretic indices            src/network.py
       global efficiency, local efficiency, modularity, small-worldness
       (small-worldness against degree-preserving null models)
  └─ subject × band feature table       src/pipeline.py
  └─ population allometry               src/allometry.py
       power-law fit of each index on age; individual deviations (IAD)
  └─ clinical association               src/clinical.py
       partial correlations with CBCL scores; ROC for risk discrimination
  └─ figures and tables                 src/visualize.py
```

---

## Analysis scripts

### Core pipeline

| Script | Purpose |
|---|---|
| `run_all.py` | End-to-end pipeline entry point |
| `run_revision_0922.py` | Recomputes every reported quantity on the primary sample (excludes degenerate networks) and cross-checks against the full-sample tables |
| `gen_iad_primary_0922.py` | Regenerates IAD tables and ROC on the primary sample |
| `gen_tables_0922.py` | Regenerates age-band, release, and inclusion/exclusion tables |

### Robustness and sensitivity analyses

| Script | Purpose |
|---|---|
| `run_a1_model_comparison.py` | AIC / BIC / five-fold CV comparison of the power law against four alternatives |
| `run_a2_release_robustness.py` | Covariate-adjusted refits and leave-one-release-out (LORO) estimation |
| `run_a3_lobe_interaction.py` | Lobe × age omnibus tests and pairwise contrasts (HC3 covariance, BH-FDR) |
| `run_a4_age_balance.py` | KDE inverse-density reweighting and age-stratified bootstrap |
| `run_a5_node_count.py` | Node-count sensitivity for the lobe-resolved fits |
| `run_density_sweep.py` | Refits at 10–30% density thresholds |
| `run_lobe_allometry.py` | Lobe-resolved power-law fits (five lobes × five bands × two indices) |
| `run_degenerate_network_sensitivity.py` | Full-sample comparison retaining degenerate networks |
| `regen_paper_figures.py` | Redraws every manuscript figure from the result tables at publication size and 600 dpi |

---

## Method parameters

Defaults from `config/config.yaml`:

| Parameter | Value |
|---|---|
| Frequency bands | δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–45 Hz |
| Filter | 0.5–45 Hz band-pass, 60 Hz notch |
| Closed-eye window | 38 s extracted from each ~40 s block, first 2 s discarded |
| Covariance window | 2 s, 50% overlap |
| Covariance estimator | Ledoit–Wolf |
| Window pooling | log-Euclidean (Riemannian) mean |
| Threshold | density, 20% strongest edges |
| Null models | 20 degree-preserving random graphs per subject per band |
| Age range | 5–21 years |
| FDR | Benjamini–Hochberg, α = 0.05 |
| Clinical covariates | sex |

---

## Outputs

```
results/
├── logs/       run_<timestamp>.log
├── tables/     *.csv  (all quantities reported in the manuscript)
└── figures/    *.png
```

The `*_primary.csv` files correspond to the primary sample (n = 1,957);
unsuffixed files correspond to the full audited sample including degenerate
networks. The scripts recompute the published full-sample values and assert
agreement with the stored tables before writing primary-sample outputs, which
guards against silent changes to the fitting recipe.

---

## Figures

`regen_paper_figures.py` draws all figures at their final physical size
(3.3 in single column, 6.8 in double column) so that in-figure text stays at or
above 7.5 pt, and exports 600 dpi PNG on white:

| Output | Content |
|---|---|
| `fig1_concept.png` | Schematic of the network construction and the four indices |
| `fig3_loglog_grid_primary.png` | log–log age–index scatter with power-law fits, 5 bands × 2 indices |
| `fig4_trajectory_theta.png` | Age-binned distribution of the theta-band trajectory |
| `fig12_roc_primary.png` | ROC curves for externalizing-risk discrimination |
| `fig_density_sweep.png` | Exponent stability across density thresholds |
| `fig_lobe_grid_primary.png` | 5 bands × 2 indices lobe-resolved exponent grid |
| `fig_lobe_pair_gamma.png`, `fig_lobe_pair_alpha.png` | Lobe comparison panels |

Brain topographies are drawn from true lobe-level exponents rather than
128-electrode interpolation, since interpolated maps of this kind are prone to
misleading spatial detail.

---

## Results at a glance

- **1,957 participants** aged 5–21 years in the primary analysis (2,471 audited,
  1,967 passed quality control).
- **19 of 20** band-by-index exponents are negative; **15** survive FDR
  correction. The steepest is δ small-worldness (α = −0.878), implying a near
  halving per doubling of age.
- **Explained variance** ranges from 0.02% to 25.3%.
- Exponents are robust to release-related covariates, age-balance
  reweighting, threshold density, and node count.
- Within log space, information criteria prefer quadratic and spline models,
  and raw-space cross-validation never preferred the power law. Across this
  narrow age window the power law is a **parsimonious description rather than a
  uniquely superior fit**.
- **4 of 10** region-by-age interactions survive correction; regional declines
  are broadly pan-lobar.
- Deviations from the trajectory correlate only weakly with Child Behavior
  Checklist scores, and risk discrimination is at chance (AUC ≈ 0.50).

---

## Citation

If you use this code, please cite the associated manuscript:

> Li X, Tan H, Cai G, Zhang R, Zhang Q, Mai H, Liu Y, Zhang Q, Xie G, Zhang L.
> *Allometric scaling of resting-state EEG network topology across childhood and
> adolescence: a descriptive power-law analysis.*
> Brain Topography (Springer Nature).

Corresponding authors: Gaozhan Xie (`iamgzxie@njupt.edu.cn`) and Li Zhang
(`zhanglith@163.com`).

---

## How to cite the data

This code depends on the Healthy Brain Network dataset. Please cite the HBN
resource itself:

> Alexander LM, Escalera J, Ai L, et al. An open resource for transdiagnostic
> research in pediatric mental health and learning disorders. *Scientific Data*.
> 2017;4:170181. <https://doi.org/10.1038/sdata.2017.181>

and the HBN-EEG data release:

> Shirazi SY, Franco A, Hoffmann MS, et al. HBN-EEG: The FAIR implementation of
> the Healthy Brain Network EEG study. *bioRxiv*.
> <https://doi.org/10.1101/2024.10.03.615261>

Always follow the citation requirements in the HBN Data Use Agreement for the
specific release you use.

---

## License

Code is released under the [MIT License](LICENSE).

**The data are not.** HBN-EEG and all derived participant-level products
remain subject to the HBN Data Use Agreement and must be obtained by
application. See the data notice at the end of the `LICENSE` file.
