# Calibration and Uncertainty of Molecular Property Prediction Models under Distribution Shift

![Status](https://img.shields.io/badge/Project_Status-🚧_In_Progress_/_Research_Phase-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-ee4c2c)
![Domain](https://img.shields.io/badge/Domain-Cheminformatics-2ea44f)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

> **Research question.** *How should uncertainty and calibration for molecular property prediction be evaluated under chemical distribution shift, and which conclusions survive after correcting common evaluation mistakes?*

---

## Overview

This project develops an evaluation framework for assessing the reliability of **Graph Neural Networks (GNNs)** for molecular property prediction under **out-of-distribution (OOD) distribution shift**. The focus is on producing trustworthy **uncertainty quantification (UQ)** and **calibration** estimates through methodologically sound choices: consistent per-task multitask pooling, stable MC-Dropout inference (with frozen BatchNorm statistics), exchangeability-preserving conformal calibration, and non-negative Risk–Coverage integration. The framework is exercised on a 108-cell experimental grid spanning two datasets, two OOD shift types, three architectures, and three UQ methods.

---

## Key Findings (Current State)

The **Pretrained-GIN Deep Ensemble** is the strongest configuration overall, winning three of four `(dataset, split)` cells on ROC-AUC and leading on AURC in those same cells. Predictive variance is shown to be an actionable error-detection signal, with **AUC-of-error reaching 0.7524 ± 0.0085** on the BBBP scaffold split (i.e., misclassified molecules are systematically assigned higher predictive variance than correctly classified ones).

**Best configuration per `(dataset, split)` cell** — mean ± std over three seeds (`ddof=1`):

| Dataset | Split    | Architecture   | UQ       | ROC-AUC ↑       | ECE ↓           | AURC ↓          | AUC-err ↑       |
|:--------|:---------|:---------------|:---------|:----------------|:----------------|:----------------|:----------------|
| BBBP    | Scaffold | Pretrained-GIN | Ensemble | 0.9001 ± 0.0078 | 0.0406 ± 0.0125 | 0.0462 ± 0.0061 | 0.7524 ± 0.0085 |
| BBBP    | MW       | Pretrained-GIN | Ensemble | 0.8551 ± 0.0028 | 0.1221 ± 0.0260 | 0.1209 ± 0.0092 | 0.7477 ± 0.0551 |
| Tox21   | Scaffold | Pretrained-GIN | Ensemble | 0.7727 ± 0.0243 | 0.0287 ± 0.0017 | 0.0403 ± 0.0045 | 0.7120 ± 0.0332 |
| Tox21   | MW       | D-MPNN         | Ensemble | 0.6930 ± 0.0068 | 0.0469 ± 0.0018 | 0.0692 ± 0.0024 | 0.6595 ± 0.0130 |

Additional observations:
- **Ensembling** improves ROC-AUC in 10 of 12 architecture triples; its calibration benefit (ECE reduction) is more variable and architecture-dependent.
- **MC-Dropout** is competitive with Deep Ensembles on calibration in some settings (e.g., Tox21 scaffold), but is weaker on discrimination and incurs a `T = 50` inference-time multiplier.
- Reported standard deviations capture **training stability** (initialization, shuffling, ensemble diversity), *not* split variability, because both the Scaffold and MW partitions are deterministic by construction.

> ⚠️ **These results are preliminary.** Several conclusions are not yet robust to the evaluation corrections listed in the [Development Roadmap](#development-roadmap). In particular, the Pretrained-GIN advantage is currently confounded with model capacity.

---

## Methodology

**Stack**

| Component            | Tooling                                                          |
|:--------------------|:-----------------------------------------------------------------|
| Deep learning        | PyTorch, PyTorch Geometric (`GINEConv`, `JumpingKnowledge`, global pooling) |
| Cheminformatics      | RDKit (featurization, Bemis–Murcko scaffolds, descriptors)       |
| Datasets             | MoleculeNet — Tox21 (12 toxicity endpoints), BBBP                |
| Metrics / analysis   | scikit-learn, NumPy, pandas, Matplotlib                          |

**Architectures evaluated:** Pretrained-GIN (300-dim / 5-layer, `contextpred` checkpoint), GIN-JK (128-dim / 3-layer), D-MPNN (directed bond-centric message passing).

**Distribution shifts:** Scaffold split (greedy Bemis–Murcko bucket-packing) and Molecular-Weight tail split (heaviest 10% → test).

**Evaluation protocol**
- **Per-task masked evaluation.** Masked binary cross-entropy over valid labels, with all metrics computed per task and then **macro-averaged**, avoiding pooling bias on sparse/imbalanced endpoints.
- **Expected Calibration Error (ECE).** Macro-averaged per-task to keep hard endpoints visible.
- **Risk–Coverage & AURC.** Brier-score-based Risk–Coverage curves with non-negative trapezoidal AURC integration; samples ranked by MCP (deterministic) or negative predictive variance (UQ-aware).
- **AUC-of-error.** AUROC of per-sample predictive variance treated as a classifier of the binary error indicator.
- **Split conformal prediction.** Finite-sample marginal coverage via the Angelopoulos–Bates recipe, with exchangeability preserved between calibration and test predictions.

The conformal quantile uses the standard finite-sample correction:

$$
\hat{q} = Q_{\lceil (n_{\text{cal}}+1)(1-\alpha) \rceil / n_{\text{cal}}}\big(\{s_i\}_{i=1}^{n_{\text{cal}}}\big), \qquad s_i = 1 - \hat{p}(y_i \mid x_i)
$$

---

## Development Roadmap

> The following roadmap operationalizes the reviewer feedback. It is the active to-do list for moving the project from *"we compared several uncertainty methods"* to a defensible claim about evaluation under chemical distribution shift.

### 🧪 Expanding Baselines & Evidence
- [ ] Add an **official Chemprop / D-MPNN** implementation as a reference baseline.
- [ ] Add a **modern graph transformer / GraphGPS-style** architecture.
- [ ] Add **at least one stronger pretrained molecular model**.
- [ ] Add **simple strong baselines**: Morgan fingerprints + RF/XGBoost, with post-hoc calibration.
- [ ] **Deconfound Pretrained-GIN.** Train a **matched-capacity, randomly-initialized GIN** (same 300-dim / 5-layer geometry) to isolate the effect of *pretraining* from *capacity/depth*.

### 📊 Statistical Rigor
- [ ] Increase from **3 seeds to ≥ 5 seeds** (preferably more).
- [ ] Report **confidence intervals** and/or **bootstrap analysis** for all headline metrics.
- [ ] Document explicitly that current SDs reflect **training randomness**, not split variability (Scaffold/MW partitions are near-deterministic).
- [ ] Add **split resampling** (Monte Carlo scaffold assignment / k-fold over the MW tail) to estimate true OOD-split variance.

### 🔧 Methodological Fixes
- [ ] **Separate the conformal calibration set** from the early-stopping validation set (currently shared, weakening coverage claims).
- [ ] Reframe split-conformal **under-coverage under non-exchangeable shift** as a *finding*, not merely a limitation.
- [ ] Add **Mondrian / class-conditional** and **weighted conformal** baselines for non-exchangeable settings.
- [ ] Compute **AUC-of-error for deterministic baselines** using MCP / entropy / margin signals; report **variance-based** AUC-of-error separately for MC-Dropout and ensembles.

### 📈 Reporting & Reproducibility
- [ ] Add a **Tox21 per-task appendix**: per-task label counts, missing rates, positive rates, ROC-AUC, PR-AUC, ECE, Brier, NLL, AURC, AUC-of-error, and conformal coverage.
- [ ] **Verify the per-task CSV contains no duplicated rows.**
- [ ] Ensure **main figures use official macro-averaged, multi-seed metrics**.
- [ ] Remove **pooled single-seed values** from plot titles (keep them only in figures explicitly demonstrating pooling bias).

---



## References

1. Wu et al. (2018). *MoleculeNet: A benchmark for molecular machine learning.* Chemical Science.
2. Yang et al. (2019). *Analyzing learned molecular representations for property prediction (D-MPNN).* JCIM.
3. Hu et al. (2020). *Strategies for pre-training graph neural networks.* ICLR.
4. Angelopoulos & Bates (2023). *Conformal prediction: A gentle introduction.* FnT in ML.
5. Tibshirani et al. (2019). *Conformal prediction under covariate shift.* NeurIPS.
6. Bemis & Murcko (1996). *The properties of known drugs. 1. Molecular frameworks.* J. Med. Chem.
7. Lakshminarayanan et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS.
8. Gal & Ghahramani (2016). *Dropout as a Bayesian approximation.* ICML.
