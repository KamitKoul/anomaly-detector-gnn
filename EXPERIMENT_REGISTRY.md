# EXPERIMENT REGISTRY: Complete Chronological Experiment History

**Generated:** October 30, 2026  
**Repository:** `anomaly-detector-gnn`  
**All metrics sourced from actual experiment output files**

---

## Experiment Registry Format

| Experiment ID | Date | Architecture | Dataset | Training Events | Evaluation Events | Configuration | Result | Status |
|---|---|---|---|---|---|---|---|---|

---

## Phase 0: Pre-Experiment Baseline (Frozen at tag `pre-experiments-684c99e`)

| Exp ID | Date | Architecture | Dataset | Train Events | Eval Events | Config | Result | Status |
|---|---|---|---|---|---|---|---|---|
| BASE-001 | Pre-2026 | NodeGNN (discovery_model) | LHCO2020 | 5,000 | 500 bg / 500 sig | λ_MET=0.1, no adv | ROC AUC=0.6414, Sep=1.13x, Sig=0.37σ | **Frozen baseline** |
| BASE-002 | Pre-2026 | BaselineMLP (autoencoder_bg) | LHCO2020 | 5,000 | 500 bg / 500 sig | Flat 64D | Reference tabular | **Frozen baseline** |

**Source:** `BASELINE.md`, `discovery_exhibition_proof.png`, `discovery_mass_spectrum.png`

---

## Phase 1: Infrastructure & Framework (Commit `33fbc34`)

| Exp ID | Date | Architecture | Dataset | Train Events | Eval Events | Config | Result | Status |
|---|---|---|---|---|---|---|---|---|
| INFRA-001 | 2026-10 | All framework components | N/A | N/A | N/A | Config system, provenance, contracts | 15 unit tests pass | **Complete** |
| INFRA-002 | 2026-10 | Data loader validation | LHCO2020 | N/A | Chunk streaming | PyTables streaming, truth filtering | 100k signal extracted in 0.16s | **Complete** |
| INFRA-003 | 2026-10 | Model factory strict load | All 4 arch | N/A | N/A | state_dict strict=True | All 4 architectures load exactly | **Complete** |
| INFRA-004 | 2026-10 | Smoke tests (MPS) | LHCO2020 | 20 | N/A | All 4 arch, 1 epoch | Losses: MLP=2.93, GAT=3.42, GAT+MET=396.9, GAT+Adv=396.7 | **Complete** |

---

## Phase 2: Architecture Ablation (Family: `architecture`)

**All 4 experiments trained and evaluated on FULL dataset (995k bg, 100k sig)**

### A. Original Baseline Comparisons

| Exp ID | Date | Architecture | Dataset | Train Events | Eval Events | Config | Result | Status |
|---|---|---|---|---|---|---|---|---|
| ARCH-001 | 2026-10-29 | BaselineMLP | LHCO2020 | 5,000 | 995k bg / 100k sig | Flat 64D, no MET | ROC AUC=0.4848, Rej@50%=1.91, Sep=0.9755x, Sig=-0.04σ | **Complete** |
| ARCH-002 | 2026-10-29 | NodeGNN (standalone) | LHCO2020 | 5,000 | 995k bg / 100k sig | GAT, ΔR<0.4, no MET | ROC AUC=0.7815, Rej@50%=6.27, Sep=1.2414x, Sig=0.96σ | **Complete** |
| ARCH-003 | 2026-10-29 | NodeGNN (production) | LHCO2020 | 5,000 | 995k bg / 100k sig | GAT, ΔR<0.4, MET node, λ_MET=0.1 | ROC AUC=0.6584, Rej@50%=3.25, Sep=1.1561x, Sig=0.36σ | **Complete** |
| ARCH-004 | 2026-10-29 | AdversarialNodeGNN | LHCO2020 | 5,000 | 995k bg / 100k sig | GAT, ΔR<0.4, MET node, λ_MET=0.1, λ_adv=0.05 | ROC AUC=0.6608, Rej@50%=3.25, Sep=1.1596x, Sig=0.35σ | **Complete** |

**Source files:** `experiments/architecture/*/metrics.json`, `comparisons/architecture/comparison_table.md`

---

## Phase 3: Mass Sculpting Evaluation (20,000 Background Events)

**Family:** `architecture` → `evaluate_mass_sculpting.py --limit 20000`

| Exp ID | Date | Architecture | Dataset | Train Events | Eval Events | Config | Result | Status |
|---|---|---|---|---|---|---|---|---|
| MASS-001 | 2026-10-30 | BaselineMLP | LHCO2020 bg | 5,000 | 20,000 bg (idx 5k-25k) | 1/JSD at 10%/5%/1% | Raw: 414.0±83.8 / 252.6±48.0 / 51.7±7.5 | **Complete** |
| MASS-002 | 2026-10-30 | GAT Standalone | LHCO2020 bg | 5,000 | 20,000 bg (idx 5k-25k) | 1/JSD at 10%/5%/1% | Raw: 303.4±75.0 / 239.8±44.3 / 29.0±6.9 | **Complete** |
| MASS-003 | 2026-10-30 | GAT+MET Production | LHCO2020 bg | 5,000 | 20,000 bg (idx 5k-25k) | 1/JSD at 10%/5%/1% | Raw: 427.2±83.1 / 195.6±43.0 / 25.3±6.8 | **Complete** |
| MASS-004 | 2026-10-30 | GAT+MET+Adv | LHCO2020 bg | 5,000 | 20,000 bg (idx 5k-25k) | 1/JSD at 10%/5%/1% | Raw: 350.9±87.9 / 208.9±43.6 / 26.9±7.1 | **Complete** |

**Source files:** `experiments/architecture/*/mass_sculpting.json`, `comparisons/architecture/mass_sculpting_table.md`, `comparisons/architecture/deep_physics_investigation.md`

---

## Phase 4: Raw & RSD Invariant Mass Analysis

**Embedded in MASS-001 through MASS-004 above** — both raw dijet mass and Recursive Soft Drop (m_RSD) mass computed for each event via `physics_grooming.py:compute_groomed_dijet_mass()`

| Exp ID | Architecture | 1/JSD Raw (10%) | 1/JSD RSD (10%) | 1/JSD Raw (5%) | 1/JSD RSD (5%) | 1/JSD Raw (1%) | 1/JSD RSD (1%) |
|---|---|---|---|---|---|---|---|
| MASS-001 | BaselineMLP | 414.0 | 441.8 | 252.6 | 190.5 | 51.7 | 30.1 |
| MASS-002 | GAT Standalone | 303.4 | 218.1 | 239.8 | 186.7 | 29.0 | 28.1 |
| MASS-003 | GAT+MET Production | 427.2 | 423.7 | 195.6 | 177.2 | 25.3 | 26.8 |
| MASS-004 | GAT+MET+Adv | 350.9 | 257.8 | 208.9 | 160.5 | 26.9 | 26.5 |

---

## Phase 5: Score vs Mass Correlation Analysis

**Family:** `architecture` → `evaluate_mass_correlation.py --limit 10000 --signal-limit 5000`

| Exp ID | Date | Architecture | Dataset | Train Events | Eval Events | Config | Result | Status |
|---|---|---|---|---|---|---|---|---|
| CORR-001 | 2026-10-30 | BaselineMLP | LHCO2020 | 5,000 | 10k bg / 5k sig | Pearson r, Spearman ρ | r=-0.0388, ρ=-0.0613 | **Complete** |
| CORR-002 | 2026-10-30 | GAT Standalone | LHCO2020 | 5,000 | 10k bg / 5k sig | Pearson r, Spearman ρ | r=+0.0858, ρ=+0.1227 | **Complete** |
| CORR-003 | 2026-10-30 | GAT+MET Production | LHCO2020 | 5,000 | 10k bg / 5k sig | Pearson r, Spearman ρ | r=+0.0665, ρ=+0.1133 | **Complete** |
| CORR-004 | 2026-10-30 | GAT+MET+Adv | LHCO2020 | 5,000 | 10k bg / 5k sig | Pearson r, Spearman ρ | r=+0.0758, ρ=+0.1180 | **Complete** |

**Source:** `comparisons/architecture/mass_correlation_study.md`

---

## Phase 6: Mass-Matched ROC-AUC Analysis

**Same evaluation as CORR-001 through CORR-004**

| Exp ID | Architecture | Inclusive AUC | Mass-Matched AUC [3.3, 3.7 TeV] | ΔAUC | Events in Window |
|---|---|---|---|---|---|
| CORR-001 | BaselineMLP | 0.4913 | **0.5138** | +0.0225 | 1,224 bg / 3,829 sig |
| CORR-002 | GAT Standalone | **0.7076** | **0.6804** | -0.0272 | 1,224 bg / 3,829 sig |
| CORR-003 | GAT+MET Production | 0.6644 | **0.6338** | -0.0307 | 1,224 bg / 3,829 sig |
| CORR-004 | GAT+MET+Adv | 0.6577 | **0.6252** | -0.0326 | 1,224 bg / 3,829 sig |

---

## Phase 7: Narrow & Micro-Fine Mass Window Analysis

**Family:** `architecture` → `investigate_deep_physics.py --limit 10000 --signal-limit 5000` (Task 1)

| Exp ID | Date | Architecture | Window | Half-Width | N_bg | N_sig | Matched AUC | Status |
|---|---|---|---|---|---|---|---|---|
| MICRO-001 | 2026-10-30 | GAT Standalone | Inclusive | Full | 10,000 | 5,000 | 0.7076 | **Complete** |
| MICRO-002 | 2026-10-30 | GAT Standalone | [3300, 3700] | ±200 GeV | 1,224 | 3,829 | 0.6804 | **Complete** |
| MICRO-003 | 2026-10-30 | GAT Standalone | [3350, 3650] | ±150 GeV | 907 | 3,254 | 0.6812 | **Complete** |
| MICRO-004 | 2026-10-30 | GAT Standalone | [3400, 3600] | ±100 GeV | 599 | 2,475 | 0.6652 | **Complete** |
| MICRO-005 | 2026-10-30 | GAT Standalone | **[3450, 3550]** | **±50 GeV** | **289** | **1,332** | **0.6719** | **Complete** |

### 100-GeV Mass Slices (GAT Standalone)

| Slice | N_bg | N_sig | Slice AUC |
|---|---|---|---|
| [3000, 3100] | 551 | 92 | 0.6459 |
| [3100, 3200] | 485 | 147 | 0.6825 |
| [3200, 3300] | 443 | 275 | 0.7048 |
| [3300, 3400] | 360 | 677 | 0.6946 |
| [3400, 3500] | 332 | 1,240 | 0.6534 |
| [3500, 3600] | 267 | 1,235 | 0.6803 |
| [3600, 3700] | 265 | 677 | 0.6891 |
| [3700, 3800] | 193 | 220 | 0.6644 |
| [3800, 3900] | 161 | 108 | 0.6897 |
| [3900, 4000] | 155 | 50 | 0.6453 |

**Source:** `comparisons/architecture/deep_physics_investigation.md` (Table: "Window Narrowing", "Sliced Mass Spectrum")

---

## Phase 8: Feature-Wise Anomaly Score Ablation

**Family:** `architecture` → `investigate_deep_physics.py` (Task 2)

| Exp ID | Date | Architecture | Score Type | Inclusive AUC | Matched AUC [3.3, 3.7 TeV] | BG Loss Share | Status |
|---|---|---|---|---|---|---|---|
| FEAT-001 | 2026-10-30 | GAT Standalone | Full 3D | 0.7076 | 0.6804 | 100% | **Complete** |
| FEAT-002 | 2026-10-30 | GAT Standalone | **pT-only** | **0.7954** | **0.7676** | **80.5%** | **Complete** |
| FEAT-003 | 2026-10-30 | GAT Standalone | Angular (η,φ) | 0.4237 | **0.4172** | 19.5% | **Complete** |
| FEAT-004 | 2026-10-30 | GAT Standalone | Weighted (0.8/0.2) | — | — | — | **Proposed** |

**Note:** Angular-only AUC < 0.50 indicates inversion — signal jets more collimated (easier to reconstruct) than QCD background.

---

## Phase 9: Graph Structure Experiments (Inference-Time Interventions)

**Family:** `architecture` → `investigate_deep_physics.py` (Task 3) — **Same trained GAT Standalone model**

| Exp ID | Date | Graph Mode | Description | Inclusive AUC | Matched AUC [3.3, 3.7 TeV] | Δ vs Relational | Status |
|---|---|---|---|---|---|---|---|
| GRAPH-001 | 2026-10-30 | Relational | ΔR < 0.4 physical edges | 0.7076 | **0.6804** | Baseline | **Complete** |
| GRAPH-002 | 2026-10-30 | Scrambled | Permuted edge destinations | 0.5211 | **0.4709** | -0.2095 | **Complete** |
| GRAPH-003 | 2026-10-30 | Disconnected | Self-loops only (no MP) | **0.7959** | **0.7932** | +0.1128 | **Complete** |

**Critical:** These are **inference-time interventions** on the SAME trained model — NOT independently trained models. The disconnected mode reveals that message passing with soft radiation degrades discrimination.

---

## Phase 10: Independently Trained Node-Wise Autoencoder

**Family:** `controlled_structure` → `evaluate_controlled_ablation.py`

| Exp ID | Date | Architecture | Seed | Train Events | Eval Events | Config | Result | Status |
|---|---|---|---|---|---|---|---|---|
| NODE-001 | 2026-10-30 | NodeWiseAE | 999 | 5,000 | 10k bg / 5k sig | ΔR=0.0, no MET | Training complete; evaluation pending | **Trained, not evaluated** |
| NODE-002 | 2026-10 | NodeWiseAE | 42 | 5,000 | — | ΔR=0.0, no MET | Config only | **Not trained** |
| NODE-003 | 2026-10 | NodeWiseAE | 123 | 5,000 | — | ΔR=0.0, no MET | Config only | **Not trained** |

**Source:** `experiments/controlled_structure/nodewise_ae_s999/checkpoint.pt` + `training_log.json`

---

## Phase 11: DeepSets / PointNet Experiments

**Family:** `controlled_structure` — **NO TRAINING EXECUTED**

| Exp ID | Date | Architecture | Seed | Train Events | Config | Status |
|---|---|---|---|---|---|---|
| DS-001 | 2026-10 | DeepSetsPointNet | 42 | 5,000 | ΔR=0.0, global pool | Config only |
| DS-002 | 2026-10 | DeepSetsPointNet | 123 | 5,000 | ΔR=0.0, global pool | Config only |
| DS-003 | 2026-10 | DeepSetsPointNet | 999 | 5,000 | ΔR=0.0, global pool | Config only |

---

## Phase 12: Multi-Seed Experiments (Controlled Structure)

**Family:** `controlled_structure` — **11 of 12 configs NOT TRAINED**

| Config | Architecture | Seed | Status |
|---|---|---|---|
| baseline_mlp_s42 | BaselineMLP | 42 | Config only |
| baseline_mlp_s123 | BaselineMLP | 123 | Config only |
| baseline_mlp_s999 | BaselineMLP | 999 | Config only |
| gat_relational_s42 | NodeGNN | 42 | Config only |
| gat_relational_s123 | NodeGNN | 123 | Config only |
| gat_relational_s999 | NodeGNN | 999 | Config only |
| nodewise_ae_s42 | NodeWiseAE | 42 | Config only |
| nodewise_ae_s123 | NodeWiseAE | 123 | Config only |
| **nodewise_ae_s999** | **NodeWiseAE** | **999** | **TRAINED** |
| deepsets_pointnet_s42 | DeepSetsPointNet | 42 | Config only |
| deepsets_pointnet_s123 | DeepSetsPointNet | 123 | Config only |
| deepsets_pointnet_s999 | DeepSetsPointNet | 999 | Config only |

---

## Phase 13: Graph Construction Sweep (ΔR Threshold)

**Family:** `graph_construction` — **NO TRAINING EXECUTED**

| Config | ΔR Threshold | Architecture | Status |
|---|---|---|---|
| dr_02 | 0.2 | NodeGNN | Config only |
| dr_04_baseline | 0.4 | NodeGNN | Config only (reference) |
| dr_06 | 0.6 | NodeGNN | Config only |

---

## Phase 14: MET Weight Sweep

**Family:** `met_weight` — **NO TRAINING EXECUTED**

| Config | λ_MET | Architecture | Status |
|---|---|---|---|
| lambda_0 | 0.0 | NodeGNN | Config only |
| lambda_005 | 0.05 | NodeGNN | Config only |
| lambda_010_baseline | 0.1 | NodeGNN | Config only (reference) |
| lambda_020 | 0.2 | NodeGNN | Config only |

---

## Phase 15: Mass Decorrelation Sweep (Adversarial)

**Family:** `mass_decorrelation` — **NO TRAINING EXECUTED**

| Config | λ_adv | Architecture | Status |
|---|---|---|---|
| without_decorrelation | 0.0 | AdversarialNodeGNN | Config only |
| with_decorrelation | 0.05 | AdversarialNodeGNN | Config only |

---

## Phase 16: Score-Level Decorrelation (PROPOSED — Zero Training)

**Family:** `score_decorrelation` — **ALL PROPOSED, NOT TRAINED**

| Config | Decorrelation | Score Type | λ | Architecture | Status |
|---|---|---|---|---|---|
| gat_score_disco_s42 | DisCo | pt_only | 1.0 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_score_disco_s123 | DisCo | pt_only | 1.0 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_score_disco_s999 | DisCo | pt_only | 1.0 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_score_grl_s42 | GRL | pt_only | 0.05 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_score_grl_s123 | GRL | pt_only | 0.05 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_score_grl_s999 | GRL | pt_only | 0.05 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_score_disco_heavy_s42 | DisCo | pt_only | 2.0 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_score_disco_weighted_s42 | DisCo | weighted | 1.0 | ScoreDecorrelatedNodeGNN | Proposed |
| gat_baseline_ref_s42 | None | full | 0.0 | NodeGNN | Proposed (reference) |

---

## Summary Statistics

| Category | Total Configs | Trained | Evaluated | Pending |
|---|---|---|---|---|
| Architecture Ablation | 4 | 4 | 4 | 0 |
| Mass Sculpting | 4 | 4 | 4 | 0 |
| Mass Correlation | 4 | 4 | 4 | 0 |
| Micro-Fine Mass Windows | 5 | 1 model | 1 model | 0 |
| Feature Ablation | 4 | 1 model | 1 model | 3 scores |
| Graph Structure (inference) | 3 | 1 model | 1 model | 0 |
| Controlled Structure (independent) | 12 | 1 | 0 | 11 |
| Graph Construction Sweep | 3 | 0 | 0 | 3 |
| MET Weight Sweep | 4 | 0 | 0 | 4 |
| Mass Decorrelation Sweep | 2 | 0 | 0 | 2 |
| Score Decorrelation | 9 | 0 | 0 | 9 |
| **TOTAL** | **53** | **14** | **13** | **39** |

---

## Artifact Sources for Every Metric

| Metric | Source File |
|---|---|
| ROC AUC, Rejection, Separation, Significance | `experiments/architecture/<name>/metrics.json` |
| Mass Sculpting (1/JSD) | `experiments/architecture/<name>/mass_sculpting.json` |
| Mass Correlation (Pearson, Spearman) | `comparisons/architecture/mass_correlation_study.md` |
| Mass-Matched ROC AUC | `comparisons/architecture/mass_correlation_study.md` |
| Latent Probe R² | `comparisons/architecture/mass_correlation_study.md` |
| Micro-Fine Windows | `comparisons/architecture/deep_physics_investigation.md` |
| Feature Attribution | `comparisons/architecture/deep_physics_investigation.md` |
| Graph Structure Ablation | `comparisons/architecture/deep_physics_investigation.md` |
| MET Investigation | `comparisons/architecture/deep_physics_investigation.md` |
| Training Logs | `experiments/<family>/<name>/training_log.json` |
| Configurations | `configs/<family>/<name>.json` |

---

**End of Experiment Registry**