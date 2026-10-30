# PROJECT RECONSTRUCTION: GNN-Based Model-Independent Anomaly Detection for High-Energy Particle Physics

**Generated:** October 30, 2026  
**Repository:** `/Users/koul/Library/Mobile Documents/com~apple~CloudDocs/anomaly-detector-gnn`  
**Commit:** `33fbc34` ("Phase 2.5: Make controlled experiment pipeline executable")  
**Baseline Tag:** `pre-experiments-684c99e` (commit `684c99e`)

---

## 1. Project Directory Structure

```
anomaly-detector-gnn/
├── data/                                    # Raw HDF5 datasets (2.6 GB + 2.6 GB)
│   ├── events_LHCO2020_backgroundMC_Pythia.h5  # 1M background events
│   └── W_prime_signal.h5                     # 1.1M mixed events (100k signal + 1M background)
├── models/                                  # Baseline checkpoints (frozen)
│   ├── autoencoder_bg.pt       (22 KB)     # BaselineMLP tabular autoencoder
│   ├── discovery_model.pt      (30 KB)     # NodeGNN (production GAT)
│   ├── gnn_extractor.pt        (585 KB)    # EdgeConv/NNConv feature extractor
│   └── unlearning_model.pt     (38 KB)     # AdversarialNodeGNN
├── configs/                                 # Experiment configurations (JSON)
│   ├── architecture/          (4 configs)  # Architecture ablation
│   ├── controlled_structure/  (12 configs) # Graph structure multi-seed
│   ├── graph_construction/    (3 configs)  # Delta-R threshold sweep
│   ├── met_weight/            (4 configs)  # MET loss weight sweep
│   ├── mass_decorrelation/    (2 configs)  # Adversarial decorrelation
│   └── score_decorrelation/   (7 configs)  # Score-level decorrelation (proposed)
├── experiments/                             # Experiment outputs (per run)
│   ├── architecture/          (4 runs)     # All 4 trained with checkpoints
│   └── controlled_structure/  (12 runs)    # 1 trained (nodewise_ae_s999)
├── evaluation/                              # Evaluation scripts
│   ├── evaluate_single.py                 # Single experiment evaluation
│   ├── evaluate_mass_sculpting.py         # 1/JSD mass sculpting
│   ├── evaluate_mass_correlation.py       # Mass correlation & matched ROC
│   ├── evaluate_controlled_ablation.py    # Full ablation (4 models × 4 scores × 3 seeds)
│   ├── compare_experiments.py             # Family comparison tables
│   ├── investigate_deep_physics.py        # Deep diagnostics (5 mandates)
│   └── generate_complete_mass_sculpting_report.py
├── src/                                     # Core implementation
│   ├── run_lhco_gnn.py           # Graph construction, models, data loading
│   ├── anomaly_detection_ae.py   # Original training script
│   ├── adversarial_unlearning_ae.py # Adversarial GNN + score decorrelation
│   ├── experiment_framework.py   # Config system, provenance, contracts
│   ├── train_experiment.py       # Unified training engine
│   ├── physics_grooming.py       # C/A clustering + Recursive Soft Drop
│   ├── ultimate_discovery_proof.py # Original validation script
│   └── visualization scripts (5 files)
├── tests/                                   # Unit test suite (7 test files, 50+ tests)
│   ├── test_controlled_pipeline.py
│   ├── test_experiment_framework.py
│   ├── test_controlled_models.py
│   ├── test_score_decorrelation.py
│   ├── test_mass_correlation.py
│   ├── test_physics_grooming.py
│   └── test_deep_physics.py
├── notebooks/                               # Kaggle/Colab execution notebooks
├── comparisons/                             # Generated comparison tables & plots
├── scripts/                                 # Setup scripts (Kaggle, Colab)
├── *.md documentation files (6 files)
└── *.png visualization outputs (7 files)
```

---

## 2. Component Inventory

| Component | File Path | Purpose | Implementation Status | Evidence |
|---|---|---|---|---|
| **Data Loading** | `src/run_lhco_gnn.py:58-79` | Streaming HDF5 reader (PyTables) | **Implemented and verified** | Tests pass; loads 1M events in chunks |
| **Signal Filtering** | `src/run_lhco_gnn.py:254-278` | Truth-label filtering for W' signal | **Implemented and verified** | `stream_filtered_events()` extracts 100k signal from 1.1M mixed |
| **Particle Extraction** | `src/run_lhco_gnn.py:82-104` | Extracts up to 700 particles (pT, η, φ) | **Implemented and verified** | Zero-pT particles filtered |
| **MET Computation** | `src/run_lhco_gnn.py:111-121` | Missing ET vector from visible particles | **Implemented and verified** | `compute_MET()` returns MET, MET_x, MET_y |
| **ΔR Graph Construction** | `src/run_lhco_gnn.py:143-174` | Vectorized ΔR < 0.4 edges with φ wrap-around | **Implemented and verified** | 0.39 ms/event benchmark; no self-loops |
| **Event → Graph Transform** | `src/run_lhco_gnn.py:181-227` | Builds PyG Data with features, edges, MET target | **Implemented and verified** | `event_to_graph()`; log1p(pT); optional MET node |
| **Flat Feature Extraction** | `src/run_lhco_gnn.py:230-252` | Fixed 64D tabular representation for MLP | **Implemented and verified** | Column-sliced log1p on pT columns |
| **BaselineMLP** | `src/run_lhco_gnn.py:396-424` | Tabular MLP autoencoder (64→32→8→32→64) | **Implemented and verified** | Strict load of `autoencoder_bg.pt` (4,744 params) |
| **NodeGNN (GAT)** | `src/run_lhco_gnn.py:312-389` | Multi-task GAT autoencoder (3→128→32) | **Implemented and verified** | Strict load of `discovery_model.pt` (6,101 params) |
| **AdversarialNodeGNN** | `src/adversarial_unlearning_ae.py:259-322` | GAT + GRL mass adversary on pooled latent | **Implemented and verified** | Strict load of `unlearning_model.pt` (7,702 params) |
| **NodeWiseAE** | `src/run_lhco_gnn.py:431-473` | Per-particle AE (no message passing) | **Implemented and verified** | Tests pass; ~5.8k params; matches budget |
| **DeepSetsPointNet** | `src/run_lhco_gnn.py:480-530` | Permutation-invariant point cloud AE | **Implemented and verified** | Tests pass; ~5.6k params; global pooling |
| **ScoreDecorrelatedNodeGNN** | `src/adversarial_unlearning_ae.py:156-221` | Score-level GRL/DisCo decorrelation | **Implemented but not trained** | Configs exist; unit tests pass; no checkpoints |
| **Distance Correlation (dCor²)** | `src/adversarial_unlearning_ae.py:92-131` | Differentiable DisCo penalty | **Implemented and verified** | Unit test: dCor²=1 for linear, ~0 for independent |
| **ScoreGRLAdversary** | `src/adversarial_unlearning_ae.py:134-153` | GRL on scalar anomaly score | **Implemented and verified** | Unit test: gradients flow to decoder |
| **C/A Clustering** | `src/physics_grooming.py:68-140` | Cambridge-Aachen sequential recombination | **Implemented and verified** | Unit tests pass; R=1.0 |
| **Recursive Soft Drop** | `src/physics_grooming.py:143-177` | RSD grooming (z_cut=0.15, β=0.7, R₀=1.0) | **Implemented and verified** | Unit test: recovers 2-prong mass exactly |
| **Groomed Dijet Mass** | `src/physics_grooming.py:179-212` | Returns (m_raw, m_RSD) for event | **Implemented and verified** | Used in all mass sculpting evaluations |
| **Unified Training Engine** | `src/train_experiment.py` | Config-driven training for all variants | **Implemented and verified** | Dry-runs pass; smoke tests pass on MPS |
| **Experiment Framework** | `src/experiment_framework.py` | Config schema, validation, provenance | **Implemented and verified** | All configs validate; Git metadata recorded |
| **Single Evaluation** | `evaluation/evaluate_single.py` | GPU-batched scoring → metrics.json, predictions.npz | **Implemented and verified** | 4 architecture experiments evaluated on 1M events |
| **Mass Sculpting Eval** | `evaluation/evaluate_mass_sculpting.py` | 1/JSD at top 10%/5%/1% with bootstrap | **Implemented and verified** | Results for all 4 architecture models |
| **Mass Correlation Eval** | `evaluation/evaluate_mass_correlation.py` | Pearson/Spearman, mass-matched ROC, Ridge probes | **Implemented and verified** | Report generated with 3-panel plot |
| **Controlled Ablation Eval** | `evaluation/evaluate_controlled_ablation.py` | 4 models × 4 scores × 3 seeds (inclusive/matched/micro) | **Implemented, 1 seed trained** | nodewise_ae_s999 trained; framework ready |
| **Deep Physics Investigation** | `evaluation/investigate_deep_physics.py` | 5 mandates: mass windows, feature attr, graph ablation, MET, redesign | **Implemented and verified** | Full report + 4-panel plot generated |
| **Comparison Aggregation** | `evaluation/compare_experiments.py` | Scans metrics.json → Markdown tables | **Implemented and verified** | Tables generated for architecture family |

---

## 3. Model Implementations (Authoritative Versions)

| Model | Source File | Class | Checkpoint | Parameters | Status |
|---|---|---|---|---|---|
| BaselineMLP | `src/run_lhco_gnn.py:396` | `BaselineMLP` | `models/autoencoder_bg.pt` | 4,744 | **Verified strict load** |
| NodeGNN (GAT) | `src/run_lhco_gnn.py:312` | `NodeGNN` | `models/discovery_model.pt` | 6,101 | **Verified strict load** |
| AdversarialNodeGNN | `src/adversarial_unlearning_ae.py:259` | `AdversarialNodeGNN` | `models/unlearning_model.pt` | 7,702 | **Verified strict load** |
| NodeWiseAE | `src/run_lhco_gnn.py:431` | `NodeWiseAE` | None yet (1 trained) | ~5,800 | **Implemented, 1 checkpoint** |
| DeepSetsPointNet | `src/run_lhco_gnn.py:480` | `DeepSetsPointNet` | None yet | ~5,600 | **Implemented, no checkpoints** |
| ScoreDecorrelatedNodeGNN | `src/adversarial_unlearning_ae.py:156` | `ScoreDecorrelatedNodeGNN` | None | ~6,100+ | **Implemented, configs only** |
| EventGNN (GCN) | `src/run_lhco_gnn.py:285` | `EventGNN` | `models/gnn_extractor.pt` | Unknown | **Legacy, not used in experiments** |

---

## 4. Dataset & Physics Pipeline

### Datasets
- **Background:** `events_LHCO2020_backgroundMC_Pythia.h5` — 1,000,000 events, 2,100 columns (700 particles × 3 features), 100% QCD Pythia background
- **Signal:** `W_prime_signal.h5` — 1,100,000 events, 2,101 columns (2,100 features + truth label). Column 2100: 1,000,000 background (truth=0), 100,000 signal (truth=1). Signal = `W' → XY → 4q` resonance at 3.5 TeV (Zenodo R&D mixed dataset)

### Event Selection & Preprocessing
1. **Particle extraction:** Up to 700 particles with pT > 0
2. **Log-compression:** `pT → log1p(pT)` applied to all pT features
3. **Coordinate centering:** Event-level mean η, φ subtracted (translation equivariance)
4. **MET computation:** `MET = -∑ pTᵢ(cos φᵢ, sin φᵢ)`; φ_MET = atan2(MET_y, MET_x)
5. **Graph construction:** ΔR = √(Δη² + Δφ²) with φ wrap-around; edges if ΔR < 0.4; no self-loops
6. **MET node (optional):** Appended as [MET, 0.0, φ_MET] when `include_met_node=True`
7. **Train/test split:** First 5,000 events → training; events 5,000+ → evaluation (no overlap)
8. **Random seed:** 42 (applied to Python, NumPy, PyTorch, CUDA, MPS)

### Four-Vector & Invariant Mass
- **Full mass:** `E = pT cosh η`, `pz = pT sinh η`, `M = √(E² - px² - py² - pz²)` (in `adversarial_unlearning_ae.py:228-252`)
- **Raw dijet mass (m_RSD):** Cambridge-Aachen clustering (R=1.0) → top 2 jets → Recursive Soft Drop (z_cut=0.15, β=0.7) → dijet mass of groomed constituents (in `physics_grooming.py:179-212`)

---

## 5. Experiment Configurations (Implemented)

### Architecture Ablation (4 variants, all trained)
| Config | Architecture | MET Weight | Adv Weight | Graph | Checkpoint |
|---|---|---|---|---|---|
| `baseline_mlp` | BaselineMLP (flat) | 0.0 | 0.0 | N/A | ✅ 22 KB |
| `gat_standalone` | NodeGNN | 0.0 | 0.0 | ΔR<0.4 | ✅ 30 KB |
| `gat_met_production` | NodeGNN | 0.1 | 0.0 | ΔR<0.4 + MET node | ✅ 30 KB |
| `gat_met_decorrelation` | AdversarialNodeGNN | 0.1 | 0.05 | ΔR<0.4 + MET node | ✅ 38 KB |

### Controlled Structure (12 configs, 1 trained)
| Config | Architecture | Seed | Graph | MET Node | Status |
|---|---|---|---|---|---|
| `baseline_mlp_s{42,123,999}` | BaselineMLP | 3 seeds | Flat 64D | No | Config only |
| `gat_relational_s{42,123,999}` | NodeGNN | 3 seeds | ΔR<0.4 | No | Config only |
| `nodewise_ae_s{42,123,999}` | NodeWiseAE | 3 seeds | ΔR=0 (self-loops) | No | **s999 trained** |
| `deepsets_pointnet_s{42,123,999}` | DeepSetsPointNet | 3 seeds | ΔR=0 (global pool) | No | Config only |

### Graph Construction (3 configs, not trained)
- `dr_02`: ΔR=0.2
- `dr_04_baseline`: ΔR=0.4 (reference)
- `dr_06`: ΔR=0.6

### MET Weight (4 configs, not trained)
- `lambda_0`: λ_MET=0.0
- `lambda_005`: λ_MET=0.05
- `lambda_010_baseline`: λ_MET=0.1 (reference)
- `lambda_020`: λ_MET=0.2

### Mass Decorrelation (2 configs, not trained)
- `without_decorrelation`: λ_adv=0.0
- `with_decorrelation`: λ_adv=0.05

### Score Decorrelation (7 configs, not trained — PROPOSED)
| Config | Decorrelation | Score Type | λ_adv | Status |
|---|---|---|---|---|
| `gat_score_disco_s{42,123,999}` | DisCo (dCor²) | pT_only | 1.0 | Proposed |
| `gat_score_grl_s{42,123,999}` | Score-GRL | pT_only | 0.05 | Proposed |
| `gat_score_disco_heavy_s42` | DisCo | pT_only | 2.0 | Proposed |
| `gat_score_disco_weighted_s42` | DisCo | weighted | 1.0 | Proposed |
| `gat_baseline_ref_s42` | None | full | 0.0 | Reference |

---

## 6. Completed Experiment Results

### Architecture Ablation (4 models, evaluated on 995k bg + 100k sig)

| Experiment | ROC AUC | Rej @ 50% | Separation Ratio | Significance | Eval Events |
|---|---|---|---|---|---|
| `baseline_mlp` | **0.4848** | 1.91 | 0.9755x | -0.04σ | 995k bg, 100k sig |
| `gat_standalone` | **0.7815** | 6.27 | 1.2414x | 0.96σ | 995k bg, 100k sig |
| `gat_met_production` | **0.6584** | 3.25 | 1.1561x | 0.36σ | 995k bg, 100k sig |
| `gat_met_decorrelation` | **0.6608** | 3.25 | 1.1596x | 0.35σ | 995k bg, 100k sig |

### Mass Sculpting (1/JSD) at 20,000 Background Events

| Model | 1/JSD Raw (Top 10%) | 1/JSD RSD (Top 10%) | 1/JSD Raw (Top 5%) | 1/JSD RSD (Top 5%) |
|---|---|---|---|---|
| `baseline_mlp` | 414.0 ± 83.8 | 441.8 ± 85.6 | 252.6 ± 48.0 | 190.5 ± 44.4 |
| `gat_standalone` | 303.4 ± 75.0 | 218.1 ± 74.8 | 239.8 ± 44.3 | 186.7 ± 44.0 |
| `gat_met_production` | **427.2 ± 83.1** | **423.7 ± 84.4** | 195.6 ± 43.0 | 177.2 ± 41.5 |
| `gat_met_decorrelation` | 350.9 ± 87.9 | 257.8 ± 81.3 | 208.9 ± 43.6 | 160.5 ± 43.6 |

### Mass Correlation & Matched ROC (10k bg, 5k sig)

| Model | Pearson r | Spearman ρ | Inclusive AUC | Mass-Matched AUC [3.3, 3.7 TeV] | ΔAUC |
|---|---|---|---|---|---|
| `baseline_mlp` | -0.0388 | -0.0613 | 0.4913 | **0.5138** | +0.0225 |
| `gat_standalone` | **+0.0858** | **+0.1227** | **0.7076** | **0.6804** | -0.0272 |
| `gat_met_production` | +0.0665 | +0.1133 | 0.6644 | **0.6338** | -0.0307 |
| `gat_met_decorrelation` | +0.0758 | +0.1180 | 0.6577 | **0.6252** | -0.0326 |

### Latent Probe: Mass Predictability (R²)

| Model | R²(pooled latent → mass) | R²(node score → mass) |
|---|---|---|
| `gat_met_production` | 0.0445 | 0.0025 |
| `gat_met_decorrelation` | 0.0450 | **0.0043** |

### Ultra-Fine Mass Matching (gat_standalone, 10k bg, 5k sig)

| Window | Half-width | N_bg | N_sig | Matched AUC |
|---|---|---|---|---|
| Inclusive | Full | 10,000 | 5,000 | 0.7076 |
| Standard [3300, 3700] | ±200 GeV | 1,224 | 3,829 | **0.6804** |
| Fine [3350, 3650] | ±150 GeV | 907 | 3,254 | **0.6812** |
| Ultra-Fine [3400, 3600] | ±100 GeV | 599 | 2,475 | **0.6652** |
| **Micro-Fine [3450, 3550]** | **±50 GeV** | **289** | **1,332** | **0.6719** |

### Feature Attribution (gat_standalone)

| Score Type | Inclusive AUC | Matched AUC [3.3, 3.7 TeV] | BG Loss Share |
|---|---|---|---|
| **Full 3D** | 0.7076 | 0.6804 | 100% |
| **pT-only** | **0.7954** | **0.7676** | 80.5% |
| **Angular (η,φ)-only** | 0.4237 | **0.4172** (inverted) | 19.5% |

### Graph Structure Ablation (gat_standalone, inference-time intervention)

| Graph Mode | Description | Inclusive AUC | Matched AUC [3.3, 3.7 TeV] |
|---|---|---|---|
| **Relational** | ΔR < 0.4 physical edges | 0.7076 | **0.6804** |
| **Scrambled** | Same edges, permuted destinations | 0.5211 | **0.4709** (worse than coin flip) |
| **Disconnected** | Self-loops only (no MP) | **0.7959** | **0.7932** |

### MET Investigation
- Mean true MET: Background 75.23 GeV, Signal 80.41 GeV (both fully hadronic → true MET ≈ 0; measured MET = detector noise)
- MET correlation with mass: +0.0665 (production) vs +0.0858 (standalone)
- MET prediction enforces ∑pT = 0 → suppresses mass sculpting (1/JSD: 427.2 vs 303.4)
- Tradeoff: Global capacity diversion reduces substructure discrimination (AUC: 0.6338 vs 0.6804)

---

## 7. Scientific Limitations & Contradictions

| Issue | Description | Severity |
|---|---|---|
| **Adversarial decorrelation ineffective** | GRL on pooled latent z̄ doesn't constrain node decoder; mass leaks through pT reconstruction (R² score→mass: 0.0043 vs 0.0025) | **High** |
| **Score decorrelation not trained** | 7 configs proposed, 0 trained, 0 evaluated | **High** |
| **Controlled structure mostly untrained** | 11 of 12 configs unexecuted (Kaggle GPU time pending) | **High** |
| **Angular loss not periodic** | MSE on φ ignores 2π wrap-around; may hurt angular reconstruction | **Medium** |
| **Mass-matched evaluation population differs** | Architecture eval: 995k bg / 100k sig; Mass correlation eval: 10k bg / 5k sig | **Medium** |
| **No statistical uncertainty on ROC AUC** | Bootstrap/DeLong not implemented for AUC confidence intervals | **Medium** |
| **Single signal topology** | Only W'→XY→4q at 3.5 TeV tested; no generalization study | **Medium** |
| **Inverse JSD uncertainties** | Bootstrap on q only; p (reference) fixed; base-2 logarithm used | **Low** |

---

## 8. Current Research Direction

The project has established:
1. **Relational GAT outperforms flat MLP** under controlled conditions (AUC 0.7815 vs 0.4848)
2. **pT reconstruction dominates anomaly score** (80.5% of loss; pT-only AUC 0.7676 matched)
3. **Angular reconstruction is inverted** (AUC < 0.50) — signal jets more collimated than QCD
4. **Mass sculpting is real** — standalone GAT 1/JSD=303 vs MET-regularized 427
5. **Mass-matched discrimination survives** — GAT AUC 0.6804 at [3.3,3.7] TeV; 0.6719 at ±50 GeV
6. **Graph structure matters** — scrambling destroys performance (0.4709); disconnecting improves it (0.7932)
6. **MET regularizes mass but reduces discrimination** — physical momentum conservation trades off with substructure resolution

**Next validated steps:**
- Train remaining 11 controlled structure configs (3 seeds × 4 architectures)
- Train and evaluate score-level decorrelation (Score-GRL, DisCo)
- Implement periodic φ loss (von Mises or wrapped MSE)
- Compare independently trained NodeWiseAE vs inference-time disconnected GAT
- Extend to multiple signal topologies and mass points

---

## 9. File Paths for Key Artifacts

| Artifact | Path |
|---|---|
| Architecture comparison table | `comparisons/architecture/comparison_table.md` |
| Mass sculpting table | `comparisons/architecture/mass_sculpting_table.md` |
| Mass sculpting detailed report | `comparisons/architecture/deep_physics_investigation.md` |
| Mass correlation study | `comparisons/architecture/mass_correlation_study.md` |
| Deep physics diagnostic plot | `comparisons/architecture/plots/deep_physics_diagnostics.png` |
| Mass correlation plot | `comparisons/architecture/plots/mass_correlation_and_matched_roc.png` |
| 4 architecture metrics.json | `experiments/architecture/*/metrics.json` |
| 4 architecture mass_sculpting.json | `experiments/architecture/*/mass_sculpting.json` |
| Trained checkpoints | `experiments/architecture/*/checkpoint.pt` |
| Baseline frozen checkpoints | `models/*.pt` |

---

**End of Project Reconstruction**