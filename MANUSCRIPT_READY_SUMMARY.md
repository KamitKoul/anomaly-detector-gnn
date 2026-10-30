# MANUSCRIPT-READY PROJECT SUMMARY

**Generated:** October 30, 2026  
**Repository:** `anomaly-detector-gnn`  
**For:** Manuscript revision preparation  
**Format:** Each statement labeled as **VERIFIED MEASUREMENT**, **IMPLEMENTATION FACT**, **SCIENTIFIC INTERPRETATION**, **HYPOTHESIS**, or **PROPOSED FUTURE WORK**

---

## 1. Research Problem

**VERIFIED MEASUREMENT:** The LHC Olympics 2020 (LHCO2020) dataset provides 1,000,000 Standard Model QCD background events (`events_LHCO2020_backgroundMC_Pythia.h5`) and a mixed signal dataset (`W_prime_signal.h5` = 1,000,000 background + 100,000 `W' → XY → 4q` signal events at 3.5 TeV resonance) for model-independent anomaly detection benchmarking.

**SCIENTIFIC INTERPRETATION:** Model-independent anomaly detection searches for deviations from Standard Model predictions without assuming a specific BSM signal model. This is critical because the space of possible BSM physics is vast, and dedicated searches for each model are impractical. An autoencoder trained only on background learns the "normal" manifold; events with high reconstruction error are anomalies potentially indicating new physics.

**HYPOTHESIS:** Graph Neural Networks can capture relativistic geometric correlations between particles (jet substructure, angular clustering) that flat tabular representations miss, leading to better anomaly discrimination.

---

## 2. Motivation

**IMPLEMENTATION FACT:** Existing anomaly detection approaches in HEP primarily use:
- Flat tabular autoencoders (e.g., `BaselineMLP` in this repo: 64→32→8→32→64)
- Jet-level conditional autoencoders (CATHODE, Kasieczka et al. 2021)
- Contrastive learning on event-level features (ATLASSML 2022)

**SCIENTIFIC INTERPRETATION:** Particle collisions produce complex relational structures: particles from the same parton shower cluster in angular space (ΔR < 0.4 defines jet cones). Flat representations lose this topology. GNNs with message passing along physical ΔR edges can learn "hadronic grammar" — the pattern of QCD radiation — and flag events violating it.

**HYPOTHESIS:** Multi-task learning (reconstruction + MET prediction) injects physical conservation laws into the latent space, regularizing against mass sculpting artifacts.

---

## 3. Scientific Hypothesis

**ORIGINAL HYPOTHESIS (Pre-Experiment):**
1. Relational GAT autoencoder outperforms flat MLP on LHCO2020 anomaly detection
2. Multi-task MET prediction improves background mass preservation
3. Adversarial decorrelation on pooled latent removes mass bias from anomaly scores

**HYPOTHESES INTRODUCED DURING EXPERIMENTS:**
4. pT-only reconstruction outperforms full 3D reconstruction (discovered in feature ablation)
5. Angular reconstruction is inverted (signal easier to reconstruct than background)
6. Message passing with soft radiation degrades hard-prong discrimination
7. Pooled-latent adversary fails because node decoder bypasses it → score-level decorrelation needed
8. MET prediction regularizes mass via global momentum conservation, not because MET discriminates signal

---

## 4. Dataset and Event Selection

**VERIFIED MEASUREMENT:**
- **Background:** 1,000,000 events, 700 particles × [pT, η, φ] = 2,100 features/event
- **Signal:** 1,100,000 events (2,101 columns); column 2100 = truth label (0=bg, 1=sig); 100,000 genuine `W' → XY → 4q` at 3.5 TeV
- **Training:** First 5,000 background events only (semi-supervised: no signal in training)
- **Evaluation (architecture):** Events 5,000–1,000,000 background (995,000) + all 100,000 signal
- **Evaluation (mass studies):** 10,000 background (events 5,000–15,000) + 5,000 signal
- **Mass sculpting:** 20,000 background (events 5,000–25,000)

**IMPLEMENTATION FACT:** `stream_filtered_events()` in `src/run_lhco_gnn.py:254` streams HDF5 chunks via PyTables and filters signal on-the-fly by truth column (no data duplication).

**IMPLEMENTATION FACT:** Particle extraction (`extract_particles`, line 82) keeps only pT > 0 particles (variable multiplicity per event, typically 150–300).

---

## 5. Physics Representation

**IMPLEMENTATION FACT (Particle Features):**
```
Input per particle: [pT, η, φ] 
Preprocessing: log1p(pT) compression; event-level η, φ centering (subtract mean)
```

**IMPLEMENTATION FACT (Graph Construction):**
```
ΔR = √(Δη² + Δφ²) with φ wrap-around at 2π
Edges: bidirectional if ΔR < 0.4 (hadronic jet cone)
No self-loops
MET node (optional): [MET, 0.0, atan2(MET_y, MET_x)] appended
```

**IMPLEMENTATION FACT (Four-Vector & Invariant Mass):**
```
E = pT cosh η, px = pT cos φ, py = pT sin φ, pz = pT sinh η
M = √(E² - px² - py² - pz²)  [full event mass]
```

**IMPLEMENTATION FACT (Recursive Soft Drop - `physics_grooming.py`):**
```
Cambridge-Aachen clustering (R=1.0) → top 2 fat jets
Recursive Soft Drop: z = min(pT1,pT2)/(pT1+pT2) > z_cut(ΔR/R₀)^β
z_cut=0.15, β=0.7, R₀=1.0 (Kasieczka et al. 2026 defaults)
Returns (m_raw, m_RSD) in GeV
```

---

## 6. Model Architectures

### 6.1 BaselineMLP (Flat Tabular)
**IMPLEMENTATION FACT:** `src/run_lhco_gnn.py:396` — 4,744 params; 64D input (first 21 particles); encoder 64→32→8, decoder 8→32→64; MSE loss.

### 6.2 NodeGNN (Standalone GAT)
**IMPLEMENTATION FACT:** `src/run_lhco_gnn.py:312` — 6,101 params; GATConv(3→32, 4 heads)→ELU→GATConv(128→32, 1 head)→ELU; Node decoder 32→16→3; no MET head.

### 6.3 NodeGNN (Production: GAT + MET)
**IMPLEMENTATION FACT:** Same architecture as standalone; trained with `include_met_node=True`, `met_loss_weight=0.1`; MET head: global_mean_pool(z) → 32→16→2.

### 6.4 AdversarialNodeGNN (GAT + MET + GRL)
**IMPLEMENTATION FACT:** `src/adversarial_unlearning_ae.py:259` — 7,702 params; adds adversary head on pooled latent: GradReverse(α) → 32→32→16→1 predicting m_jj/1000; λ_adv=0.05.

### 6.5 NodeWiseAE (No Message Passing)
**IMPLEMENTATION FACT:** `src/run_lhco_gnn.py:431` — ~5,800 params; per-particle encoder 3→128→32, decoder 32→16→3; graph edges ignored (ΔR=0.0); MET head from global_mean_pool.

### 6.6 DeepSetsPointNet (Permutation-Invariant)
**IMPLEMENTATION FACT:** `src/run_lhco_gnn.py:480` — ~5,600 params; φ(x_i): 3→64→32; ρ = global_mean_pool(φ); z_i = concat[φ_i, ρ]; decoder 64→32→16→3; MET head on ρ.

### 6.7 ScoreDecorrelatedNodeGNN (Proposed)
**IMPLEMENTATION FACT:** `src/adversarial_unlearning_ae.py:156` — Wraps base NodeGNN; computes scalar anomaly score in forward; applies Score-GRL adversary OR DisCo penalty directly on score; configs exist, **not trained**.

---

## 7. Training Methodology

**VERIFIED MEASUREMENT (Hyperparameters - Architecture Family):**
| Model | Epochs | Batch | LR | Optimizer | λ_MET | λ_adv | Seed |
|---|---|---|---|---|---|---|---|
| BaselineMLP | 20 | 32 | 0.001 | Adam | 0.0 | 0.0 | 42 |
| GAT Standalone | 20 | 32 | 0.001 | Adam | 0.0 | 0.0 | 42 |
| GAT+MET Production | 20 | 32 | 0.001 | Adam | 0.1 | 0.0 | 42 |
| GAT+MET+Adv | 15 | 32 | 0.001 | Adam | 0.1 | 0.05 | 42 |

**IMPLEMENTATION FACT:** Unified training engine `src/train_experiment.py` reads all params from JSON config; sets seeds for Python/NumPy/PyTorch/CUDA/MPS; writes checkpoint, config, training_log to `experiments/<family>/<name>/`.

**IMPLEMENTATION FACT:** GRL α schedule: `α = 2/(1+exp(-10p)) - 1` where `p = epoch/epochs` (Ganin et al.)

---

## 8. Anomaly Score Definition

**IMPLEMENTATION FACT (Standard):**
```
Score = (1/N) Σ_i ||recon_x_i - x_i||² = MSE over 3 features per node, averaged per event
```

**IMPLEMENTATION FACT (Ablation Scores — `evaluate_controlled_ablation.py:203-219`):**
```
Full:      mean((recon - x)² over all 3 features)
pT-only:   (recon_pT - pT)²
Angular:   mean((recon_η - η)² + (recon_φ - φ)²)
Weighted:  0.8 × pT_err + 0.2 × angular_err
```

**VERIFIED MEASUREMENT:** pT error contributes 80.5% of background loss; angular 19.5%.

---

## 9. Evaluation Methodology

**IMPLEMENTATION FACT (ROC AUC):** `sklearn.metrics.roc_auc_score` on combined background (label 0) + signal (label 1) scores.

**IMPLEMENTATION FACT (Background Rejection):** 1/FPR at TPR ≥ 30% and TPR ≥ 50%.

**IMPLEMENTATION FACT (Mass-Matched ROC):** Restrict both bg and sig to m_jj ∈ [3300, 3700] GeV (±200 GeV around 3.5 TeV), then compute AUC.

**IMPLEMENTATION FACT (Micro-Fine Mass Windows):** Progressive narrowing: ±200, ±150, ±100, **±50 GeV**; also 100-GeV slices across 3–4 TeV.

**IMPLEMENTATION FACT (Mass Sculpting / 1/JSD):** 
- Reference P: all background masses (20k events)
- Selected Q: masses of events with score > τ (top 10%/5%/1%)
- JSD computed on 40 bins [2000, 5000] GeV with eps=1e-12 regularization
- **Inconsistency:** `evaluate_mass_sculpting.py` uses natural log; `evaluate_controlled_ablation.py` uses base-2 log

**IMPLEMENTATION FACT (Ridge Probe R²):** 80/20 temporal split; Ridge(α=1.0) predicting mass/1000 from pooled latent z̄ or anomaly score.

---

## 10. Original Baseline Results (Frozen at `pre-experiments-684c99e`)

**VERIFIED MEASUREMENT (from `BASELINE.md`, `discovery_exhibition_proof.png`):**
- ROC AUC: 0.6414
- Separation Ratio: 1.13x
- Significance: 0.37σ
- Real-time median latency: 9.37 ms (MPS)

---

## 11. New Experimental Findings (Architecture Ablation)

| Finding | Result | Label |
|---|---|---|
| **Relational GAT outperforms MLP** | AUC 0.7815 vs 0.4848 (995k/100k eval) | **VERIFIED MEASUREMENT** |
| **MET head reduces discrimination but improves mass preservation** | AUC: 0.7815→0.6584; 1/JSD Raw@10%: 303→427 | **VERIFIED MEASUREMENT** |
| **Adversarial decorrelation (pooled latent) fails** | 1/JSD: 427→351; R²(score→mass): 0.0025→0.0043 (worse) | **VERIFIED MEASUREMENT** |
| **pT-only score best** | Matched AUC: full=0.6804, pT=0.7676, ang=0.4172 | **VERIFIED MEASUREMENT** |
| **Angular score inverted** | AUC < 0.50 → signal reconstructs angles better than bg | **VERIFIED MEASUREMENT** |
| **Mass-matched discrimination survives** | AUC 0.6804 at [3.3,3.7] TeV; 0.6719 at ±50 GeV | **VERIFIED MEASUREMENT** |
| **Graph ablation: disconnected > relational > scrambled** | Matched AUC: 0.7932 > 0.6804 > 0.4709 | **VERIFIED MEASUREMENT** (inference-time) |
| **MET regularizes via momentum conservation** | True MET ~75 GeV for both; predicts detector noise | **SCIENTIFIC INTERPRETATION** |

---

## 12. Mass-Sculpting Investigation

**VERIFIED MEASUREMENT (1/JSD at 20k events, top 10%):**
| Model | Raw Mass | RSD Mass |
|---|---|---|
| BaselineMLP | 414.0 ± 83.8 | 441.8 ± 85.6 |
| GAT Standalone | 303.4 ± 75.0 | 218.1 ± 74.8 |
| GAT+MET Production | **427.2 ± 83.1** | **423.7 ± 84.4** |
| GAT+MET+Adv | 350.9 ± 87.9 | 257.8 ± 81.3 |

**SCIENTIFIC INTERPRETATION:** Standalone GAT sculpts mass because reconstruction error grows with event hardness (high pT → high mass). MET head enforces ∑pT=0, preventing mass bias. Adversary on pooled latent fails to constrain node decoder.

---

## 13. Graph Structure Findings

**VERIFIED MEASUREMENT (Inference-time interventions on trained GAT Standalone):**
| Graph Mode | Matched AUC [3.3, 3.7 TeV] | Δ vs Relational |
|---|---|---|
| Relational (ΔR<0.4) | 0.6804 | Baseline |
| Scrambled (permuted edges) | 0.4709 | -0.2095 (worse than random) |
| Disconnected (self-loops only) | **0.7932** | +0.1128 |

**SCIENTIFIC INTERPRETATION:** Soft radiation in dense clusters averages down hard-prong anomaly signals. Disconnected mode isolates per-particle kinematics. Scrambled topology destroys physical message passing.

**HYPOTHESIS:** Independently trained NodeWiseAE (same architecture as disconnected inference) will match or exceed disconnected GAT performance. **NOT YET TESTED** — only 1 of 3 seeds trained.

---

## 14. Score-Level Decorrelation Direction

**IMPLEMENTATION FACT (Proposed, `src/adversarial_unlearning_ae.py`):**

**Approach A: Score-Level GRL**
```
Score = mean_node_error → GradReverse(α) → MLP → pred_mass
L = L_recon + λ_MET·L_MET + λ_adv·MSE(pred_mass, mass)
```

**Approach B: Differentiable Distance Correlation (DisCo)**
```
L = L_recon + λ_MET·L_MET + λ_disco·dCor²(Scores, masses)
dCor² = 0 iff statistically independent (captures non-linear dependence)
```

**CONFIG STATUS:** 9 configs in `configs/score_decorrelation/` — **0 trained, 0 evaluated**

**PROPOSED FUTURE WORK:** Train and evaluate both approaches; compare to pooled-latent adversary.

---

## 15. Limitations

| Limitation | Severity | Evidence |
|---|---|---|
| **Adversarial decorrelation ineffective** | High | 1/JSD worse; R² score→mass higher than baseline |
| **Score decorrelation not trained** | High | 9 configs proposed, 0 executed |
| **Controlled structure mostly untrained** | High | 11/12 configs pending |
| **Angular loss not periodic** | Medium | MSE on φ ignores 2π wrap; may cause angular inversion |
| **No ROC AUC uncertainty** | Medium | No bootstrap/DeLong CIs |
| **Population mismatch across evals** | Medium | 995k vs 10k vs 20k events |
| **Single signal topology** | Medium | Only W'→XY at 3.5 TeV |
| **Inverse JSD not physical smoothness** | Low | Monotonic but non-linear transform |

---

## 16. Remaining Validation

| Validation | Status | Required |
|---|---|---|
| Train all 11 remaining controlled structure configs | Pending | Kaggle GPU time |
| Train and evaluate 9 score decorrelation configs | Pending | Kaggle GPU time |
| Implement periodic φ loss (von Mises/wrapped MSE) | Proposed | Code change in `train_experiment.py` |
| Bootstrap ROC AUC confidence intervals | Proposed | Add to evaluation scripts |
| Multi-signal generalization (SUSY, dark sector, etc.) | Proposed | Additional datasets |
| Systematic ΔR sweep (0.2, 0.4, 0.6) | Configs exist | Training execution |
| Statistical significance (p-values, CLs) | Missing | Add to `ultimate_discovery_proof.py` |

---

## 17. Potential Contributions (If Completed)

1. **First demonstration that pT-only reconstruction outperforms full 3D in GNN anomaly detection** with angular inversion explanation
2. **Systematic graph structure ablation** showing soft radiation message passing degrades discrimination
3. **MET prediction as implicit mass regularizer** outperforming explicit adversarial decorrelation
4. **Architectural diagnosis of adversarial bypass** with score-level fixes (Score-GRL, DisCo)
5. **Ultra-fine mass matching (±50 GeV)** showing GNN learns genuine substructure
6. **Controlled comparison framework** with config-driven provenance for reproducible HEP ML

---

## 18. Proposed Next Experiments (Priority Order)

| Priority | Experiment | Configs | Expected GPU Time |
|---|---|---|---|
| **1** | Complete controlled structure (11 configs × 3 seeds) | `nodewise_ae_s{42,123}`, `deepsets_pointnet_s{42,123,999}`, `gat_relational_s{42,123,999}`, `baseline_mlp_s{42,123,999}` | ~11 × 20 epochs × 5k events ≈ 5-10 hrs |
| **2** | Score decorrelation (9 configs) | `gat_score_grl_s{42,123,999}`, `gat_score_disco_s{42,123,999}`, `gat_score_disco_heavy_s42`, `gat_score_disco_weighted_s42`, `gat_baseline_ref_s42` | ~9 × 20 epochs ≈ 5-10 hrs |
| **3** | Periodic φ loss + retrain architecture family | Modify loss in `train_experiment.py`; re-run 4 arch | ~4 × 20 epochs ≈ 2-4 hrs |
| **4** | Graph construction sweep | `dr_02`, `dr_04_baseline`, `dr_06` | ~3 × 20 epochs ≈ 2-4 hrs |
| **5** | MET weight sweep | `lambda_0`, `lambda_005`, `lambda_020` | ~3 × 20 epochs ≈ 2-4 hrs |
| **6** | Multi-signal benchmark | Download additional LHCO signals | Dataset dependent |

---

**End of Manuscript-Ready Summary**