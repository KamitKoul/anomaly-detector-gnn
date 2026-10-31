# Phase 0 Baseline Audit: Data-Scaling Research Experiment

**Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Target:** Background-Data Scaling Study for PointNet–GAT Hybrid Anomaly Detector  
**Author:** Antigravity Senior ML Research Engineer (HEP Specialist)  
**Date:** October 31, 2026  
**Status:** Audit Complete — Ready to Freeze Protocol  

---

## 1. Executive Summary

This audit establishes the empirical, architectural, and data-engineering ground truth of the repository prior to launching the large-scale background data scaling study. Every statement in this audit has been verified directly against source code in `src/`, `evaluation/`, `scripts/`, `configs/`, and `tests/`.

### Core Verdict
1. **Model Architecture is Verified and Intact:** The active `PointNetGATHybrid` model contains **exactly 7,957 parameters** and supports pure reconstruction-only training without architectural modification by setting `met_loss_weight: 0.0` and `adv_loss_weight: 0.0`.
2. **Data Availability vs Historical Usage:** While **1,000,000 background events** and **100,000 signal events** are available in the LHC Olympics 2020 dataset, all historical experiments were trained on only **5,000 background events** (0.5% utilization) and evaluated on **10,000 background / 5,000 signal events**.
3. **Partition & Scaling Preparedness:** We must enforce strict deterministic nested subsets ($D_0 \subset D_1 \subset \dots \subset D_6$) using a dedicated generator (`src/data/scaling_split.py`) to guarantee zero test leakage and strict event provenance.
4. **Experimental Isolation:** Experiment A (Fixed Epochs: 20 epochs across $D_0$–$D_6$) and Experiment B (Fixed Updates: exactly 10,000 optimizer steps across $D_0$–$D_6$) decouple data exposure from optimizer update counts.

---

## 2. Exhaustive Audit of the 16 Repository Subsystems

### Subsystem 1: Current PointNet–GAT Architecture
* **Source Location:** [`src/run_lhco_gnn.py:PointNetGATHybrid`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/run_lhco_gnn.py#L580-L656)
* **Design:** Combines a local relational Graph Attention Network (GAT) branch with a global permutation-invariant PointNet branch.
* **Component Flow:**
  1. *Local Relational GAT Branch:*
     - `conv1`: `GATConv(in=3, out=16, heads=4, concat=True)` $\to$ ELU $\to \mathbb{R}^{64}$.
     - `conv2`: `GATConv(in=64, out=32, heads=1, concat=False)` $\to$ ELU $\to \mathbb{R}^{32}$.
  2. *Global PointNet Branch:*
     - `point_phi`: `Linear(3, 32) -> ELU -> Linear(32, 32) -> ELU` $\to \mathbb{R}^{32}$.
     - `rho`: `global_mean_pool(phi, batch)` $\to \mathbb{R}^{32}$ (event-level context).
  3. *Multi-Scale Fusion:*
     - Node concatenation: $\mathbf{h}_{\text{fused}} = [\mathbf{h}_{\text{GAT}} \, (32) \,\|\, \boldsymbol{\phi}_i \, (32) \,\|\, \boldsymbol{\rho} \, (32)] \in \mathbb{R}^{96}$.
  4. *Conditioned Kinematic Reconstruction Decoder:*
     - `Linear(96, 32) -> ELU -> Linear(32, 16) -> ELU -> Linear(16, 3)` predicting reconstructed $[\ln(1+p_T), \eta, \phi]$.
  5. *Global MET Head:*
     - `Linear(32, 16) -> ELU -> Linear(16, 2)` predicting $[\text{MET}_x, \text{MET}_y]$.

### Subsystem 2: Parameter Count & Architecture Budget
* **Verified Parameter Count:** **7,957 total parameters** (7,957 trainable, 0 frozen).
* **Layer-by-Layer Breakdown:**
  - `conv1` (`att_src`: 64, `att_dst`: 64, `bias`: 64, `lin.weight`: 192) = 384
  - `conv2` (`att_src`: 32, `att_dst`: 32, `bias`: 32, `lin.weight`: 2,048) = 2,144
  - `point_phi` (`0.weight`: 96, `0.bias`: 32, `2.weight`: 1,024, `2.bias`: 32) = 1,184
  - `decoder` (`0.weight`: 3,072, `0.bias`: 32, `2.weight`: 512, `2.bias`: 16, `4.weight`: 48, `4.bias`: 3) = 3,683
  - `met_head` (`0.weight`: 512, `0.bias`: 16, `2.weight`: 32, `2.bias`: 2) = 562
  - **Total:** $384 + 2144 + 1184 + 3683 + 562 = \mathbf{7,957}$.

### Subsystem 3: Dataset Loading & Preprocessing Pipeline
* **Source Location:** [`src/streaming_data_loader.py:ChunkedLHCODataset`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/streaming_data_loader.py#L30-L162) and [`src/run_lhco_gnn.py:event_to_graph`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/run_lhco_gnn.py#L220-L260)
* **Mechanism:**
  - On-the-fly streaming from HDF5 chunks without in-memory accumulation.
  - Particle filtering: parses only particles with $p_T > 0$.
  - Scaling: $p_T \to \ln(1 + p_T)$, retaining continuous coordinates for $\eta$ and $\phi$.
  - Single particle extraction reuse: avoids redundant extraction per event.
  - Peak memory footprint bounded to $< 500$ MB RAM.

### Subsystem 4: HDF5 Data Organization
* **Background File:** `data/events_LHCO2020_backgroundMC_Pythia.h5`
  - Node path: `/df/block0_values` (PyTables fixed DataFrame table).
  - Shape: $(1000000, 2100)$ float64.
  - Structure: 700 particles $\times$ 3 features $(p_T, \eta, \phi)$.
* **Signal File:** `data/W_prime_signal.h5`
  - Node path: `/df/block0_values`.
  - Shape: $(100000, 2101)$ float64.
  - Structure: 2,100 particle features + 1 truth label column (`truth == 1.0`).

### Subsystem 5: Background & Signal Event Indexing
* **Identification:** Positional integer indexing $[0, N-1]$. The raw files contain no collision run numbers or event IDs.
* **Integrity Mandate:** Splitting must rely on deterministic integer permutations saved to reproducible `.npy` index arrays with cryptographic SHA-256 validation.

### Subsystem 6: Existing Train, Validation, and Test Splits
* **Historical Baseline:**
  - Train: Sequential slice $[0, 5000)$ of background.
  - Validation: None (0 events). Checkpoints saved on training loss.
  - Test: Sequential slice $[5000, 15000)$ (10,000 background) and $[0, 5000)$ (5,000 signal).
* **Target Scaling Split:**
  - Total Background Pool: 1,000,000 events.
  - Training Pool: **900,000 events** (partitioned into nested tiers $D_0 \subset \dots \subset D_6$).
  - Validation: **50,000 background events** (strictly held-out).
  - Final Test: **50,000 background events** (strictly held-out).
  - Signal: 25,000 test, 25,000 val (diagnostic), 50,000 reserve.

### Subsystem 7: Graph Construction Algorithm & $\Delta R$ Threshold
* **Source Location:** [`src/run_lhco_gnn.py:build_deltaR_edges`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/run_lhco_gnn.py#L182-L215)
* **Connectivity Metric:**
  $$\Delta R = \sqrt{(\eta_u - \eta_v)^2 + \Delta \phi(u, v)^2} < 0.4$$
* **Periodic Wrap-Around:** $\Delta \phi = \min(|\phi_u - \phi_v|, 2\pi - |\phi_u - \phi_v|)$ on $[0, \pi]$.
* **Optimization:** Evaluated via $(\Delta R)^2 < 0.4^2$ with in-place diagonal $\infty$ masking, strictly excluding self-loops.

### Subsystem 8: Reconstruction Loss & Anomaly-Score Implementation
* **Loss Function:** [`src/run_lhco_gnn.py:compute_kinematic_recon_loss`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/run_lhco_gnn.py#L150-L177)
  $$\mathcal{L}_{\text{recon}} = w_{p_T} \text{MSE}(\hat{p}_T, p_T) + w_{\text{ang}} \frac{1}{2}\left[\text{MSE}(\hat{\eta}, \eta) + \text{periodic\_MSE}(\hat{\phi}, \phi)\right]$$
* **Anomaly-Score Formulations:**
  1. Full reconstruction score: $S_{\text{full}} = \frac{1}{3}(S_{p_T} + S_\eta + S_\phi)$.
  2. $p_T$-only score: $S_{p_T} = \text{MSE}(\hat{p}_T, p_T)$.
  3. Angular score: $S_{\text{ang}} = \frac{1}{2}(\text{MSE}_\eta + \text{MSE}_\phi)$.
  4. Weighted score: $S_{\text{weighted}} = 0.8 S_{p_T} + 0.2 S_{\text{ang}}$.

### Subsystem 9: MET Auxiliary Prediction Head & Loss
* **Implementation:** `self.met_head(rho)` predicts $[\text{MET}_x, \text{MET}_y]$.
* **Scaling Experiment Protocol:** **Disabled** ($\lambda_{\text{MET}} = 0.0$).
* **Verification:** Setting `met_loss_weight = 0.0` completely bypasses the MET loss term during backpropagation in [`src/train_experiment.py:279`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/train_experiment.py#L279), zeroing any auxiliary gradients while keeping the 562-parameter head intact.

### Subsystem 10: DisCo Implementation & Mathematical Objective
* **Source Location:** [`src/stable_disco.py`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/stable_disco.py)
* **Scaling Experiment Protocol:** **Disabled** ($\lambda_{\text{adv}} = 0.0$).
* **Verification:** Preserved in the codebase for subsequent decorrelation studies, but strictly inactive in the primary data-scaling runs.

### Subsystem 11: Training Loop & Optimizer Configuration
* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
* **Learning Rate:** $0.001$, Weight Decay: $0.0$.
* **Gradient Clipping:** Maximum $L_2$ norm = $50.0$.
* **Batch Size:** Fixed at $32$.

### Subsystem 12: Checkpoint Selection Implementation
* **Source Location:** [`src/validation_pipeline.py:CheckpointManager`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/validation_pipeline.py#L58-L100)
* **Integrity Gate:** Tracks strictly background validation reconstruction loss (`val_loss_recon`).
* **Zero Signal Leakage:** Signal events are never seen during validation, checkpoint ranking, or training.

### Subsystem 13: Existing Evaluation & Mass-Sculpting Calculations
* **Source Location:** [`evaluation/evaluate_controlled_ablation.py`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_controlled_ablation.py)
* **Metrics:**
  - Inclusive ROC-AUC ($m_{jj} \in [2.5, 5.0]$ TeV).
  - Mass-matched ROC-AUC ($m_{jj} \in [3.3, 3.7]$ TeV).
  - Micro-fine ROC-AUC ($m_{jj} \in [3.45, 3.55]$ TeV).
  - Background rejection ($1/\text{FPR}$) at $\epsilon_S = 0.30, 0.50$.
  - Base-2 JSD mass sculpting at Top 10%, 5%, 1% cuts with bootstrap uncertainties.
  - Linear and rank correlations (Pearson $r$, Spearman $\rho$, DisCo).

### Subsystem 14: Existing Experiment Configuration Generator
* **Current Generators:** `scripts/generate_hybrid_configs.py` (legacy 5k configs).
* **Scaling Requirement:** A centralized YAML configuration file `configs/scaling/scaling_baseline.yaml` that explicitly defines all hyperparameters, paths, and constraints.

### Subsystem 15: Existing Tests & Coverage
* **Status:** 67 passing unit tests across 15 test files covering model instantiation, gradient flow, periodic losses, DisCo stability, and GPU pipeline vectorization.

### Subsystem 16: Kaggle GPU Execution & Memory Requirements
* **Target Hardware:** NVIDIA Tesla T4 (16 GB GDDR6) or Tesla P100 (16 GB HBM2).
* **Memory Bounding:** Peak VRAM $< 1.5$ GB, Peak Host RAM $< 500$ MB.
* **Throughput:** ~3,140 events/sec (~98 batches/sec).

---

## 3. Discrepancies, Missing Components & Scientific Risks

| Issue / Discrepancy | Current State | Risk | Required Modification |
| :--- | :--- | :--- | :--- |
| **Fixed Updates Runner** | Only fixed-epoch training existed | Data scale and update counts were confounded | Implement `experiments/scaling/experiment_b_fixed_updates.py` with infinite cyclical sampler |
| **Split Generation Script** | Ad-hoc array generation | Incomplete provenance and manifest hashing | Implement `src/data/scaling_split.py` with deterministic SHA-256 manifests |
| **YAML Configuration** | Configurations were JSON only | Missed explicit baseline specification | Author `configs/scaling/scaling_baseline.yaml` |
| **Evaluation Split Loading** | `evaluate_controlled_ablation.py` used sequential slicing | Hardcoded `skip=5000` could cause split overlap | Modify evaluation to load explicit index arrays from `data/splits/` |
| **Droplast Policy** | Partial batches were allowed | Update counts varied slightly per epoch | Enforce explicit `drop_last=True` and record dropped event tallies |

---

## 4. Required File Modifications & New Deliverables

1. `configs/scaling/scaling_baseline.yaml` — Master frozen scientific configuration.
2. `src/data/scaling_split.py` — Deterministic nested subset generator ($D_0 \subset \dots \subset D_6$) with manifest hashing.
3. `tests/test_scaling_splits.py` — Automated integrity verification for splits and nested subsets.
4. `experiments/scaling/experiment_a_fixed_epochs.py` — Experiment A runner (20 epochs).
5. `experiments/scaling/experiment_b_fixed_updates.py` — Experiment B runner (exactly 10,000 updates).
6. `reports/scaling/EXPERIMENT_PROTOCOL.md` — Formal scientific protocol.
7. `reports/scaling/DATASET_INTEGRITY_REPORT.md` — Data verification and manifest hashes.
