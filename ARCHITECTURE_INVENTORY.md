# ARCHITECTURE INVENTORY: Complete Model Specifications

**Generated:** October 30, 2026  
**Repository:** `anomaly-detector-gnn`  
**All specifications sourced from actual source code implementations**

---

## 1. Baseline Flat MLP Autoencoder

### Source
- **File:** `src/run_lhco_gnn.py:396-424`
- **Class:** `BaselineMLP`
- **Checkpoint:** `models/autoencoder_bg.pt` (22 KB, 4,744 params)
- **Config:** `configs/architecture/baseline_mlp.json`

### Architecture
```
Input: 64 features (first 64 values of event: 21 particles × [pT, η, φ] + 1 padding)
         log1p(pT) applied to columns 0,3,6,...,63

Encoder:
  Linear(64 → 32) + ELU
  Linear(32 → 8)                    ← Latent bottleneck (8D)

Decoder:
  Linear(8 → 32) + ELU
  Linear(32 → 64)                   ← Reconstruction target: same 64 features
```

### Parameters
- **Total:** 4,744
- **Trainable:** 4,744

### Loss
```
L = MSE(recon_x, x)                     (reconstruction only)
```

### Anomaly Score
```
Score = MSE(recon_x, x) per event (mean over 64 features)
```

### Graph Construction
- **None** — flat tabular representation
- `event_to_flat_features()` truncates to 64 features, applies log1p to pT columns

---

## 2. Standalone Relational GAT Autoencoder

### Source
- **File:** `src/run_lhco_gnn.py:312-389`
- **Class:** `NodeGNN`
- **Checkpoint:** `models/discovery_model.pt` (30 KB, 6,101 params)
- **Config:** `configs/architecture/gat_standalone.json`

### Architecture
```
Input: Graph with node features [log1p(pT), η_centered, φ_centered] (3D per particle)
       Edges: ΔR < 0.4 (bidirectional, no self-loops)
       Optional: MET node [MET, 0.0, φ_MET] appended when include_met_node=True

Encoder (Message Passing):
  GATConv(3 → 32, heads=4, concat=True)  → 128 features + ELU
  GATConv(128 → 32, heads=1, concat=False) → 32 features + ELU
                                              ↑ Latent z_i per node (32D)

Decoder Head 1 (Node Reconstruction):
  Linear(32 → 16) + ELU
  Linear(16 → 3)                          → recon_x: [log1p(pT), η, φ] per node

Decoder Head 2 (Global MET Prediction):
  global_mean_pool(z, batch) → 32D global embedding
  Linear(32 → 16) + ELU
  Linear(16 → 2)                          → pred_met: [MET_x, MET_y]
```

### Parameters
- **Total:** 6,101
- **Trainable:** 6,101

### Loss
```
L = L_recon + λ_MET × L_MET
L_recon = MSE(recon_x, x)                  (node-level, all 3 features)
L_MET = MSE(pred_met, [MET_x, MET_y])      (global, vector)
λ_MET = 0.0 (standalone) or 0.1 (production)
```

### Anomaly Score
```
Score = mean_i MSE(recon_x_i, x_i) = mean_i ||recon_x_i - x_i||² / 3
       = mean_i [(log1p(pT)_i - log1p(pT̂)_i)² + (η_i - η̂_i)² + (φ_i - φ̂_i)²] / 3
```

### Graph Construction
- `event_to_graph()` with `delta_r_threshold=0.4`, `include_met_node=False`
- Node features: log1p(pT), η (centered), φ (centered)
- Edges: Vectorized pairwise ΔR < 0.4 with φ wrap-around

---

## 3. Multi-Task GAT + MET (Production)

### Source
- **File:** `src/run_lhco_gnn.py:312-389` (same class `NodeGNN`)
- **Checkpoint:** `experiments/architecture/gat_met_production/checkpoint.pt`
- **Config:** `configs/architecture/gat_met_production.json`

### Architecture
**Identical to Standalone GAT** — only difference is training configuration:
- `include_met_node=True` (MET node added to graph)
- `met_loss_weight=0.1` (MET prediction active)
- `phi_met_mode="atan2"` (physical MET φ from atan2)

### Parameters
- **Total:** 6,101 (same as standalone)

### Loss
```
L = L_recon + 0.1 × L_MET
```

---

## 4. GAT + MET + Adversarial Mass Decorrelation (Pooled Latent GRL)

### Source
- **File:** `src/adversarial_unlearning_ae.py:259-322`
- **Class:** `AdversarialNodeGNN`
- **Checkpoint:** `models/unlearning_model.pt` (38 KB, 7,702 params) and `experiments/architecture/gat_met_decorrelation/checkpoint.pt`
- **Config:** `configs/architecture/gat_met_decorrelation.json`

### Architecture
```
Input: Same as NodeGNN (with MET node)

Encoder (shared with NodeGNN):
  GATConv(3 → 32, heads=4, concat=True)  → 128 + ELU
  GATConv(128 → 32, heads=1, concat=False) → 32 + ELU
  global_mean_pool(z, batch) → z̄ (32D global)

Decoder Head 1 (Node Reconstruction):
  Linear(32 → 16) + ELU
  Linear(16 → 3)                          → recon_x

Decoder Head 2 (Global MET Prediction):
  Linear(32 → 16) + ELU
  Linear(16 → 2)                          → pred_met

Adversary Head (Invariant Mass Regressor on Pooled Latent):
  GradReverse(α=1.0) on z̄
  Linear(32 → 32) + ELU
  Linear(32 → 16) + ELU
  Linear(16 → 1)                          → pred_mass (TeV)
```

### Parameters
- **Total:** 7,702
- **Trainable:** 7,702

### Loss
```
L = L_recon + 0.1 × L_MET + 0.05 × L_adv
L_adv = MSE(pred_mass, m_jj / 1000)        (mass normalized to TeV)

GRL Schedule: α = 2/(1+exp(-10p)) - 1, p = epoch/epochs  (Ganin et al.)
```

### Anomaly Score
```
Score = MSE(recon_x, x) per event (node-level, same as NodeGNN)
⚠️ CRITICAL: Adversary acts on z̄, but score computed from node decoder — BYPASS
```

---

## 5. Node-Wise Autoencoder (No Message Passing)

### Source
- **File:** `src/run_lhco_gnn.py:431-473`
- **Class:** `NodeWiseAE`
- **Checkpoint:** `experiments/controlled_structure/nodewise_ae_s999/checkpoint.pt` (28 KB)
- **Config:** `configs/controlled_structure/nodewise_ae_s42.json` (and s123, s999)

### Architecture
```
Input: Node features [log1p(pT), η, φ] (3D) — graph structure IGNORED
       edge_index passed but NOT USED (only for batch compatibility)

Encoder (per-particle, independent):
  Linear(3 → 128) + ELU
  Linear(128 → 32) + ELU                    → z_i (32D per node)

Decoder (Node Reconstruction):
  Linear(32 → 16) + ELU
  Linear(16 → 3)                            → recon_x_i

MET Head (Global):
  global_mean_pool(z, batch) → z̄
  Linear(32 → 16) + ELU
  Linear(16 → 2)                            → pred_met
```

### Parameters
- **Total:** ~5,800 (matched to NodeGNN budget)
- **Trainable:** ~5,800

### Loss
```
L = L_recon + λ_MET × L_MET  (λ_MET = 0.0 in controlled_structure configs)
```

### Graph Construction
- `delta_r_threshold=0.0` → `build_deltaR_edges()` returns empty edge_index
- `include_met_node=False`
- Effectively: each particle processed independently (self-loop equivalent)

---

## 6. DeepSets / PointNet (Permutation-Invariant Point Cloud)

### Source
- **File:** `src/run_lhco_gnn.py:480-530`
- **Class:** `DeepSetsPointNet`
- **Checkpoint:** None trained yet
- **Config:** `configs/controlled_structure/deepsets_pointnet_s42.json` (and s123, s999)

### Architecture
```
Input: Node features [log1p(pT), η, φ] (3D) — graph structure IGNORED

Particle Encoder φ(x_i):
  Linear(3 → 64) + ELU
  Linear(64 → 32) + ELU                    → φ_i (32D per node)

Global Aggregator ρ:
  global_mean_pool(φ, batch) → ρ (32D global)

Context Conditioning:
  z_i = concat([φ_i, ρ[batch[i]]])         → 64D per node (local + global)

Conditioned Decoder:
  Linear(64 → 32) + ELU
  Linear(32 → 16) + ELU
  Linear(16 → 3)                            → recon_x_i

MET Head:
  Linear(32 → 16) + ELU (on ρ)
  Linear(16 → 2)                            → pred_met
```

### Parameters
- **Total:** ~5,600 (matched to NodeGNN budget)
- **Trainable:** ~5,600

### Loss
```
L = L_recon + λ_MET × L_MET  (λ_MET = 0.0 in controlled_structure configs)
```

### Graph Construction
- `delta_r_threshold=0.0` → no edges
- `include_met_node=False`
- Pure point-cloud: global context via mean pooling, no pairwise edges

---

## 7. Score-Level Decorrelation: ScoreDecorrelatedNodeGNN

### Source
- **File:** `src/adversarial_unlearning_ae.py:156-221`
- **Class:** `ScoreDecorrelatedNodeGNN` (wraps base `NodeGNN`)
- **Checkpoint:** None trained
- **Configs:** `configs/score_decorrelation/*.json` (7 configs, all PROPOSED)

### Architecture
```
Base: NodeGNN (any variant) → produces recon_x, pred_met

Score Computation (in forward pass):
  if score_type == "pt_only":
      node_err = (recon_x[:, 0] - x[:, 0])²
  elif score_type == "weighted":
      pt_err = (recon_x[:, 0] - x[:, 0])²
      ang_err = mean((recon_x[:, 1:] - x[:, 1:])², dim=1)
      node_err = w_pt × pt_err + w_ang × ang_err  (default 0.8/0.2)
  elif score_type == "ang_only":
      node_err = mean((recon_x[:, 1:] - x[:, 1:])², dim=1)
  else:  # "full"
      node_err = mean((recon_x - x)², dim=1)

  Event Score = mean(node_err) per graph (scatter_add_ / counts)

Decorator A: Score-GRL Adversary (if decorrelation_type="grl")
  ScoreGRLAdversary:
    GradReverse(α) on scalar Score
    Linear(1 → 32) + ELU
    Linear(32 → 16) + ELU
    Linear(16 → 1)                → pred_mass
  L_adv = MSE(pred_mass, m_jj)

Decorator B: DisCo Penalty (if decorrelation_type="disco")
  L_adv = dCor²(Scores, m_jj)     (differentiable distance correlation)
  No separate adversary network needed
```

### Parameters
- **Base NodeGNN:** 6,101
- **ScoreGRLAdversary:** ~500 additional
- **Total (GRL):** ~6,600
- **Total (DisCo):** 6,101 (no extra params)

### Loss
```
GRL: L = L_recon + 0.1 × L_MET + λ_adv × L_adv(Score, mass)
DisCo: L = L_recon + 0.1 × L_MET + λ_disco × dCor²(Score, mass)
```

### Key Configs
| Config | Decorrelation | Score Type | λ_adv/λ_disco | Status |
|---|---|---|---|---|
| `gat_score_grl_s42` | GRL | pt_only | 0.05 | Proposed |
| `gat_score_disco_s42` | DisCo | pt_only | 1.0 | Proposed |
| `gat_score_disco_heavy_s42` | DisCo | pt_only | 2.0 | Proposed |
| `gat_score_disco_weighted_s42` | DisCo | weighted (0.8/0.2) | 1.0 | Proposed |

---

## 8. Legacy / Unused Models

### EventGNN (GCN Classifier)
- **File:** `src/run_lhco_gnn.py:285-305`
- **Class:** `EventGNN`
- **Checkpoint:** `models/gnn_extractor.pt` (585 KB)
- **Architecture:** 2-layer GCNConv(3→64→64) + global_mean_pool + Linear(64→2)
- **Purpose:** Graph classification (not autoencoder); legacy, not used in experiments

---

## 9. Architecture Comparison Summary

| Model | Params | Message Passing | MET Head | Adversary | Score Decorrelation | Trained |
|---|---|---|---|---|---|---|
| BaselineMLP | 4,744 | ❌ Flat | ❌ | ❌ | ❌ | ✅ (baseline) |
| NodeGNN (standalone) | 6,101 | ✅ GAT (4-head) | ❌ | ❌ | ❌ | ✅ |
| NodeGNN (production) | 6,101 | ✅ GAT (4-head) | ✅ | ❌ | ❌ | ✅ |
| AdversarialNodeGNN | 7,702 | ✅ GAT (4-head) | ✅ | ✅ GRL on z̄ | ❌ | ✅ |
| NodeWiseAE | ~5,800 | ❌ Independent | ✅ | ❌ | ❌ | ✅ (1 seed) |
| DeepSetsPointNet | ~5,600 | ❌ Global pool | ✅ | ❌ | ❌ | ❌ |
| ScoreDecorrelatedNodeGNN | ~6,600 | ✅ GAT (4-head) | ✅ | ❌ | ✅ GRL/DisCo on Score | ❌ |

---

## 10. Architecture Diagrams (Textual)

### NodeGNN / AdversarialNodeGNN Forward Pass
```
Event Particles (N × 3) → event_to_graph() → PyG Data(x, edge_index, y=[MET_x, MET_y])
                                                                      │
                                                                      ▼
                                                  ┌───────────────────────────────┐
                                                  │         Encoder (GAT)         │
                                                  │  GATConv(3→32, h=4) → ELU     │
                                                  │  GATConv(128→32, h=1) → ELU   │
                                                  │  Output: z (N × 32)           │
                                                  └───────────────┬───────────────┘
                                                                    │
                              ┌───────────────────────┬─────────────┴─────────────┬───────────────────────┐
                              ▼                       ▼                           ▼
                    ┌───────────────┐         ┌───────────────┐         ┌───────────────────┐
                    │ Node Decoder  │         │  MET Head     │         │   Adversary       │
                    │ Linear(32→16) │         │ Linear(32→16) │         │ (AdversarialNode- │
                    │ ELU           │         │ ELU           │         │  GNN only)        │
                    │ Linear(16→3)  │         │ Linear(16→2)  │         │ GradReverse(α)    │
                    │ recon_x       │         │ pred_met      │         │ Linear(32→32)     │
                    └───────┬───────┘         └───────┬───────┘         │ ELU               │
                            │                       │                   │ Linear(32→16)     │
                            │                       │                   │ ELU               │
                            │                       │                   │ Linear(16→1)      │
                            │                       │                   │ pred_mass         │
                            ▼                       ▼                   └───────────────────┘
                     ┌─────────────────────────────────────────────────────────────────┐
                     │                    Loss Computation                              │
                     │  L_recon = MSE(recon_x, x)                                       │
                     │  L_MET = MSE(pred_met, y)                                        │
                     │  L_adv = MSE(pred_mass, mass/1000)  [if adversarial]             │
                     │  L_total = L_recon + λ_MET·L_MET + λ_adv·L_adv                   │
                     └─────────────────────────────────────────────────────────────────┘
```

### ScoreDecorrelatedNodeGNN Score Computation
```
recon_x, pred_met = base_model(x, edge_index, batch)
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │   Per-Node Error Decomposition  │
                            │  diff = (recon_x - x)²          │
                            │  pt_err   = diff[:, 0]          │
                            │  ang_err  = mean(diff[:, 1:], 1)│
                            └───────────────┬─────────────────┘
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                        score_type:    score_type:   score_type:
                        "pt_only"      "weighted"    "full"
                        node_err       0.8*pt +      mean(diff, 1)
                                       0.2*ang
                              │             │             │
                              └─────────────┼─────────────┘
                                            ▼
                            ┌───────────────────────────────┐
                            │   Event Score = mean(node_err)  │
                            │   (scatter_add_ over batch)     │
                            └───────────────┬─────────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                        Decor: "grl"                 Decor: "disco"
                        ScoreGRLAdversary(Score)      dCor²(Score, mass)
                        → pred_mass                   → scalar penalty
                        L_adv = MSE                   L_adv = dCor²
```

---

## 11. Implementation Status Matrix

| Component | File | Class | Strict Load Test | Unit Tests | Trained Checkpoints |
|---|---|---|---|---|---|
| BaselineMLP | run_lhco_gnn.py | BaselineMLP | ✅ | ✅ | 1 (baseline + exp) |
| NodeGNN | run_lhco_gnn.py | NodeGNN | ✅ | ✅ | 2 (baseline + exp) |
| AdversarialNodeGNN | adversarial_unlearning_ae.py | AdversarialNodeGNN | ✅ | ✅ | 2 (baseline + exp) |
| NodeWiseAE | run_lhco_gnn.py | NodeWiseAE | N/A | ✅ | 1 (s999) |
| DeepSetsPointNet | run_lhco_gnn.py | DeepSetsPointNet | N/A | ✅ | 0 |
| ScoreDecorrelatedNodeGNN | adversarial_unlearning_ae.py | ScoreDecorrelatedNodeGNN | N/A | ✅ | 0 |
| dCor² | adversarial_unlearning_ae.py | compute_distance_correlation | N/A | ✅ | N/A |
| ScoreGRLAdversary | adversarial_unlearning_ae.py | ScoreGRLAdversary | N/A | ✅ | N/A |
| C/A + RSD | physics_grooming.py | cluster_cambridge_aachen, recursive_soft_drop | N/A | ✅ | N/A |

---

**End of Architecture Inventory**