# Frozen Baseline

## Git State
- **Branch:** `main`
- **Commit:** `684c99e9eab22a5f77c0ce4259ac20b4f0d357e8` (`684c99e`)
- **Tag:** `pre-experiments-684c99e`
- **Working Tree:** Clean prior to creating this manifest
- **Remote Synchronization:** Up to date with `origin/main` at `684c99e`

## Repository State
The repository implements a Geometric Deep Learning pipeline for High Energy Physics (HEP) collider anomaly detection based on the LHC Olympics 2020 (LHCO2020) dataset. Particle collision showers are converted into relational interaction graphs and processed through a multi-task Graph Attention Network (GAT) autoencoder trained on Standard Model background events to detect Beyond the Standard Model (BSM) anomalies via reconstruction error.

## Current Model
- **Primary Model Class:** `NodeGNN` (`src/run_lhco_gnn.py`)
  - **Encoder:**
    - `conv1`: `GATConv(in_channels=3, out_channels=32, heads=4, concat=True)` followed by `ELU` activation (128 output features).
    - `conv2`: `GATConv(in_channels=128, out_channels=32, heads=1, concat=False)` followed by `ELU` activation (32 output features).
    - Backward-compatible property aliases: `gat1` -> `conv1`, `gat2` -> `conv2`.
  - **Decoder Head 1 (Node Reconstruction):**
    - `decoder`: `nn.Sequential(nn.Linear(32, 16), nn.ELU(), nn.Linear(16, 3))`
    - Backward-compatible property alias: `recon_head` -> `decoder`.
  - **Decoder Head 2 (Global MET Prediction):**
    - `met_head`: `nn.Sequential(nn.Linear(32, 16), nn.ELU(), nn.Linear(16, 2))` applied to pooled representation `global_mean_pool(z, batch)`.
- **Secondary Models:**
  - `EventGNN` (`src/run_lhco_gnn.py`): 2-layer `GCNConv` network for graph classification/testing.
  - `AdversarialNodeGNN` (`src/adversarial_unlearning_ae.py`): Multi-task GAT autoencoder combined with an invariant mass adversarial head (`adversary`: `Linear(32, 32)` -> `ELU` -> `Linear(32, 16)` -> `ELU` -> `Linear(16, 1)`) conditioned through a Gradient Reversal Layer (`grad_reverse`, `alpha=1.0`).

## Current Graph Construction
- **Distance Metric:** Relativistic angular distance $\Delta R = \sqrt{(\Delta \eta)^2 + (\Delta \phi)^2}$ with periodic wrap-around $\Delta \phi = \min(|\phi_1 - \phi_2|, 2\pi - |\phi_1 - \phi_2|)$.
- **Threshold:** `DELTA_R_THRESHOLD = 0.4`
- **Implementation:** Vectorized pairwise matrix operations (`build_deltaR_edges` in `src/run_lhco_gnn.py`) using NumPy broadcasting; self-loops excluded.
- **Node Features:** Visible particles with features $[p_T, \eta, \phi]$.
- **Virtual MET Node:** Appended to node array with coordinates $[p_T = \text{MET}, \eta = 0.0, \phi = \phi_{\text{MET}}]$, where $\phi_{\text{MET}} = \text{atan2}(\text{MET}_y, \text{MET}_x)$ if $\text{MET} > 0$ else $0.0$.
- **Feature Scaling:** Transverse momentum log-compression $p_T \to \ln(1 + p_T)$ applied to all node features (`log1p(pT)` via `log_pt=True` in `event_to_graph`).

## Current Objectives
- **Node-Level Reconstruction Objective:**
  $\mathcal{L}_{\text{recon}} = \text{MSE}(\hat{x}, x)$
- **Physics Target Head (Global MET Objective):**
  $\mathcal{L}_{\text{MET}} = \text{MSE}([\hat{\text{MET}}_x, \hat{\text{MET}}_y], [\text{MET}_x, \text{MET}_y])$
- **Multi-Task Discovery Loss (`src/anomaly_detection_ae.py`):**
  $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + 0.1 \times \mathcal{L}_{\text{MET}}$
- **Adversarial Mass-Decorrelation Loss (`src/adversarial_unlearning_ae.py`):**
  $\mathcal{L}_{\text{adv\_total}} = \mathcal{L}_{\text{recon}} + 0.1 \times \mathcal{L}_{\text{MET}} + 0.05 \times \mathcal{L}_{\text{adv}}$, where $\mathcal{L}_{\text{adv}} = \text{MSE}(\hat{m}, m / 1000.0)$ regresses system invariant mass under gradient reversal.

## Existing Checkpoints
- `models/discovery_model.pt` (30 KB): Checkpoint for trained `NodeGNN` (keys: `conv1.*`, `conv2.*`, `decoder.*`, `met_head.*`).
- `models/autoencoder_bg.pt` (22 KB): Checkpoint for baseline tabular MLP autoencoder (`encoder`: 64 -> 32 -> 8, `decoder`: 8 -> 32 -> 64).
- `models/gnn_extractor.pt` (585 KB): Checkpoint for EdgeConv/NNConv GNN feature extractor.
- `models/unlearning_model.pt` (38 KB): Checkpoint for trained `AdversarialNodeGNN`.

## Existing Scientific Outputs
- `discovery_exhibition_proof.png` (98 KB): Anomaly score distributions and CDF separation plot with ROC AUC.
- `discovery_mass_spectrum.png` (63 KB): Bump hunt invariant mass spectrum comparing background vs. AI-selected anomalous events.
- `adversarial_unlearning_proof.png` (25 KB): Invariant mass distributions of decorrelated background vs. signal.
- `anomaly_reasoning_saliency.png` (280 KB): Detector coordinate mapping comparing original particle $p_T$ against loss reconstruction error contributions.
- `gnn_particle_saliency.png` (90 KB): Backpropagation input-gradient loss saliency attribution map ($\|\nabla_x \mathcal{L}\|_2$).
- `event_visualization.png` (388 KB): 2×2 grid of detector views in $(\eta, \phi)$ connected by $\Delta R < 0.4$ edges.
- `gnn_embeddings_tsne.png` (214 KB): 2D t-SNE and PCA projections of the 32D latent graph embeddings.
- `gnn_message_passing.png` (248 KB): GAT attention flow, input graph, and layer activation heatmaps.
- `realtime_latency_distribution.png` (183 KB): Inference latency distribution and real-time streaming anomaly scores.

## Important Baseline Numbers
Values explicitly recorded in the repository documentation, commit records, or generated figures:
- **ROC AUC:** `0.6414` (recorded in `discovery_exhibition_proof.png` and `discovery_mass_spectrum.png`)
- **Separation Ratio:** `1.13x` (recorded in `discovery_exhibition_proof.png` and `discovery_mass_spectrum.png`)
- **Significance:** `0.37σ` (recorded in `discovery_exhibition_proof.png` and `discovery_mass_spectrum.png`)
- **Real-Time Simulation Median Latency:** `9.37 ms` (recorded in `realtime_latency_distribution.png`)
- **Real-Time Simulation P95 Latency:** `19.02 ms` (recorded in `realtime_latency_distribution.png`)
- **Calibrated Anomaly Threshold:** `0.78 MSE` (90th percentile of background, recorded in `src/realtime_simulation.py` and `realtime_latency_distribution.png`)
- **Graph Construction Benchmark:** `0.39 ms` per event (~50x speedup over nested Python loops, recorded in commit `c94d2dd`)

## Experiment Boundary
This commit (`684c99e`) and tag (`pre-experiments-684c99e`) constitute the frozen pre-experiment baseline. No model architectures, training configurations, graph construction logic, or evaluation pipelines should be altered without reference to this baseline.
