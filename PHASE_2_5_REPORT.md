# Phase 2.5 Audit & Implementation Report: Controlled Experiment Pipeline Executable

**Repository:** `anomaly-detector-gnn`  
**Current HEAD:** `33fbc34` ("Phase 2.5: Make controlled experiment pipeline executable with unified training engine and BaselineMLP")  
**Frozen Baseline Tag:** `pre-experiments-684c99e` -> `684c99e`  
**Working Tree:** Clean  
**Remote Sync:** Local branch `main` ahead of `origin/main` by 3 commits (unpushed per constraint)

---

## 1. Executive Summary

Phase 2.5 has resolved all operational and scientific blockers identified in the Phase 2 audit:
1. Established a fully functional runtime environment with PyTorch 2.8.0, PyG 2.6.1, PyTables 3.9.2, and Apple Silicon Metal Performance Shaders (MPS) GPU acceleration.
2. Verified the physical structure of both LHCO datasets and implemented a streaming truth-filter for `data/W_prime_signal.h5`, preventing background contamination in signal evaluations.
3. Implemented `BaselineMLP` reproducing `models/autoencoder_bg.pt` with 100% strict state-dict compatibility.
4. Created a unified, config-driven training engine (`src/train_experiment.py`) supporting all 4 architecture ablation variants with config-controlled loss weights.
5. Implemented evaluation execution in `evaluation/evaluate_single.py` adhering to the experiment output contract.
6. Validated the entire pipeline via 15 passing unit tests, dry-runs on all 4 architecture configs, real data chunk streaming, and mini-batch training smoke tests.
7. **Strict Constraint Adherence:** Zero full training runs or scientific comparisons were executed during Phase 2.5.

---

## 2. Runtime Environment & Dependencies

| Dependency | Version | Status |
| :--- | :--- | :--- |
| Python | 3.9.6 (arm64) | Active in project `.venv/` (gitignored) |
| PyTorch | 2.8.0 | MPS acceleration built and functional (`torch.device("mps")`) |
| PyTorch Geometric | 2.6.1 | GATConv, global pooling, and DataLoader operational |
| PyTables | 3.9.2 | Compiled with Homebrew HDF5 2.2.0 and c-blosc 1.21.6 |
| NumPy | 1.26.4 | ABI-compatible with PyTables C-extensions |
| Scikit-Learn | 1.6.1 | Metrics calculation (`roc_auc_score`, `roc_curve`) |
| Pandas | 2.3.3 | Fixed-format HDF5 parsing |

---

## 3. Dataset Audit & Streaming Signal Filter

### Inspection Findings
- `data/events_LHCO2020_backgroundMC_Pythia.h5`:
  - 1,000,000 events, 2,100 columns (700 particles x 3 features: pT, eta, phi).
  - 100% background events.
- `data/W_prime_signal.h5`:
  - 1,100,000 events, 2,101 columns.
  - Conclusively identified as the Zenodo mixed R&D dataset (`events_anomalydetection.h5`).
  - Column 2100 contains truth labels: 1,000,000 background events (`truth == 0`) and 100,000 genuine signal events (`truth == 1`).

### Solution
Implemented `stream_filtered_events()` in `src/run_lhco_gnn.py`:
- Automatically detects presence of column 2100.
- When `is_signal=True`, filters chunks on-the-fly for `truth == 1.0`.
- Streaming speed: 1,000 genuine signal events retrieved in 0.16 seconds without duplicating 2.6 GB on disk.

---

## 4. Architecture Implementation & Model Factory

Implemented in `src/run_lhco_gnn.py`:
- `BaselineMLP`:
  - Encoder: `Linear(64, 32)` -> `ELU` -> `Linear(32, 8)`
  - Decoder: `Linear(8, 32)` -> `ELU` -> `Linear(32, 64)`
  - Exact match with `models/autoencoder_bg.pt` (4,744 parameters).
- `event_to_flat_features()`: Truncates event to first 64 features and applies log1p scaling to pT columns (0, 3, 6, ...).
- `create_model(config)`: Factory supporting:
  - `baseline_mlp`: `BaselineMLP` (4,744 params)
  - `gat_standalone`: `NodeGNN` (6,101 params)
  - `gat_met_production`: `NodeGNN` (6,101 params)
  - `gat_met_decorrelation`: `AdversarialNodeGNN` (7,702 params)

### Strict Checkpoint Compatibility
Verified with `strict=True`:
- `models/autoencoder_bg.pt` strictly loaded into `BaselineMLP`
- `models/discovery_model.pt` strictly loaded into `NodeGNN`
- `models/unlearning_model.pt` strictly loaded into `AdversarialNodeGNN`

---

## 5. Unified Training Engine (`src/train_experiment.py`)

- Reads all configuration parameters from JSON.
- Respects mathematical loss formulation:
  $$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{MET}} \mathcal{L}_{\text{MET}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$$
- Sets seeds across Python, NumPy, and PyTorch.
- Outputs strictly to `experiments/<family>/<experiment_name>/`.
- Never overwrites `models/*.pt`.
- Supports `--dry-run` and `--smoke-test`.

---

## 6. Verification & Test Results

1. **Unit Test Suite:**
   15 passing unit tests (`tests/test_controlled_pipeline.py` and `tests/test_experiment_framework.py`).
2. **Dry-Run Validation:**
   All 4 architecture configs passed `--dry-run` with exit code 0.
3. **Data Streaming Validation:**
   Loaded real background and genuine signal chunks verifying node/edge counts and truth labels.
4. **Mini-Batch Smoke Test (MPS Device):**
   - `baseline_mlp`: 1 epoch, 20 events, Loss: 2.92668
   - `gat_standalone`: 1 epoch, 20 events, Loss: 3.41817
   - `gat_met_production`: 1 epoch, 20 events, Loss: 396.93835 (Recon: 3.52430, MET: 3934.14)
   - `gat_met_decorrelation`: 1 epoch, 20 events, Loss: 396.71045 (Recon: 3.11619, MET: 3926.61, Adv: 18.67)
5. **Evaluation Verification:**
   `evaluation/evaluate_single.py` successfully scored 500 background and 500 signal events, producing `metrics.json` and `predictions.npz`.

---

## 7. Status for Phase 3 Execution

The pipeline is verified and ready to execute the Phase 3 architecture ablation study.
