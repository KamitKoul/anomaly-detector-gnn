# Google Colab GPU Workflow & Phase 3 Architecture Ablation Guide

This guide details how to execute the **Phase 3 Controlled Architecture Ablation Experiments** on **Google Colab using NVIDIA CUDA GPUs**, preserving all scientific invariants from Phase 0 and Phase 1.

---

## 1. Why Google Colab with NVIDIA CUDA?

### The Mac Apple Silicon (MPS) Failure Analysis
During local testing on Apple Silicon (`mps`), the baseline MLP autoencoder (`baseline_mlp`) trained without issues because it processes fixed flat tensors (`shape: [32, 64]`). However, all three Graph Attention Network (`NodeGNN` and `AdversarialNodeGNN`) training runs terminated due to memory exhaustion at approximately 20 GB.

**Root-Cause Diagnosis:**
1. **Dynamic Graph Neighborhoods:** Each LHCO collision event produces a variable number of particles ($N \in [150, 300]$) and dense angular clusters ($\Delta R < 0.4$), yielding ~5,200–6,200 nodes and **~160,000–180,000 edges per batch** of 32 graphs.
2. **MPS Metal Heap Driver Fragmentation:** PyTorch's Metal Performance Shaders (MPS) allocator creates native Metal heap pages for tensor buffers. Because every batch has a slightly different edge count and attention tensor shape, the MPS allocator does not reuse previously cached heap allocations. Instead, `driver_allocated_memory` steadily grew by **300–400 MB per batch** (reaching 2.1 GB in just 4 batches), eventually exhausting system unified memory across the 157 batches of an epoch.
3. **The CUDA Advantage:** NVIDIA CUDA uses a Best-Fit with Coalescing (BFC) caching allocator with dedicated high-bandwidth VRAM (16 GB on Tesla T4; 24 GB on L4; 40/80 GB on A100) and native CUDA kernels for PyTorch Geometric message passing, handling these dynamic graph sizes reliably without driver leakage.

---

## 2. Intended Development & Training Workflow

```
┌─────────────────────────────────┐
│     Local Mac / VS Code         │
│  - Edit code, tests, configs    │
│  - Run dry-runs & unit tests    │
│  - git commit & git push        │
└───────────────┬─────────────────┘
                │ git push / pull
                ▼
┌─────────────────────────────────┐
│        Google Colab GPU         │
│  - Mount Google Drive (datasets)│
│  - NVIDIA CUDA GPU acceleration │
│  - Train GAT architecture runs  │
│  - Evaluate single runs         │
│  - Aggregate comparison table   │
│  - Save results to Drive / Git  │
└───────────────┬─────────────────┘
                │ Drive sync / git pull
                ▼
┌─────────────────────────────────┐
│     Local Mac / VS Code         │
│  - Inspect metrics.json         │
│  - Review comparisons/ summary  │
│  - Scientific interpretation    │
└─────────────────────────────────┘
```

> **Note on VS Code Remote:** While VS Code supports connecting to remote Jupyter kernels, it is not required. The clean, recommended method is running [`notebooks/phase3_colab.ipynb`](file:///Users/jhum/Documents/Padhai%20stuff/college/mca/coding/anomaly-detector-gnn/notebooks/phase3_colab.ipynb) directly in Google Colab's web interface and syncing experiment results via Google Drive or Git.
> **Note on College Cluster:** If later transferring training to a college HPC cluster, do not run interactive SSH training; submit via SLURM job scripts (`sbatch` or `srun`).

---

## 3. Dataset Handling in Google Colab

The LHCO2020 datasets are large (~2.5 GB background, ~2.6 GB mixed Zenodo signal). To avoid downloading them on every Colab session or duplicating 5+ GB of files, use Google Drive:

### Setup in Google Drive:
1. In your personal Google Drive, create a folder named `lhco_data`:
   ```
   My Drive/
   └── lhco_data/
       ├── events_LHCO2020_backgroundMC_Pythia.h5
       └── W_prime_signal.h5
   ```
2. When launching Colab, mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Create symbolic links in the repository's `data/` directory:
   ```bash
   mkdir -p data
   ln -sf "/content/drive/MyDrive/lhco_data/events_LHCO2020_backgroundMC_Pythia.h5" data/events_LHCO2020_backgroundMC_Pythia.h5
   ln -sf "/content/drive/MyDrive/lhco_data/W_prime_signal.h5" data/W_prime_signal.h5
   ```
4. **Signal Filtering:** The repository's data loader automatically filters `data/W_prime_signal.h5` on-the-fly (`truth == 1.0` in column 2100) via `stream_filtered_events()`, ensuring pure signal events are evaluated without needing pre-extracted files.

---

## 4. Execution Step-by-Step

### Option A: Using the Jupyter Notebook (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Select **Upload** and upload [`notebooks/phase3_colab.ipynb`](file:///Users/jhum/Documents/Padhai%20stuff/college/mca/coding/anomaly-detector-gnn/notebooks/phase3_colab.ipynb) (or open via GitHub).
3. Ensure the runtime has GPU enabled: **Runtime &rarr; Change runtime type &rarr; T4 GPU**.
4. Run cells sequentially:
   - Cell 1: Clone / pull repository.
   - Cell 2: Mount Google Drive and symlink datasets.
   - Cell 3: Run `!bash scripts/setup_colab.sh` (installs dependencies, checks CUDA, runs dry-runs).
   - Cell 4: Run isolated GAT smoke test (20 events, 1 epoch).
   - Cell 5: Run the 4 controlled experiments.
   - Cell 6: Run single-experiment evaluations.
   - Cell 7: Run comparative summary aggregation.
   - Cell 8: Backup results to Google Drive.

### Option B: Using the Colab Shell Terminal / Bash Script
```bash
# 1. Clone repository
git clone https://github.com/KamitKoul/anomaly-detector-gnn.git
cd anomaly-detector-gnn

# 2. Mount drive and symlink data (in Python cell or terminal)
mkdir -p data
ln -sf "/content/drive/MyDrive/lhco_data/events_LHCO2020_backgroundMC_Pythia.h5" data/
ln -sf "/content/drive/MyDrive/lhco_data/W_prime_signal.h5" data/

# 3. Automated setup and dry-runs
bash scripts/setup_colab.sh

# 4. Train the 4 controlled architecture experiments
python src/train_experiment.py --config configs/architecture/baseline_mlp.json
python src/train_experiment.py --config configs/architecture/gat_standalone.json
python src/train_experiment.py --config configs/architecture/gat_met_production.json
python src/train_experiment.py --config configs/architecture/gat_met_decorrelation.json

# 5. Evaluate all experiments
python evaluation/evaluate_single.py --experiment experiments/architecture/baseline_mlp
python evaluation/evaluate_single.py --experiment experiments/architecture/gat_standalone
python evaluation/evaluate_single.py --experiment experiments/architecture/gat_met_production
python evaluation/evaluate_single.py --experiment experiments/architecture/gat_met_decorrelation

# 6. Generate comparative summary table
python evaluation/compare_experiments.py --family architecture
```

---

## 5. Scientific Experimental Invariants

The following experimental parameters are fixed across all models and must not be altered:
- **Seed:** `42` (applied to Python `random`, `numpy`, and `torch`)
- **Training Population:** Exactly 5,000 background events (`events_LHCO2020_backgroundMC_Pythia.h5`)
- **Evaluation Population:** Exactly 500 background events vs 500 genuine $W'$ signal events (`truth == 1.0`)
- **Input Scaling:** $p_T \to \ln(1 + p_T)$ log-compression
- **Graph Neighborhood:** $\Delta R < 0.4$ pairwise Euclidean distance in $(\eta, \phi)$ detector space
- **MET Target Vector:** True physical coordinates $(MET_x, MET_y)$ with azimuthal angle $\phi_{\text{MET}} = \text{atan2}(MET_y, MET_x)$
- **Loss Formulations:**
  - `baseline_mlp`: $\mathcal{L} = \mathcal{L}_{\text{recon}}$
  - `gat_standalone`: $\mathcal{L} = \mathcal{L}_{\text{recon}}$
  - `gat_met_production`: $\mathcal{L} = \mathcal{L}_{\text{recon}} + 0.1 \times \mathcal{L}_{\text{MET}}$
  - `gat_met_decorrelation`: $\mathcal{L} = \mathcal{L}_{\text{recon}} + 0.1 \times \mathcal{L}_{\text{MET}} + 0.05 \times \mathcal{L}_{\text{adv}}$ (GRL $\alpha = 1.0$)
- **Optimizer:** Adam with learning rate $\eta = 0.001$, batch size 32
- **Epochs:** 20 for standard models; 15 for adversarial model

---

## 6. GPU Memory Diagnostics

During training, `src/train_experiment.py` prints real-time memory telemetry:
- **Startup:** Device name, initial allocated memory, and reserved memory.
- **Per Epoch:** Active allocated CUDA memory and peak memory allocated.
- **Artifacts:** `training_log.json` logs `device_diagnostics` alongside per-epoch loss history for full scientific provenance.
