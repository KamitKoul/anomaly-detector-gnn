# Kaggle GPU Workflow & Phase 3 Architecture Ablation Guide

This guide details how to execute the **Phase 3 Controlled Architecture Ablation Experiments** on **Kaggle Notebooks using NVIDIA CUDA GPUs (T4 / P100)** with 30 free GPU hours per week, preserving all scientific invariants.

---

## 1. Why Kaggle Notebooks for HEP GNNs?

### Overcoming Colab Usage Exhaustion & Local MPS Memory Fragmentation
- **Free 30 GPU Hours / Week:** Kaggle provides 30 hours of free GPU compute every week (resets weekly) with zero credit card or subscription requirement.
- **Dedicated 16 GB VRAM:** NVIDIA Tesla T4 or P100 with dedicated high-bandwidth memory solves Apple Silicon MPS heap driver fragmentation, seamlessly accommodating dynamic graph sizes (~160,000–180,000 edges per batch).
- **30 GB System RAM:** Ample headroom for chunked streaming of the 2.5 GB LHCO collision events.
- **High-Speed Datacenter Network:** Downloads the full Zenodo datasets (~5 GB total) in under 2 minutes at 50–100 MB/s.

---

## 2. Pre-Flight Checklist (Kaggle Account Requirements)

Before launching your notebook, make sure your Kaggle account is configured:

1. **Phone Verification (One-Time Setup):**
   - Kaggle requires phone number verification via SMS to unlock free GPU accelerators and outbound Internet access.
   - If not already verified: Go to [kaggle.com/settings](https://www.kaggle.com/settings) &rarr; **Phone Verification** &rarr; Verify your number.
2. **Kaggle Notebook Settings (Right Panel):**
   When you create a notebook, you **must** configure these two toggles in the right sidebar under **Notebook options**:
   - **Accelerator:** Set to **`GPU T4 x2`** (or **`GPU P100`**).
   - **Internet:** Set to **`Internet On`** (Mandatory! If left *Off*, `git clone`, `pip install`, and dataset downloads will fail with network errors).

---

## 3. Step-by-Step Execution Guide

### Option A: Import the Dedicated Kaggle Notebook (Recommended)

1. Open [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the notebook top menu, click **File** &rarr; **Import Notebook**.
3. Select the **GitHub** tab (or upload [`notebooks/phase3_kaggle.ipynb`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/notebooks/phase3_kaggle.ipynb)):
   - Enter repo URL: `https://github.com/KamitKoul/anomaly-detector-gnn`
   - Select notebook: `notebooks/phase3_kaggle.ipynb`
4. In the right panel under **Notebook options**:
   - Select **Accelerator &rarr; GPU T4 x2**
   - Toggle **Internet &rarr; On**
5. Click **Run All** (or execute cells sequentially).

---

### Option B: From Scratch in a Fresh Kaggle Notebook

If starting with an empty notebook at [kaggle.com/code](https://www.kaggle.com/code):

#### Cell 1: Clone Repository
```python
import os
REPO_DIR = "/kaggle/working/anomaly-detector-gnn"
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/KamitKoul/anomaly-detector-gnn.git {REPO_DIR}

%cd {REPO_DIR}
!git fetch origin main
!git reset --hard origin/main
!git log -n 3 --oneline
```

#### Cell 2: Dataset Acquisition (Fast Zenodo Download or Kaggle Input)
```python
import os, glob

!mkdir -p data

bg_target = "data/events_LHCO2020_backgroundMC_Pythia.h5"
sig_target = "data/W_prime_signal.h5"

def is_valid_dataset(path):
    return os.path.exists(path) and os.path.getsize(path) > 100_000_000

# 1. Auto-discover from /kaggle/input if attached as a Kaggle dataset
input_dir = "/kaggle/input"
if os.path.exists(input_dir):
    if not is_valid_dataset(bg_target):
        bg_c = glob.glob(os.path.join(input_dir, "**/*background*.h5"), recursive=True)
        if bg_c and is_valid_dataset(bg_c[0]):
            !ln -sf "{bg_c[0]}" {bg_target}
    if not is_valid_dataset(sig_target):
        sig_c = glob.glob(os.path.join(input_dir, "**/*W_prime*.h5"), recursive=True) + \
                glob.glob(os.path.join(input_dir, "**/*anomalydetection*.h5"), recursive=True)
        if sig_c and is_valid_dataset(sig_c[0]):
            !ln -sf "{sig_c[0]}" {sig_target}

# 2. Direct download from Zenodo via high-speed links (~1-2 min)
if not is_valid_dataset(bg_target):
    print("Downloading Background MC from Zenodo (~2.5 GB)...")
    !wget -q --show-progress -c -O {bg_target} "https://zenodo.org/api/records/4536624/files/events_LHCO2020_backgroundMC_Pythia.h5/content"

if not is_valid_dataset(sig_target):
    print("Downloading Signal Dataset from Zenodo (~2.6 GB)...")
    !wget -q --show-progress -c -O {sig_target} "https://zenodo.org/api/records/4536377/files/events_anomalydetection.h5/content"

print("Datasets ready in data/:", os.listdir("data"))
```

#### Cell 3: Setup Dependencies & Run Dry-Runs
```python
!bash scripts/setup_kaggle.sh
```

#### Cell 4: Isolated CUDA GAT Smoke Test (20 events, 1 epoch)
```python
!python src/train_experiment.py \
    --config configs/architecture/gat_met_production.json \
    --limit 20 \
    --epochs 1 \
    --smoke-test \
    --output-dir experiments/smoke_test/kaggle_gat_smoke

!rm -rf experiments/smoke_test/kaggle_gat_smoke
```

#### Cell 5: Train Controlled Architecture Experiments
```python
# 1. Baseline flat MLP
!python src/train_experiment.py --config configs/architecture/baseline_mlp.json

# 2. Standalone GAT autoencoder
!python src/train_experiment.py --config configs/architecture/gat_standalone.json

# 3. Production Multi-Task GAT + MET objective
!python src/train_experiment.py --config configs/architecture/gat_met_production.json

# 4. Adversarial GAT + MET + Invariant Mass Decorrelation (GRL)
!python src/train_experiment.py --config configs/architecture/gat_met_decorrelation.json
```

#### Cell 6: Systematic Single-Model Evaluation
```python
import os

# Set EVAL_ALL = True for evaluation on ALL dataset events (~1M events)
EVAL_ALL = True
eval_flag = "--all" if EVAL_ALL else "--eval-limit 20000"

experiments = [
    "experiments/architecture/baseline_mlp",
    "experiments/architecture/gat_standalone",
    "experiments/architecture/gat_met_production",
    "experiments/architecture/gat_met_decorrelation",
]

for exp in experiments:
    print(f"\n---> Evaluating {exp}...")
    !python evaluation/evaluate_single.py --experiment "{exp}" {eval_flag}
```

#### Cell 7: Comparative Analysis Table
```python
!python evaluation/compare_experiments.py --family architecture

print("\n--- Comparative Results Summary ---")
!cat comparisons/architecture/summary.md
```

#### Cell 8: Background Mass-Sculpting & 1/JSD Evaluation
```python
!python evaluation/evaluate_mass_sculpting.py --family architecture --limit 2000
```

#### Cell 9: Package Results for One-Click Download
```python
!zip -q -r /kaggle/working/phase3_results.zip experiments/ comparisons/
!tar -czf /kaggle/working/phase3_results.tar.gz experiments/ comparisons/
print("Results packaged successfully in /kaggle/working/phase3_results.zip")
```

---

## 4. How to Download Checkpoints & Results Back to Your Mac

Once evaluation is finished:
1. In the Kaggle notebook, open the right-hand panel.
2. Scroll to the **Output** section (or click `/kaggle/working`).
3. You will see `phase3_results.zip` (and `phase3_results.tar.gz`).
4. Click the **three vertical dots (`...`)** next to the file and choose **Download**.
5. On your Mac, unzip the archive directly into your project root:
   ```bash
   unzip phase3_results.zip -d .
   ```
6. All trained checkpoints (`checkpoint.pt`), `training_log.json`, `metrics.json`, `predictions.npz`, and comparative summary tables will be restored in your local workspace!

---

## 5. Summary of Colab vs Kaggle Differences

| Feature | Google Colab | Kaggle Notebooks |
| :--- | :--- | :--- |
| **Free GPU Allowance** | Unpredictable compute units (exhausts quickly) | **30 GPU hours every week** (resets weekly) |
| **Available GPU** | T4 (15 GB) | **T4 x2 or P100 (16 GB)** |
| **System RAM** | 12 GB (standard) | **30 GB** |
| **Working Directory** | `/content` | **`/kaggle/working`** |
| **Outbound Internet** | Enabled by default | **Settings &rarr; Internet &rarr; On** (requires SMS verification) |
| **Dataset Mounting** | Google Drive (`/content/drive`) | Kaggle Input (`/kaggle/input`) or Zenodo `wget` |
| **Output Persistence** | Syncs to Google Drive | Saved in `/kaggle/working` (one-click download) |
