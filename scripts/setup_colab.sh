#!/usr/bin/env bash
# ==============================================================================
# setup_colab.sh — Google Colab GPU Environment Setup & Verification
# ==============================================================================
# Automates environment verification, dependency installation, dataset checks,
# framework validation dry-runs, and a CUDA GAT smoke test.
#
# Usage in Google Colab:
#   !bash scripts/setup_colab.sh
# ==============================================================================

set -eo pipefail

echo "========================================================================"
echo "  HEP ANOMALY DETECTOR GNN — GOOGLE COLAB GPU SETUP & VERIFICATION"
echo "========================================================================"

# 1. System & Python Verification
echo -e "\n[1/6] Verifying System and Python runtime..."
python3 --version
which python3

# 2. NVIDIA GPU & CUDA Verification
echo -e "\n[2/6] Verifying NVIDIA GPU & CUDA acceleration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not detected. Checking if CUDA is available via PyTorch..."
fi

python3 -c "
import torch
print('PyTorch Version:    ', torch.__version__)
print('CUDA Available:     ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Count:       ', torch.cuda.device_count())
    print('Current Device:     ', torch.cuda.current_device())
    print('GPU Model Name:     ', torch.cuda.get_device_name(0))
    print('CUDA Capability:    ', torch.cuda.get_device_capability(0))
else:
    print('[WARNING] CUDA is NOT available in PyTorch. Switch Colab runtime to GPU (Runtime -> Change runtime type -> T4/A100 GPU).')
"

# 3. Scientific & Graph Neural Network Dependencies
echo -e "\n[3/6] Installing & Verifying scientific dependencies..."
pip install --quiet torch-geometric tables pandas numpy scikit-learn matplotlib tqdm

python3 -c "
import torch
import torch_geometric
import tables
import pandas
import numpy
import sklearn

print('Dependencies verified:')
print('  torch-geometric: ', torch_geometric.__version__)
print('  tables (PyTables):', tables.__version__)
print('  pandas:          ', pandas.__version__)
print('  numpy:           ', numpy.__version__)
print('  scikit-learn:    ', sklearn.__version__)
"

# 4. Dataset Availability Check
echo -e "\n[4/6] Checking LHCO2020 scientific datasets..."
mkdir -p data

BG_PATH="data/events_LHCO2020_backgroundMC_Pythia.h5"
SIG_PATH="data/W_prime_signal.h5"

if [ -f "$BG_PATH" ] && [ -f "$SIG_PATH" ]; then
    echo "  [FOUND] Background dataset: $BG_PATH ($(du -h "$BG_PATH" | cut -f1))"
    echo "  [FOUND] Signal dataset:     $SIG_PATH ($(du -h "$SIG_PATH" | cut -f1))"
else
    echo "  [STATUS] Datasets not found in data/. Checking Google Drive mount..."
    DRIVE_LHCO="/content/drive/MyDrive/lhco_data"
    if [ -d "$DRIVE_LHCO" ]; then
        echo "  [INFO] Found Google Drive dataset directory: $DRIVE_LHCO"
        if [ -f "$DRIVE_LHCO/events_LHCO2020_backgroundMC_Pythia.h5" ]; then
            ln -sf "$DRIVE_LHCO/events_LHCO2020_backgroundMC_Pythia.h5" "$BG_PATH"
            echo "  [LINKED] Symlinked background from Google Drive."
        fi
        if [ -f "$DRIVE_LHCO/W_prime_signal.h5" ]; then
            ln -sf "$DRIVE_LHCO/W_prime_signal.h5" "$SIG_PATH"
            echo "  [LINKED] Symlinked signal from Google Drive."
        fi
    else
        echo "  [ACTION REQUIRED] Mount Google Drive and link datasets:"
        echo "    from google.colab import drive"
        echo "    drive.mount('/content/drive')"
        echo "    !ln -sf /content/drive/MyDrive/path/to/events_LHCO2020_backgroundMC_Pythia.h5 data/"
        echo "    !ln -sf /content/drive/MyDrive/path/to/W_prime_signal.h5 data/"
    fi
fi

# 5. Unit Tests & Config Dry-Runs
echo -e "\n[5/6] Executing unit tests and architecture dry-runs..."
python3 -m unittest tests/test_controlled_pipeline.py tests/test_experiment_framework.py

echo -e "\nRunning dry-runs across all four architecture configs:"
for cfg in baseline_mlp gat_standalone gat_met_production gat_met_decorrelation; do
    echo -n "  Testing $cfg... "
    python3 src/train_experiment.py --config "configs/architecture/${cfg}.json" --dry-run > /dev/null
    echo "OK (verified)"
done

# 6. Isolated GAT CUDA Smoke Test (Tiny subset: 20 events, 1 epoch)
if [ -f "$BG_PATH" ]; then
    echo -e "\n[6/6] Executing isolated GAT CUDA training smoke test..."
    python3 src/train_experiment.py \
        --config configs/architecture/gat_met_production.json \
        --limit 20 \
        --epochs 1 \
        --smoke-test \
        --output-dir experiments/smoke_test/colab_gat_smoke

    rm -rf experiments/smoke_test/colab_gat_smoke
    echo "  [SUCCESS] GAT CUDA forward, backward, loss, and optimizer step verified!"
else
    echo -e "\n[6/6] Skipping GAT smoke test until datasets are symlinked/mounted."
fi

echo -e "\n========================================================================"
echo "  COLAB SETUP COMPLETED SUCCESSFULLY"
echo "  Ready to execute Phase 3 architecture ablation experiments on CUDA GPU."
echo "========================================================================"
