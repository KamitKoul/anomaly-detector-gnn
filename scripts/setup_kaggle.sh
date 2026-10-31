#!/usr/bin/env bash
# ==============================================================================
# setup_kaggle.sh — Kaggle GPU Environment Setup & Verification
# ==============================================================================
# Automates environment verification, dependency installation, dataset checks,
# framework validation dry-runs, and a CUDA GAT smoke test on Kaggle Notebooks.
#
# Usage in Kaggle Notebook:
#   !bash scripts/setup_kaggle.sh
# ==============================================================================

set -eo pipefail

# Ensure repository root is on PYTHONPATH for all python invocations
export PYTHONPATH=.

echo "========================================================================"
echo "  HEP ANOMALY DETECTOR GNN — KAGGLE GPU SETUP & VERIFICATION"
echo "========================================================================"

# Detect Python interpreter
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY_CMD="$VIRTUAL_ENV/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY_CMD=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    PY_CMD="python"
fi

# 1. System & Python Verification
echo -e "\n[1/6] Verifying System and Python runtime..."
$PY_CMD --version
which $PY_CMD 2>/dev/null || echo "$PY_CMD"
pwd

# 2. NVIDIA GPU & CUDA Verification
echo -e "\n[2/6] Verifying NVIDIA GPU & CUDA acceleration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not detected. Checking if CUDA is available via PyTorch..."
fi

$PY_CMD -c "
try:
    import torch
    print('PyTorch Version:    ', torch.__version__)
    print('CUDA Available:     ', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Device Count:       ', torch.cuda.device_count())
        print('Current Device:     ', torch.cuda.current_device())
        print('GPU Model Name:     ', torch.cuda.get_device_name(0))
        print('CUDA Capability:    ', torch.cuda.get_device_capability(0))
    else:
        print('[WARNING] CUDA is NOT available in PyTorch.')
        print('          In Kaggle: In the right panel, select Settings/Notebook options -> Accelerator -> GPU T4 x2 or GPU P100.')
except ImportError:
    print('[WARNING] PyTorch not yet installed in this environment.')
"

# 3. Scientific & Graph Neural Network Dependencies
echo -e "\n[3/6] Installing & Verifying scientific dependencies..."
$PY_CMD -m pip install --quiet torch-geometric tables

$PY_CMD -c "
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

# 4. Dataset Availability Check & Auto-Discovery
echo -e "\n[4/6] Checking LHCO2020 scientific datasets..."
mkdir -p data

BG_PATH="data/events_LHCO2020_backgroundMC_Pythia.h5"
SIG_PATH="data/W_prime_signal.h5"

# Auto-discover from /kaggle/input if attached as a Kaggle dataset
$PY_CMD -c "
import os, glob

bg_path = '$BG_PATH'
sig_path = '$SIG_PATH'

input_dir = '/kaggle/input'
if os.path.exists(input_dir):
    # Search for background dataset
    if not os.path.exists(bg_path):
        candidates = glob.glob(os.path.join(input_dir, '**/*background*Pythia*.h5'), recursive=True) + \
                     glob.glob(os.path.join(input_dir, '**/*background*.h5'), recursive=True)
        if candidates:
            print(f'  [DISCOVERED] Linking Background dataset from {candidates[0]} -> {bg_path}')
            os.symlink(candidates[0], bg_path)

    # Search for signal dataset
    if not os.path.exists(sig_path):
        candidates = glob.glob(os.path.join(input_dir, '**/*W_prime*.h5'), recursive=True) + \
                     glob.glob(os.path.join(input_dir, '**/*anomalydetection*.h5'), recursive=True)
        if candidates:
            print(f'  [DISCOVERED] Linking Signal dataset from {candidates[0]} -> {sig_path}')
            os.symlink(candidates[0], sig_path)
"

# Verify dataset status
$PY_CMD -c "
import os, sys

paths = [
    ('Background (Pythia MC)', '$BG_PATH'),
    ('Signal (W prime / R&D)', '$SIG_PATH'),
]

all_valid = True
for name, p in paths:
    if os.path.exists(p):
        try:
            size_bytes = os.path.getsize(p)
            size_gb = size_bytes / (1024 ** 3)
            if size_bytes < 100_000_000:
                print(f'  [WARNING] {name}: {p} exists but is incomplete ({size_bytes} bytes < 100 MB)!')
                all_valid = False
            else:
                print(f'  [FOUND]   {name}: {p} ({size_gb:.2f} GB)')
        except Exception as e:
            print(f'  [ERROR]   {name}: {p} could not be read ({e})')
            all_valid = False
    else:
        print(f'  [MISSING] {name}: {p} not found.')
        all_valid = False

if not all_valid:
    print('  [ACTION] Run Cell 3 in the Kaggle notebook to download datasets or attach a Kaggle dataset.')
"

# 5. Unit Tests & Config Dry-Runs
echo -e "\n[5/6] Executing unit tests and architecture dry-runs..."
$PY_CMD -m unittest discover -s tests -p "test_*.py"

echo -e "\nRunning dry-runs across all four architecture configs:"
for cfg in baseline_mlp gat_standalone gat_met_production gat_met_decorrelation; do
    echo -n "  Testing $cfg... "
    $PY_CMD src/train_experiment.py --config "configs/architecture/${cfg}.json" --dry-run > /dev/null
    echo "OK (verified)"
done

# 6. Isolated GAT CUDA Smoke Test (Tiny subset: 20 events, 1 epoch)
if [ -f "$BG_PATH" ] && [ $(wc -c < "$BG_PATH") -gt 100000000 ]; then
    echo -e "\n[6/6] Executing isolated GAT CUDA training smoke test..."
    $PY_CMD src/train_experiment.py \
        --config configs/architecture/gat_met_production.json \
        --limit 20 \
        --epochs 1 \
        --smoke-test \
        --output-dir experiments/smoke_test/kaggle_gat_smoke

    rm -rf experiments/smoke_test/kaggle_gat_smoke
    echo "  [SUCCESS] GAT CUDA forward, backward, loss, and optimizer step verified!"
else
    echo -e "\n[6/6] Skipping GAT smoke test until datasets are available in data/."
fi

echo -e "\n========================================================================"
echo "  KAGGLE SETUP COMPLETED SUCCESSFULLY"
echo "  To diagnose GPU utilization & throughput benchmarks, run:"
echo "    python scripts/diagnose_gpu_pipeline.py"
echo "  Ready to execute Phase 3 architecture ablation experiments on CUDA GPU."
echo "========================================================================"
