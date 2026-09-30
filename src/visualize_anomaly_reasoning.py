"""
Visualize Anomaly Reasoning — Decision Threshold Decomposition

Analyzes how the anomaly detection threshold separates background from
signal events. Loads the trained NodeGNN model and computes per-event
reconstruction MSE scores, then produces a 2x2 diagnostic figure:
  1. Score distribution histograms (overlapping) with threshold line
  2. ROC curve with AUC
  3. Precision-Recall curve
  4. Score vs event index scatter plot showing threshold boundary

Output: anomaly_reasoning_saliency.png
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve

sys.path.append(os.getcwd())

from src.run_lhco_gnn import stream_lhco_events, event_to_graph, NodeGNN, DEVICE


########################################
# CONFIG
########################################

MODEL_PATH = "models/discovery_model.pt"
BG_DATA_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"
SIG_DATA_PATH = "data/W_prime_signal.h5"

EVAL_LIMIT = 500          # Number of events to score per class
OUTPUT_PATH = "anomaly_reasoning_saliency.png"


########################################
# LOAD TRAINED MODEL
########################################

def load_model(model_path, device):
    """
    Loads the trained NodeGNN state dict from disk.

    Args:
        model_path: Path to the saved .pt state dict.
        device: torch device to map the model onto.

    Returns:
        NodeGNN model in eval mode on the target device.
    """
    model = NodeGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path} → {device}")
    return model


########################################
# COMPUTE ANOMALY SCORES
########################################

def compute_anomaly_scores(model, data_path, limit, device, label=""):
    """
    Streams events from an HDF5 file, converts each to a graph, and
    computes the per-event reconstruction MSE as the anomaly score.

    Args:
        model: Trained NodeGNN in eval mode.
        data_path: Path to the LHCO HDF5 file.
        limit: Maximum number of events to process.
        device: torch device.
        label: Description for the progress bar.

    Returns:
        List of per-event anomaly scores (float).
    """
    scores = []
    count = 0

    with torch.no_grad():
        for df_chunk in stream_lhco_events(data_path):
            graphs = []
            for i in range(len(df_chunk)):
                if count >= limit:
                    break
                graphs.append(event_to_graph(df_chunk.iloc[i].values))
                count += 1

            if len(graphs) == 0:
                break

            loader = DataLoader(graphs, batch_size=1, shuffle=False)

            for data in tqdm(loader, desc=f"Scoring {label}", leave=False):
                data = data.to(device)
                recon_x, _ = model(data.x, data.edge_index, data.batch)
                score = F.mse_loss(recon_x, data.x).item()
                scores.append(score)

            if count >= limit:
                break

    return scores


########################################
# CHOOSE OPTIMAL THRESHOLD (Youden's J)
########################################

def select_threshold(bg_scores, sig_scores):
    """
    Computes the optimal anomaly score threshold using Youden's J statistic
    (maximises TPR - FPR) on the combined background / signal scores.

    Args:
        bg_scores: List of background anomaly scores.
        sig_scores: List of signal anomaly scores.

    Returns:
        Optimal threshold (float), fpr array, tpr array, roc_auc (float).
    """
    scores = np.concatenate([bg_scores, sig_scores])
    labels = np.concatenate([np.zeros(len(bg_scores)),
                             np.ones(len(sig_scores))])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Youden's J — best separation point
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]

    return optimal_threshold, fpr, tpr, roc_auc


########################################
# VISUALIZATION — 2×2 DIAGNOSTIC PANEL
########################################

def plot_anomaly_reasoning(bg_scores, sig_scores, threshold,
                           fpr, tpr, roc_auc, output_path):
    """
    Produces a 2×2 figure with four anomaly-reasoning panels.

    Panel layout:
      (0,0) Score distribution histograms with threshold line
      (0,1) ROC curve with AUC
      (1,0) Precision-Recall curve
      (1,1) Score vs event index scatter with threshold boundary

    Args:
        bg_scores: List of background anomaly scores.
        sig_scores: List of signal anomaly scores.
        threshold: Optimal anomaly score threshold.
        fpr: False positive rate array (from roc_curve).
        tpr: True positive rate array (from roc_curve).
        roc_auc: Area under the ROC curve.
        output_path: File path for the saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Anomaly Decision Threshold — Reasoning Decomposition",
                 fontsize=15, fontweight="bold", y=0.98)

    # ---- combined arrays for sklearn metrics ----
    all_scores = np.concatenate([bg_scores, sig_scores])
    all_labels = np.concatenate([np.zeros(len(bg_scores)),
                                 np.ones(len(sig_scores))])

    # ================================================
    # Panel 1 — Score Distribution Histograms
    # ================================================
    ax = axes[0, 0]
    bins = np.linspace(min(all_scores), np.percentile(all_scores, 99), 60)
    ax.hist(bg_scores, bins=bins, color="cyan", alpha=0.55,
            label="Background", density=True, edgecolor="black", linewidth=0.3)
    ax.hist(sig_scores, bins=bins, color="magenta", alpha=0.55,
            label="W' Signal", density=True, edgecolor="black", linewidth=0.3)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.8,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score (Reconstruction MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions")
    ax.legend(fontsize=9)

    # ================================================
    # Panel 2 — ROC Curve
    # ================================================
    ax = axes[0, 1]
    ax.plot(fpr, tpr, color="darkorange", linewidth=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color="darkorange")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # ================================================
    # Panel 3 — Precision-Recall Curve
    # ================================================
    ax = axes[1, 0]
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, color="teal", linewidth=2,
            label=f"PR (AUC = {pr_auc:.4f})")
    ax.fill_between(recall, precision, alpha=0.12, color="teal")
    ax.set_xlabel("Recall (Signal Efficiency)")
    ax.set_ylabel("Precision (Signal Purity)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # ================================================
    # Panel 4 — Score vs Event Index Scatter
    # ================================================
    ax = axes[1, 1]
    n_bg = len(bg_scores)
    n_sig = len(sig_scores)
    bg_idx = np.arange(n_bg)
    sig_idx = np.arange(n_bg, n_bg + n_sig)

    ax.scatter(bg_idx, bg_scores, c="cyan", s=6, alpha=0.6,
               label="Background", edgecolors="none")
    ax.scatter(sig_idx, sig_scores, c="magenta", s=6, alpha=0.6,
               label="W' Signal", edgecolors="none")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.8,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Anomaly Score (MSE)")
    ax.set_title("Per-Event Score Landscape")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Figure saved → {output_path}")


########################################
# MAIN
########################################

if __name__ == "__main__":

    device = DEVICE
    print(f"Using device: {device}")

    # --- Load trained model ---
    model = load_model(MODEL_PATH, device)

    # --- Compute anomaly scores for both classes ---
    print("\n[1/2] Scoring background events …")
    bg_scores = compute_anomaly_scores(
        model, BG_DATA_PATH, EVAL_LIMIT, device, label="Background"
    )

    print(f"\n[2/2] Scoring signal events …")
    sig_scores = compute_anomaly_scores(
        model, SIG_DATA_PATH, EVAL_LIMIT, device, label="Signal"
    )

    # --- Summary statistics ---
    print(f"\nBackground  — mean: {np.mean(bg_scores):.6f}  "
          f"std: {np.std(bg_scores):.6f}  n={len(bg_scores)}")
    print(f"Signal      — mean: {np.mean(sig_scores):.6f}  "
          f"std: {np.std(sig_scores):.6f}  n={len(sig_scores)}")

    # --- Determine optimal threshold ---
    threshold, fpr, tpr, roc_auc = select_threshold(bg_scores, sig_scores)
    print(f"\nOptimal threshold (Youden's J): {threshold:.6f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # --- Produce 2×2 diagnostic figure ---
    plot_anomaly_reasoning(
        bg_scores, sig_scores, threshold,
        fpr, tpr, roc_auc, OUTPUT_PATH
    )

    print("\nAnomaly reasoning visualization complete.")
