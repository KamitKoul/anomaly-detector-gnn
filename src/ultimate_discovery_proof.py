"""
Ultimate Discovery Proof — Scientific Validation Engine

Executes statistical validation separating Standard Model background from
simulated W' signal events. Performs a "Bump Hunt" by computing full
invariant mass via four-vector arithmetic and measuring mass distribution
density shifts in events flagged as anomalous by the trained GNN.

Pipeline:
    1. Load trained NodeGNN from models/discovery_model.pt
    2. Score background and signal events via reconstruction MSE
    3. Compute invariant mass for high-anomaly events
    4. Generate mass spectrum and exhibition proof plots
    5. Print statistical metrics (separation ratio, AUC, significance)
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.append(os.getcwd())

from src.run_lhco_gnn import (
    stream_lhco_events,
    event_to_graph,
    extract_particles,
    NodeGNN,
    DEVICE,
)


########################################
# CONFIG
########################################

BG_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"
SIG_PATH = "data/W_prime_signal.h5"
MODEL_PATH = "models/discovery_model.pt"

EVAL_LIMIT = 1000          # Events per dataset for evaluation
ANOMALY_PERCENTILE = 90    # Top-N% anomaly scores → "flagged" events
MASS_BINS = 80             # Histogram bin count
MASS_RANGE = (0, 8000)     # GeV range for invariant mass axis


########################################
# FOUR-VECTOR INVARIANT MASS
########################################

def compute_invariant_mass(particles):
    """
    Computes the full invariant mass of a particle system via
    four-vector arithmetic.

    For each particle with (pT, eta, phi):
        E  = pT * cosh(eta)
        px = pT * cos(phi)
        py = pT * sin(phi)
        pz = pT * sinh(eta)

    The invariant mass of the system is:
        M = sqrt( (ΣE)² - (Σpx)² - (Σpy)² - (Σpz)² )

    Args:
        particles: Nx3 numpy array with columns [pT, eta, phi].

    Returns:
        Invariant mass in GeV (float). Returns 0.0 for degenerate cases.
    """
    pt  = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]

    E  = pt * np.cosh(eta)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    total_E  = np.sum(E)
    total_px = np.sum(px)
    total_py = np.sum(py)
    total_pz = np.sum(pz)

    mass_sq = total_E**2 - total_px**2 - total_py**2 - total_pz**2

    if mass_sq < 0:
        return 0.0

    return np.sqrt(mass_sq)


########################################
# EVENT SCORING
########################################

def score_events(model, data_path, limit, device):
    """
    Streams events from an HDF5 file, converts to graphs, and computes
    per-event anomaly scores (reconstruction MSE) using the trained model.

    Also extracts raw particle arrays so invariant mass can be computed
    downstream without re-reading the file.

    Args:
        model:     Trained NodeGNN in eval mode.
        data_path: Path to the HDF5 event file.
        limit:     Maximum number of events to evaluate.
        device:    torch.device to run inference on.

    Returns:
        scores:    List[float]  — per-event reconstruction MSE.
        raw_events: List[np.ndarray] — raw particle arrays (Nx3).
    """
    scores = []
    raw_events = []
    count = 0

    for df_chunk in stream_lhco_events(data_path):
        for i in range(len(df_chunk)):
            if count >= limit:
                break

            event = df_chunk.iloc[i].values
            particles = extract_particles(event)
            graph = event_to_graph(event).to(device)

            # Single-event batch
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

            with torch.no_grad():
                recon_x, _ = model(graph.x, graph.edge_index, graph.batch)
                mse = F.mse_loss(recon_x, graph.x).item()

            scores.append(mse)
            raw_events.append(particles)
            count += 1

        if count >= limit:
            break

    return scores, raw_events


########################################
# MASS SPECTRUM FROM FLAGGED EVENTS
########################################

def compute_flagged_masses(scores, raw_events, percentile):
    """
    Selects events whose anomaly score exceeds the given percentile
    threshold and computes their invariant mass.

    Args:
        scores:     List of per-event anomaly scores.
        raw_events: List of particle arrays (Nx3).
        percentile: Only events above this percentile are kept.

    Returns:
        flagged_masses: np.ndarray of invariant masses for flagged events.
        threshold:      The anomaly score cut value used.
    """
    scores_arr = np.array(scores)
    threshold = np.percentile(scores_arr, percentile)

    flagged_masses = []
    for score, particles in zip(scores, raw_events):
        if score >= threshold:
            mass = compute_invariant_mass(particles)
            flagged_masses.append(mass)

    return np.array(flagged_masses), threshold


########################################
# STATISTICAL METRICS
########################################

def compute_statistics(bg_scores, sig_scores):
    """
    Computes key statistical metrics for signal-vs-background separation.

    Metrics:
        - Separation Ratio: mean(signal) / mean(background) scores
        - AUC: Area Under ROC Curve (sklearn)
        - Significance: (μ_sig - μ_bg) / σ_bg

    Args:
        bg_scores:  List of background anomaly scores.
        sig_scores: List of signal anomaly scores.

    Returns:
        Dictionary with keys 'ratio', 'auc', 'significance'.
    """
    bg_arr = np.array(bg_scores)
    sig_arr = np.array(sig_scores)

    # Separation ratio
    ratio = np.mean(sig_arr) / np.mean(bg_arr) if np.mean(bg_arr) > 0 else float("inf")

    # AUC
    labels = np.concatenate([np.zeros(len(bg_arr)), np.ones(len(sig_arr))])
    all_scores = np.concatenate([bg_arr, sig_arr])
    auc = roc_auc_score(labels, all_scores)

    # Gaussian significance estimate
    mu_bg = np.mean(bg_arr)
    sigma_bg = np.std(bg_arr) if np.std(bg_arr) > 0 else 1e-8
    mu_sig = np.mean(sig_arr)
    significance = (mu_sig - mu_bg) / sigma_bg

    return {"ratio": ratio, "auc": auc, "significance": significance}


########################################
# PLOT: MASS SPECTRUM (BUMP HUNT)
########################################

def plot_mass_spectrum(bg_masses, sig_masses, stats, save_path="discovery_mass_spectrum.png"):
    """
    Generates overlaid invariant-mass histograms for background and
    signal-flagged events — the classic "Bump Hunt" visualization.

    Args:
        bg_masses:  Invariant masses from flagged background events.
        sig_masses: Invariant masses from flagged signal events.
        stats:      Dict of statistical metrics for annotation.
        save_path:  Output PNG filename.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        bg_masses, bins=MASS_BINS, range=MASS_RANGE,
        color="grey", alpha=0.6, label="Background (flagged)", density=True,
    )
    ax.hist(
        sig_masses, bins=MASS_BINS, range=MASS_RANGE,
        color="magenta", alpha=0.7, label="W' Signal (flagged)", density=True,
    )

    ax.set_xlabel("Invariant Mass [GeV]", fontsize=13)
    ax.set_ylabel("Normalized Density", fontsize=13)
    ax.set_title("Bump Hunt — Invariant Mass Spectrum of Anomalous Events", fontsize=14)
    ax.legend(fontsize=12)

    # Annotate statistics
    text = (
        f"Separation Ratio: {stats['ratio']:.2f}x\n"
        f"AUC: {stats['auc']:.4f}\n"
        f"Significance: {stats['significance']:.2f}σ"
    )
    ax.text(
        0.97, 0.95, text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Mass spectrum saved to {save_path}")


########################################
# PLOT: EXHIBITION PROOF (SCORE DISTS)
########################################

def plot_exhibition_proof(bg_scores, sig_scores, stats, save_path="discovery_exhibition_proof.png"):
    """
    Side-by-side exhibition panel:
      Left  — anomaly score distributions (background vs signal)
      Right — cumulative score distributions showing separation

    Args:
        bg_scores:  Background anomaly scores.
        sig_scores: Signal anomaly scores.
        stats:      Dict of statistical metrics for annotation.
        save_path:  Output PNG filename.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Panel A: Score Distributions ----
    ax = axes[0]
    ax.hist(bg_scores, bins=60, color="cyan", alpha=0.5, label="Background", density=True)
    ax.hist(sig_scores, bins=60, color="magenta", alpha=0.5, label="W' Signal", density=True)
    ax.set_xlabel("Anomaly Score (Reconstruction MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Anomaly Score Distribution", fontsize=13)
    ax.legend(fontsize=11)

    # ---- Panel B: Cumulative Distributions ----
    ax = axes[1]
    bg_sorted = np.sort(bg_scores)
    sig_sorted = np.sort(sig_scores)
    ax.plot(
        bg_sorted, np.linspace(0, 1, len(bg_sorted)),
        color="cyan", linewidth=2, label="Background",
    )
    ax.plot(
        sig_sorted, np.linspace(0, 1, len(sig_sorted)),
        color="magenta", linewidth=2, label="W' Signal",
    )
    ax.set_xlabel("Anomaly Score (MSE)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title(f"CDF Separation — AUC: {stats['auc']:.4f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Discovery Exhibition Proof  |  Ratio: {stats['ratio']:.2f}x  |  "
        f"Significance: {stats['significance']:.2f}σ",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Exhibition proof saved to {save_path}")


########################################
# MAIN VALIDATION PIPELINE
########################################

def run_discovery_validation():
    """
    End-to-end scientific validation pipeline:
        1. Load trained NodeGNN weights
        2. Score background and signal events
        3. Compute statistical separation metrics
        4. Extract invariant mass for high-anomaly events
        5. Generate bump hunt and exhibition proof plots
    """
    device = DEVICE
    print(f"=== ULTIMATE DISCOVERY VALIDATION ===")
    print(f"Device: {device}")
    print(f"Background: {BG_PATH}")
    print(f"Signal:     {SIG_PATH}")
    print(f"Model:      {MODEL_PATH}")
    print()

    # ---- 1. Load Model ----
    print("Loading trained NodeGNN...")
    model = NodeGNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.\n")

    # ---- 2. Score Background Events ----
    print(f"Scoring {EVAL_LIMIT} background events...")
    bg_scores, bg_events = score_events(model, BG_PATH, EVAL_LIMIT, device)
    print(f"  Background scored: {len(bg_scores)} events\n")

    # ---- 3. Score Signal Events ----
    print(f"Scoring {EVAL_LIMIT} signal events...")
    sig_scores, sig_events = score_events(model, SIG_PATH, EVAL_LIMIT, device)
    print(f"  Signal scored: {len(sig_scores)} events\n")

    # ---- 4. Statistical Metrics ----
    stats = compute_statistics(bg_scores, sig_scores)
    print("=" * 50)
    print("  STATISTICAL RESULTS")
    print("=" * 50)
    print(f"  Separation Ratio : {stats['ratio']:.4f}x")
    print(f"  ROC AUC          : {stats['auc']:.4f}")
    print(f"  Significance     : {stats['significance']:.2f}σ")
    print(f"  Bg Mean Score    : {np.mean(bg_scores):.6f}")
    print(f"  Sig Mean Score   : {np.mean(sig_scores):.6f}")
    print("=" * 50)
    print()

    # ---- 5. Invariant Mass for Flagged Events ----
    print(f"Computing invariant mass for top-{100 - ANOMALY_PERCENTILE}% anomalous events...")

    bg_masses, bg_threshold = compute_flagged_masses(
        bg_scores, bg_events, ANOMALY_PERCENTILE
    )
    sig_masses, sig_threshold = compute_flagged_masses(
        sig_scores, sig_events, ANOMALY_PERCENTILE
    )

    print(f"  Background: {len(bg_masses)} flagged events (threshold: {bg_threshold:.6f})")
    print(f"  Signal:     {len(sig_masses)} flagged events (threshold: {sig_threshold:.6f})")
    print()

    # ---- 6. Generate Plots ----
    print("Generating visualizations...")
    plot_mass_spectrum(bg_masses, sig_masses, stats)
    plot_exhibition_proof(bg_scores, sig_scores, stats)

    print()
    print("=== VALIDATION COMPLETE ===")


########################################
# ENTRY POINT
########################################

if __name__ == "__main__":
    run_discovery_validation()
