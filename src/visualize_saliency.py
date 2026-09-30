"""
visualize_saliency.py
=====================
Backpropagation saliency maps for the GNN anomaly detector.

Highlights which particles in a high-anomaly event most strongly
triggered the anomaly detection by computing the L2 norm of
input-feature gradients obtained via a reverse gradient pass.

Outputs
-------
gnn_particle_saliency.png
    1×2 figure: (left) event in (η, φ) space colored by saliency,
    (right) bar chart of top-10 most salient particles with [pT, η, φ].
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.append(os.getcwd())

from src.run_lhco_gnn import (
    stream_lhco_events,
    event_to_graph,
    extract_particles,
    NodeGNN,
    DEVICE,
)

########################################
# Paths
########################################

SIGNAL_PATH = "data/W_prime_signal.h5"
MODEL_PATH = "models/discovery_model.pt"
OUTPUT_PATH = "gnn_particle_saliency.png"

########################################
# Load trained model
########################################


def load_model(model_path: str) -> NodeGNN:
    """Load a trained NodeGNN from a state-dict checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the saved ``state_dict`` file.

    Returns
    -------
    NodeGNN
        Model in eval mode on ``DEVICE``.
    """
    model = NodeGNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


########################################
# Select a high-anomaly event
########################################


def select_high_anomaly_event(model: NodeGNN, data_path: str, n_scan: int = 200):
    """Scan signal events and return the one with the highest anomaly score.

    The anomaly score is the per-node mean-squared reconstruction error
    averaged over the event.

    Parameters
    ----------
    model : NodeGNN
        Trained model (eval mode, no grad needed here).
    data_path : str
        Path to the signal HDF5 file.
    n_scan : int
        Maximum number of events to scan.

    Returns
    -------
    best_event : np.ndarray
        Raw event array for the highest-scoring event.
    best_score : float
        Reconstruction loss (MSE) for that event.
    """
    best_event = None
    best_score = -1.0
    count = 0

    for df_chunk in stream_lhco_events(data_path):
        for i in range(len(df_chunk)):
            if count >= n_scan:
                break

            event = df_chunk.iloc[i].values
            graph = event_to_graph(event)
            if graph is None or graph.x.size(0) < 2:
                count += 1
                continue

            graph = graph.to(DEVICE)
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                recon_x, _ = model(graph.x, graph.edge_index, batch)

            mse = torch.mean((recon_x - graph.x) ** 2).item()

            if mse > best_score:
                best_score = mse
                best_event = event

            count += 1

        if count >= n_scan:
            break

    print(f"Scanned {count} signal events.  Best anomaly score: {best_score:.6f}")
    return best_event, best_score


########################################
# Compute saliency via backpropagation
########################################


def compute_saliency(model: NodeGNN, event) -> tuple:
    """Compute per-node saliency via input-gradient backpropagation.

    Steps
    -----
    1. Convert event to a PyG graph.
    2. Enable ``requires_grad`` on the input node features.
    3. Forward pass → reconstruction loss (MSE).
    4. Backward pass → gradients w.r.t. input features.
    5. Saliency = L2 norm of the gradient vector per node.

    Parameters
    ----------
    model : NodeGNN
        Trained model (will be set to eval mode internally).
    event : np.ndarray
        Raw event array (1D).

    Returns
    -------
    particles : np.ndarray
        (N, 3) array of [pT, η, φ] for each particle.
    saliency : np.ndarray
        (N,) array of saliency magnitudes.
    """
    model.eval()

    graph = event_to_graph(event).to(DEVICE)
    x = graph.x.clone().detach().requires_grad_(True)

    # Create batch tensor for single-graph inference
    batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)

    recon_x, _ = model(x, graph.edge_index, batch)

    # Use raw features as target (consistent with training objective)
    target = graph.x.clone().detach()

    loss = torch.mean((recon_x - target) ** 2)
    loss.backward()

    # Saliency = L2 norm of gradient per node (across 3 features)
    grad = x.grad.detach().cpu().numpy()  # (N, 3)
    saliency = np.linalg.norm(grad, axis=1)  # (N,)

    particles = extract_particles(event)  # (N, 3): [pT, eta, phi]
    return particles, saliency


########################################
# Visualization
########################################


def plot_saliency(particles: np.ndarray, saliency: np.ndarray, save_path: str):
    """Create a 1×2 saliency figure and save to disk.

    Left panel : scatter in (η, φ) colored by saliency (inferno).
    Right panel: horizontal bar chart of the top-10 most salient particles.

    Parameters
    ----------
    particles : np.ndarray
        (N, 3) array of [pT, η, φ].
    saliency : np.ndarray
        (N,) saliency magnitudes.
    save_path : str
        Output PNG path.
    """
    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]

    norm = Normalize(vmin=saliency.min(), vmax=saliency.max())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ########################################
    # Left panel – (eta, phi) saliency map
    ########################################
    ax = axes[0]
    sc = ax.scatter(
        eta,
        phi,
        c=saliency,
        cmap="inferno",
        norm=norm,
        s=np.clip(pt / pt.max() * 120, 10, 200),  # size ∝ pT
        edgecolors="white",
        linewidths=0.4,
        alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Saliency (‖∇x L‖₂)", fontsize=11)
    ax.set_xlabel("η", fontsize=12)
    ax.set_ylabel("φ", fontsize=12)
    ax.set_title("Particle Saliency Map (η, φ)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ########################################
    # Right panel – top-10 bar chart
    ########################################
    ax = axes[1]
    top_k = min(10, len(saliency))
    top_idx = np.argsort(saliency)[::-1][:top_k]

    labels = [
        f"pT={pt[i]:.0f}  η={eta[i]:.2f}  φ={phi[i]:.2f}"
        for i in top_idx
    ]
    colors = plt.cm.inferno(norm(saliency[top_idx]))

    y_pos = np.arange(top_k)
    ax.barh(y_pos, saliency[top_idx], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Saliency (‖∇x L‖₂)", fontsize=11)
    ax.set_title("Top-10 Most Salient Particles", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saliency map saved → {save_path}")


########################################
# Main
########################################

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)

    print(f"Scanning signal events from {SIGNAL_PATH} ...")
    event, score = select_high_anomaly_event(model, SIGNAL_PATH)

    if event is None:
        raise RuntimeError("No valid signal events found – check data path.")

    print("Computing saliency via backpropagation ...")
    particles, saliency = compute_saliency(model, event)
    print(f"  Particles: {len(saliency)}   Max saliency: {saliency.max():.6f}")

    print("Generating saliency figure ...")
    plot_saliency(particles, saliency, OUTPUT_PATH)
