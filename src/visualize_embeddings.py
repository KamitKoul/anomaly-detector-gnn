"""
visualize_embeddings.py
-----------------------
Dimensionality-reduction analytics for the GNN anomaly detector.

Takes the 32-dimensional bottleneck latent vector produced by the
NodeGNN encoder and projects it to 2D using both t-SNE and PCA.
Background (cyan) and signal (magenta) events are overlaid so that
separation in the learned embedding space is immediately visible.

Output: gnn_embeddings_tsne.png  (1×2 figure, t-SNE | PCA)
"""

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from src.run_lhco_gnn import (
    stream_lhco_events,
    event_to_graph,
    NodeGNN,
    DEVICE,
)

########################################
# Paths & constants
########################################

MODEL_PATH = "models/discovery_model.pt"
BKG_DATA_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"
SIG_DATA_PATH = "data/W_prime_signal.h5"
OUTPUT_PATH = "gnn_embeddings_tsne.png"

N_EVENTS = 500  # number of events to sample per class


########################################
# Embedding extraction helpers
########################################

def load_model(model_path: str) -> NodeGNN:
    """Load a trained NodeGNN from a state-dict checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the saved ``.pt`` state dict.

    Returns
    -------
    NodeGNN
        Model in eval mode on the project DEVICE.
    """
    model = NodeGNN().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def extract_embedding(model: NodeGNN, data) -> np.ndarray:
    """Run a single graph through the encoder and return the
    global-mean-pooled 32-dim embedding as a NumPy vector.

    Parameters
    ----------
    model : NodeGNN
        Trained model (eval mode).
    data : torch_geometric.data.Data
        Single-event graph with ``x``, ``edge_index``, and ``batch``.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(32,)``.
    """
    data = data.to(DEVICE)

    # --- Encoder forward pass (mirrors NodeGNN.forward) ---
    # Layer 1: GATConv (3 → 32, 4 heads → 128 features)
    h = model.gat1(data.x, data.edge_index)
    h = torch.nn.functional.elu(h)

    # Layer 2: GATConv (128 → 32, 1 head)
    h = model.gat2(h, data.edge_index)
    h = torch.nn.functional.elu(h)

    # Global mean pool → 32-dim event embedding
    batch = data.batch if data.batch is not None else torch.zeros(
        data.x.size(0), dtype=torch.long, device=DEVICE
    )
    embedding = global_mean_pool(h, batch)  # (1, 32)

    return embedding.squeeze(0).cpu().numpy()


def collect_embeddings(data_path: str, model: NodeGNN, n_events: int) -> np.ndarray:
    """Stream events from an HDF5 file, convert to graphs, and collect
    the latent embeddings.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 event file.
    model : NodeGNN
        Trained model in eval mode.
    n_events : int
        Number of events to process.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_events, 32)``.
    """
    embeddings = []
    count = 0

    for df_chunk in stream_lhco_events(data_path):
        for i in range(len(df_chunk)):
            if count >= n_events:
                break

            event = df_chunk.iloc[i].values
            graph = event_to_graph(event)
            if graph is None or graph.x.size(0) == 0:
                continue

            emb = extract_embedding(model, graph)
            embeddings.append(emb)
            count += 1

        if count >= n_events:
            break

    return np.stack(embeddings, axis=0)


########################################
# Dimensionality reduction
########################################

def run_reductions(embeddings: np.ndarray):
    """Apply t-SNE and PCA to the combined embedding matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape ``(N, 32)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(tsne_2d, pca_2d)`` each of shape ``(N, 2)``.
    """
    print("Running t-SNE …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(embeddings)

    print("Running PCA …")
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(embeddings)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")

    return tsne_2d, pca_2d


########################################
# Plotting
########################################

def plot_embeddings(
    tsne_2d: np.ndarray,
    pca_2d: np.ndarray,
    n_bkg: int,
    n_sig: int,
    save_path: str,
) -> None:
    """Create a 1×2 figure with t-SNE (left) and PCA (right).

    Parameters
    ----------
    tsne_2d : np.ndarray
        t-SNE projected coordinates, shape ``(N, 2)``.
    pca_2d : np.ndarray
        PCA projected coordinates, shape ``(N, 2)``.
    n_bkg : int
        Number of background events (first n_bkg rows).
    n_sig : int
        Number of signal events (remaining rows).
    save_path : str
        Output file path for the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, proj, title in zip(axes, [tsne_2d, pca_2d], ["t-SNE", "PCA"]):
        ax.scatter(
            proj[:n_bkg, 0],
            proj[:n_bkg, 1],
            c="cyan",
            alpha=0.5,
            s=8,
            label="Background",
        )
        ax.scatter(
            proj[n_bkg:, 0],
            proj[n_bkg:, 1],
            c="magenta",
            alpha=0.5,
            s=8,
            label="Signal",
        )
        ax.set_title(f"{title}  –  GNN Latent Embeddings (32-D → 2-D)", fontsize=13)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(loc="best", fontsize=10, markerscale=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved embedding plot → {save_path}")


########################################
# Main
########################################

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # ---- Load trained model ----
    print(f"Loading model from {MODEL_PATH} …")
    model = load_model(MODEL_PATH)

    # ---- Collect background embeddings ----
    print(f"\nExtracting background embeddings ({N_EVENTS} events) …")
    bkg_embeddings = collect_embeddings(BKG_DATA_PATH, model, N_EVENTS)
    print(f"  Background embeddings shape: {bkg_embeddings.shape}")

    # ---- Collect signal embeddings ----
    print(f"\nExtracting signal embeddings ({N_EVENTS} events) …")
    sig_embeddings = collect_embeddings(SIG_DATA_PATH, model, N_EVENTS)
    print(f"  Signal embeddings shape: {sig_embeddings.shape}")

    # ---- Combine & reduce ----
    all_embeddings = np.concatenate([bkg_embeddings, sig_embeddings], axis=0)
    print(f"\nCombined embeddings shape: {all_embeddings.shape}")

    tsne_2d, pca_2d = run_reductions(all_embeddings)

    # ---- Plot ----
    plot_embeddings(
        tsne_2d,
        pca_2d,
        n_bkg=len(bkg_embeddings),
        n_sig=len(sig_embeddings),
        save_path=OUTPUT_PATH,
    )

    print("Done.")
