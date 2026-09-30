"""
Visualize GNN Message Passing Flow

Tracks node activation matrices and topological propagation through
the Graph Attention Network layers. Registers forward hooks on each
GATConv layer to capture intermediate activations and attention weights,
then renders a multi-panel figure showing:

  Panel 1: Input graph with particle nodes (colored by pT)
  Panel 2: First GATConv attention weights (edge thickness = attention)
  Panel 3: Node activation heatmap after GATConv Layer 1
  Panel 4: Node activation heatmap after GATConv Layer 2

Output: gnn_message_passing.png
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

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

MODEL_PATH = "models/discovery_model.pt"
OUTPUT_PATH = "gnn_message_passing.png"
MAX_DISPLAY_NODES = 80  # Cap for readable visualization


########################################
# HOOK STORAGE
########################################

class ActivationStore:
    """Stores intermediate activations and attention weights captured by hooks."""

    def __init__(self):
        self.activations = {}
        self.attention_weights = {}

    def clear(self):
        self.activations.clear()
        self.attention_weights.clear()


store = ActivationStore()


########################################
# FORWARD HOOK FACTORIES
########################################

def make_activation_hook(layer_name):
    """
    Creates a forward hook that captures the output activation tensor
    for a named GATConv layer.

    GATConv.forward returns (out, (edge_index, alpha)) when
    return_attention_weights=True, or just out otherwise.
    We capture both cases.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            # (out_tensor, (edge_index, attention_coefficients))
            store.activations[layer_name] = output[0].detach().cpu()
            edge_index, alpha = output[1]
            store.attention_weights[layer_name] = (
                edge_index.detach().cpu(),
                alpha.detach().cpu(),
            )
        else:
            store.activations[layer_name] = output.detach().cpu()
    return hook


########################################
# LOAD MODEL WITH HOOKS
########################################

def load_model_with_hooks():
    """
    Loads the trained NodeGNN from disk, enables attention weight
    returns on the GAT layers, and registers forward hooks.

    Returns:
        model: NodeGNN in eval mode on DEVICE
        handles: list of hook handles (for cleanup)
    """
    model = NodeGNN().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found — using random weights for visualization")

    model.eval()

    # Enable attention weight output on GAT layers
    model.gat1.return_attention_weights = True
    model.gat2.return_attention_weights = True

    # Register hooks
    handles = []
    handles.append(model.gat1.register_forward_hook(make_activation_hook("gat1")))
    handles.append(model.gat2.register_forward_hook(make_activation_hook("gat2")))

    return model, handles


########################################
# FETCH SAMPLE EVENT
########################################

def get_sample_graph():
    """
    Streams the first chunk of background events and returns
    a single graph (the first event with enough particles for
    an interesting visualization).

    Returns:
        data: PyG Data object
        particles: raw Nx3 numpy array [pt, eta, phi]
    """
    for df_chunk in stream_lhco_events():
        for i in range(len(df_chunk)):
            event = df_chunk.iloc[i].values
            particles = extract_particles(event)

            # Skip trivially small events
            if len(particles) < 5:
                continue

            graph = event_to_graph(event)
            return graph, particles

    raise RuntimeError("No suitable event found in dataset")


########################################
# GRAPH LAYOUT HELPERS
########################################

def pyg_to_networkx(edge_index, num_nodes):
    """Converts a PyG edge_index tensor to a NetworkX Graph."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    for s, d in zip(src, dst):
        if s < d:  # avoid duplicate undirected edges
            G.add_edge(int(s), int(d))

    return G


def truncate_graph(data, max_nodes):
    """
    Truncates a graph to the first max_nodes nodes for readability.
    Re-indexes edges accordingly.
    """
    if data.x.size(0) <= max_nodes:
        return data

    x = data.x[:max_nodes]
    edge_index = data.edge_index

    # Keep only edges within the truncated node set
    mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
    edge_index = edge_index[:, mask]

    from torch_geometric.data import Data
    return Data(x=x, edge_index=edge_index)


########################################
# PANEL 1: INPUT GRAPH
########################################

def draw_input_graph(ax, G, pos, pt_values, num_nodes):
    """
    Draws the raw particle graph with nodes colored by pT.
    """
    node_sizes = np.clip(pt_values[:num_nodes] * 2, 20, 300)
    node_colors = pt_values[:num_nodes]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="gray", width=0.5)
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="$p_T$ (GeV)", shrink=0.7, pad=0.02)
    ax.set_title("Input Graph\n(nodes = particles, color = $p_T$)", fontsize=10, fontweight="bold")
    ax.axis("off")


########################################
# PANEL 2: ATTENTION WEIGHTED EDGES
########################################

def draw_attention_edges(ax, G, pos, num_nodes):
    """
    Draws edges with thickness proportional to the mean attention
    weight from the first GATConv layer's multi-head attention.
    """
    if "gat1" not in store.attention_weights:
        ax.text(0.5, 0.5, "No attention weights\ncaptured",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("GATConv-1 Attention", fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    attn_edge_index, alpha = store.attention_weights["gat1"]

    # alpha shape: [num_edges, num_heads] — average across heads
    if alpha.dim() > 1:
        alpha_mean = alpha.mean(dim=1).numpy()
    else:
        alpha_mean = alpha.numpy()

    # Build edge→weight map (undirected: average both directions)
    edge_weights = {}
    src = attn_edge_index[0].numpy()
    dst = attn_edge_index[1].numpy()
    for s, d, w in zip(src, dst, alpha_mean):
        if s >= num_nodes or d >= num_nodes:
            continue
        key = (min(int(s), int(d)), max(int(s), int(d)))
        edge_weights.setdefault(key, []).append(w)

    # Draw nodes first
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=25,
                           node_color="steelblue", alpha=0.7)

    # Draw edges with variable width
    if edge_weights:
        edges = list(edge_weights.keys())
        weights = [np.mean(edge_weights[e]) for e in edges]
        w_min, w_max = min(weights), max(weights)

        # Normalize widths to [0.3, 4.0]
        if w_max > w_min:
            norm_w = [(w - w_min) / (w_max - w_min) * 3.7 + 0.3 for w in weights]
        else:
            norm_w = [2.0] * len(weights)

        # Color edges by weight
        cmap = plt.cm.hot_r
        norm = mcolors.Normalize(vmin=w_min, vmax=w_max)

        for (u, v), nw, w in zip(edges, norm_w, weights):
            if G.has_edge(u, v):
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], ax=ax,
                    width=nw, edge_color=[cmap(norm(w))], alpha=0.8,
                )

    ax.set_title("GATConv-1 Attention Weights\n(thickness ∝ attention)", fontsize=10, fontweight="bold")
    ax.axis("off")


########################################
# PANEL 3 & 4: ACTIVATION HEATMAPS
########################################

def draw_activation_heatmap(ax, layer_name, title, num_nodes):
    """
    Draws a heatmap of node activations for a given layer.
    Rows = nodes, Columns = feature dimensions.
    """
    if layer_name not in store.activations:
        ax.text(0.5, 0.5, f"No activations for\n{layer_name}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    acts = store.activations[layer_name][:num_nodes].numpy()

    im = ax.imshow(acts, aspect="auto", cmap="magma", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="Activation")

    ax.set_xlabel("Feature Dimension", fontsize=9)
    ax.set_ylabel("Node Index", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")


########################################
# MAIN VISUALIZATION PIPELINE
########################################

def visualize_message_passing():
    """
    End-to-end pipeline:
      1. Load trained model with hooks
      2. Fetch a sample event graph
      3. Run forward pass to capture activations
      4. Render 4-panel figure
      5. Save to gnn_message_passing.png
    """
    print("=" * 50)
    print("  GNN Message Passing Visualization")
    print("=" * 50)

    # --- Load model ---
    model, handles = load_model_with_hooks()

    # --- Get sample graph ---
    print("Fetching sample event...")
    data, raw_particles = get_sample_graph()

    # Truncate for readability
    data = truncate_graph(data, MAX_DISPLAY_NODES)
    num_nodes = data.x.size(0)
    print(f"Visualizing graph with {num_nodes} nodes, {data.edge_index.size(1)} edges")

    # --- Forward pass (captures activations via hooks) ---
    store.clear()
    data_device = data.to(DEVICE)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        recon_x, pred_met = model(data_device.x, data_device.edge_index, batch)

    print(f"Captured activations: {list(store.activations.keys())}")
    print(f"Captured attention:   {list(store.attention_weights.keys())}")

    # --- Build NetworkX layout ---
    G = pyg_to_networkx(data.edge_index, num_nodes)
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(max(num_nodes, 1)), iterations=60)

    # pT values for coloring (first feature column)
    pt_values = data.x[:, 0].numpy()

    # --- Render multi-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        "GNN Message Passing Flow — Graph Attention Network",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # Panel 1: Input graph
    draw_input_graph(axes[0, 0], G, pos, pt_values, num_nodes)

    # Panel 2: Attention-weighted edges
    draw_attention_edges(axes[0, 1], G, pos, num_nodes)

    # Panel 3: Activation heatmap after GATConv Layer 1
    draw_activation_heatmap(
        axes[1, 0], "gat1",
        "Node Activations — GATConv Layer 1\n(128-dim: 32 × 4 heads)",
        num_nodes,
    )

    # Panel 4: Activation heatmap after GATConv Layer 2
    draw_activation_heatmap(
        axes[1, 1], "gat2",
        "Node Activations — GATConv Layer 2\n(32-dim latent space)",
        num_nodes,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved visualization to {OUTPUT_PATH}")
    plt.close(fig)

    # --- Cleanup hooks ---
    for h in handles:
        h.remove()

    # --- Print summary statistics ---
    for name in ["gat1", "gat2"]:
        if name in store.activations:
            act = store.activations[name]
            print(f"  {name}: shape={list(act.shape)}, "
                  f"mean={act.mean():.4f}, std={act.std():.4f}, "
                  f"min={act.min():.4f}, max={act.max():.4f}")

    print("\nDone.")


########################################
# ENTRY POINT
########################################

if __name__ == "__main__":
    visualize_message_passing()
