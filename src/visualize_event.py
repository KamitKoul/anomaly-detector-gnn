"""
visualize_event.py
==================
Topographic detector-view visualisation of LHC collision events.

Each event is rendered in the cylindrical (η, φ) coordinate plane that
naturally maps to the detector surface.  Particles are shown as scatter
points whose size is proportional to transverse momentum (pT), coloured
by pT magnitude.  Edges connect particle pairs within ΔR < 0.4, giving
an intuitive picture of the graph structure fed into the GNN.

Output
------
event_visualization.png  –  2×2 grid of four sample background events.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.getcwd())

from src.run_lhco_gnn import (
    stream_lhco_events,
    extract_particles,
    build_deltaR_edges,
    DELTA_R_THRESHOLD,
)

########################################
# Plotting helpers
########################################

def plot_event(ax, particles, edge_index, event_idx):
    """Draw a single event on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (already styled with dark background).
    particles : np.ndarray, shape (N, 3)
        Columns are [pT, eta, phi].
    edge_index : np.ndarray, shape (2, E)
        Source / target indices for each edge.
    event_idx : int
        Event number used in the subplot title.
    """
    pt  = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]

    # ── edges ────────────────────────────────────────────────────────
    for src, dst in zip(edge_index[0], edge_index[1]):
        ax.plot(
            [eta[src], eta[dst]],
            [phi[src], phi[dst]],
            color="steelblue",
            alpha=0.15,
            linewidth=0.4,
            zorder=1,
        )

    # ── particles ────────────────────────────────────────────────────
    # Marker area proportional to pT (clamped for visibility)
    sizes = np.clip(pt, 1, None)
    sizes = (sizes / sizes.max()) * 250 + 10

    norm = mcolors.LogNorm(vmin=max(pt.min(), 0.5), vmax=pt.max())
    sc = ax.scatter(
        eta,
        phi,
        c=pt,
        s=sizes,
        cmap="inferno",
        norm=norm,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
    )

    ax.set_xlabel("η", fontsize=10)
    ax.set_ylabel("φ", fontsize=10)
    ax.set_title(
        f"Event {event_idx}  ·  {len(pt)} particles  ·  {edge_index.shape[1]} edges",
        fontsize=9,
        color="white",
    )
    ax.set_xlim(-5, 5)
    ax.set_ylim(-np.pi, np.pi)

    return sc


########################################
# Main visualisation routine
########################################

def visualize_events(n_events=4):
    """Load sample events and produce a 2×2 detector-view grid.

    Parameters
    ----------
    n_events : int
        Number of events to display (default 4, arranged 2×2).
    """
    print(f"Loading {n_events} sample events …")

    events = []
    for chunk_df in stream_lhco_events():
        for i in range(len(chunk_df)):
            event = chunk_df.iloc[i].values
            particles = extract_particles(event)
            edge_index = build_deltaR_edges(particles)
            events.append((particles, edge_index.numpy()))
            if len(events) >= n_events:
                break
        if len(events) >= n_events:
            break

    if len(events) == 0:
        print("No events found – check DATA_PATH.")
        return

    # ── figure ───────────────────────────────────────────────────────
    plt.style.use("dark_background")

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 10),
        constrained_layout=True,
    )
    fig.suptitle(
        f"Detector η–φ View  ·  ΔR < {DELTA_R_THRESHOLD} edges",
        fontsize=14,
        color="white",
        weight="bold",
    )

    sc = None
    for idx, ax in enumerate(axes.flat):
        if idx < len(events):
            particles, edge_index = events[idx]
            sc = plot_event(ax, particles, edge_index, event_idx=idx + 1)
        else:
            ax.set_visible(False)

    # Shared colour-bar
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_label("pT  [GeV]", fontsize=10)

    out_path = "event_visualization.png"
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


########################################
# Entry point
########################################

if __name__ == "__main__":
    visualize_events()
