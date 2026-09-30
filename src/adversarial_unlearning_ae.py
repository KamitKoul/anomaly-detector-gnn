"""
adversarial_unlearning_ae.py — Adversarial Mass-Decorrelated Anomaly Detection
=============================================================================
Implements Gradient Reversal Layer (GRL) mechanics to decorrelate the latent
representations learned by the GNN autoencoder from invariant mass (m_jj).
This prevents "mass sculpting" — a critical artifact in high energy physics
where anomaly detection filters artificially shape background events to mimic
a resonance peak.

Architecture:
  - Base Encoder: GATConv (conv1, conv2) + global_mean_pool
  - Reconstruction Head: 32 -> 16 -> 3 (node-level MSE)
  - MET Predictor Head: 32 -> 16 -> 2 (global physics conservation)
  - Adversary Head: 32 -> 32 -> 16 -> 1 via GRL (invariant mass regression)

Outputs:
  - Model weights: models/unlearning_model.pt
  - Evaluation plot: adversarial_unlearning_proof.png
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

sys.path.append(os.getcwd())

from src.run_lhco_gnn import (
    stream_lhco_events,
    event_to_graph,
    extract_particles,
    DEVICE,
)


########################################
# CONFIG & PATHS
########################################

BG_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"
SIG_PATH = "data/W_prime_signal.h5"
PRETRAINED_MODEL_PATH = "models/discovery_model.pt"
UNLEARNING_MODEL_PATH = "models/unlearning_model.pt"
OUTPUT_PLOT_PATH = "adversarial_unlearning_proof.png"

TRAIN_LIMIT = 5000
EVAL_LIMIT = 500
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
LAMBDA_MET = 0.1
LAMBDA_ADV = 0.05
MASS_NORM = 1000.0  # Normalize GeV -> TeV scale for stable gradients


########################################
# GRADIENT REVERSAL LAYER (GRL)
########################################

class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) as introduced by Ganin et al.
    During the forward pass, this acts as the identity mapping.
    During the backward pass, gradients are multiplied by -alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


########################################
# INVARIANT MASS CALCULATION
########################################

def compute_invariant_mass(particles):
    """
    Computes system invariant mass in GeV via relativistic four-vector summation:
      E = pT * cosh(eta), pz = pT * sinh(eta)
      M = sqrt(E^2 - px^2 - py^2 - pz^2)
    """
    if len(particles) == 0:
        return 0.0

    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]

    E = pt * np.cosh(eta)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    total_E = np.sum(E)
    total_px = np.sum(px)
    total_py = np.sum(py)
    total_pz = np.sum(pz)

    mass_sq = total_E**2 - total_px**2 - total_py**2 - total_pz**2
    return float(np.sqrt(max(0.0, mass_sq)))


########################################
# ADVERSARIAL GNN ARCHITECTURE
########################################

class AdversarialNodeGNN(nn.Module):
    """
    Multi-Task Graph Attention Autoencoder with Adversarial Mass Unlearning.

    Layer structure matches models/unlearning_model.pt exactly:
      conv1: GATConv(3 -> 32, heads=4, concat=True)
      conv2: GATConv(128 -> 32, heads=1, concat=False)
      decoder: Linear(32 -> 16) -> ELU -> Linear(16 -> 3)
      met_head: Linear(32 -> 16) -> ELU -> Linear(16 -> 2)
      adversary: Linear(32 -> 32) -> ELU -> Linear(32 -> 16) -> ELU -> Linear(16 -> 1)
    """

    def __init__(self, input_dim=3, hidden_dim=32, num_heads=4):
        super().__init__()

        # Encoder: Spatial graph attention layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)

        # Decoder Head: Node-Level Kinematic Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, input_dim),
        )

        # Physics Head: Global MET Conservation
        self.met_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, 2),
        )

        # Adversary Head: Invariant Mass Regressor (conditioned through GRL)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, edge_index, batch, alpha=1.0):
        # 1. Spatial Message Passing
        z = F.elu(self.conv1(x, edge_index))
        z = F.elu(self.conv2(z, edge_index))

        # 2. Particle Reconstruction
        recon_x = self.decoder(z)

        # 3. Global Graph Pooling
        global_z = global_mean_pool(z, batch)

        # 4. Global MET Prediction
        pred_met = self.met_head(global_z)

        # 5. Adversarial Invariant Mass Prediction (via GRL)
        reversed_z = grad_reverse(global_z, alpha)
        pred_mass = self.adversary(reversed_z)

        return recon_x, pred_met, pred_mass


########################################
# DATASET WITH INVARIANT MASS
########################################

class LHCOUnlearningDataset(Dataset):
    """
    Dataset streaming raw events and dynamically computing graphs with
    both MET targets (data.y) and invariant mass targets (data.mass).
    """

    def __init__(self, data_path, limit=TRAIN_LIMIT):
        self.data_path = data_path
        self.events = []
        self.masses = []

        print(f"Loading up to {limit} events from {data_path}...")
        count = 0
        for df_chunk in stream_lhco_events(data_path):
            for i in range(len(df_chunk)):
                if count >= limit:
                    break
                event = df_chunk.iloc[i].values
                particles = extract_particles(event)
                mass = compute_invariant_mass(particles) / MASS_NORM

                self.events.append(event)
                self.masses.append(mass)
                count += 1
            if count >= limit:
                break
        print(f"Loaded {len(self.events)} events.")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        graph = event_to_graph(self.events[idx])
        graph.mass = torch.tensor([self.masses[idx]], dtype=torch.float)
        return graph


########################################
# TRAINING ENGINE
########################################

def train_adversarial_unlearning():
    device = DEVICE
    print(f"\n{'='*60}")
    print(f"  ADVERSARIAL UNLEARNING GNN — TRAINING ACTIVE ON: {device}")
    print(f"{'='*60}\n")

    model = AdversarialNodeGNN().to(device)

    # Bootstrap weights from discovery_model.pt if present
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Bootstrapping from pretrained model: {PRETRAINED_MODEL_PATH}")
        base_state = torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True)
        model_state = model.state_dict()
        compatible_weights = {k: v for k, v in base_state.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(compatible_weights)
        model.load_state_dict(model_state)
        print(f"Transferred {len(compatible_weights)} compatible weight tensors.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_recon = nn.MSELoss()
    criterion_met = nn.MSELoss()
    criterion_adv = nn.MSELoss()

    train_dataset = LHCOUnlearningDataset(BG_PATH, limit=TRAIN_LIMIT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Beginning Adversarial Training Loop...")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        total_recon = 0.0
        total_met = 0.0
        total_adv = 0.0

        # Dynamically scale alpha (Ganin et al. GRL schedule)
        p = float(epoch) / max(1, NUM_EPOCHS)
        alpha = float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (α={alpha:.2f})")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()

            recon_x, pred_met, pred_mass = model(data.x, data.edge_index, data.batch, alpha=alpha)

            loss_recon = criterion_recon(recon_x, data.x)
            loss_met = criterion_met(pred_met, data.y.view(-1, 2))
            loss_adv = criterion_adv(pred_mass, data.mass.view(-1, 1))

            loss = loss_recon + LAMBDA_MET * loss_met + LAMBDA_ADV * loss_adv
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_met += loss_met.item()
            total_adv += loss_adv.item()

            pbar.set_postfix({
                "Rec": f"{loss_recon.item():.4f}",
                "MET": f"{loss_met.item():.4f}",
                "Adv": f"{loss_adv.item():.4f}",
            })

        n_batches = len(train_loader)
        print(f"Epoch {epoch+1} Complete | Total: {total_loss/n_batches:.4f} "
              f"| Recon: {total_recon/n_batches:.4f} | MET: {total_met/n_batches:.4f} | Adv: {total_adv/n_batches:.4f}")

    # Save model weights
    os.makedirs(os.path.dirname(UNLEARNING_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), UNLEARNING_MODEL_PATH)
    print(f"\nSaved trained adversarial unlearning model to: {UNLEARNING_MODEL_PATH}")

    # Evaluate & Plot
    evaluate_and_plot(model, device)


########################################
# EVALUATION & PLOT
########################################

def evaluate_and_plot(model, device):
    """
    Evaluates mass-decorrelated anomaly scores on background vs signal
    and produces adversarial_unlearning_proof.png.
    """
    print("\nEvaluating Mass-Decorrelated Anomaly Scoring...")
    model.eval()

    bg_scores = []
    sig_scores = []

    def score_dataset(path, limit, label):
        scores = []
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {label} scoring.")
            return scores
        dataset = LHCOUnlearningDataset(path, limit=limit)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in tqdm(loader, desc=f"Scoring {label}"):
                data = data.to(device)
                recon_x, _, _ = model(data.x, data.edge_index, data.batch, alpha=0.0)
                score = F.mse_loss(recon_x, data.x).item()
                scores.append(score)
        return scores

    bg_scores = score_dataset(BG_PATH, EVAL_LIMIT, "Background")
    sig_scores = score_dataset(SIG_PATH, EVAL_LIMIT, "W' Signal")

    if len(bg_scores) == 0:
        print("No background scores to plot.")
        return

    # Visual proof plot matching project style (dark background, teal & magenta)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(min(bg_scores), np.percentile(bg_scores + sig_scores, 98) if sig_scores else max(bg_scores), 50)
    ax.hist(bg_scores, bins=bins, color="#00a884", alpha=0.75, label="Background (Decorrelated)", density=True)
    if sig_scores:
        ax.hist(sig_scores, bins=bins, color="#a8005b", alpha=0.75, label="W' Signal (Decorrelated)", density=True)

    ax.set_xlabel("Decorrelated Anomaly Score (MSE)", fontsize=11, color="white")
    ax.set_ylabel("Density", fontsize=11, color="white")
    ax.set_title("Adversarial Mass-Decorrelated Latent Separation", fontsize=13, color="white", weight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved unlearning validation plot to: {OUTPUT_PLOT_PATH}")


########################################
# ENTRY POINT
########################################

if __name__ == "__main__":
    train_adversarial_unlearning()
