import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric")

from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


########################################
# DEVICE DETECTION
########################################

def get_device(verbose: bool = False):
    """
    Selects compute device with priority: CUDA -> MPS -> CPU.
    When verbose=True, explicitly prints the selected hardware name.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(device)
            print(f"[DEVICE] Selected NVIDIA CUDA GPU: {gpu_name}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[DEVICE] Selected Apple Silicon Metal Performance Shaders (MPS)")
        return device
    device = torch.device("cpu")
    if verbose:
        print("[DEVICE] Fallback to CPU execution")
    return device

DEVICE = get_device()


########################################
# CONFIG
########################################

DATA_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"
CHUNK_SIZE = 50
DELTA_R_THRESHOLD = 0.4


import tables

########################################
# STREAM DATASET
########################################

def stream_lhco_events(data_path=DATA_PATH):

    h5file = tables.open_file(data_path, mode='r')

    try:
        # In pandas fixed format, the data is stored in /df/block0_values
        node = h5file.get_node("/df/block0_values")
        total_rows = node.nrows

        print("Total events in dataset:", total_rows)

        for start in range(0, total_rows, CHUNK_SIZE):

            stop = min(start + CHUNK_SIZE, total_rows)

            chunk = node[start:stop]

            # Yield as DataFrame to maintain compatibility with .iloc[i].values
            yield pd.DataFrame(chunk)

    finally:
        h5file.close()


def extract_particles(event):
    """
    Extracts up to 700 particles from an event array.
    LHCO R&D file has 2101 columns (2100 features + 1 truth label).
    We only take the first 2100 values.
    """
    particles = []

    # Only process the first 2100 values to avoid including truth labels
    event_data = event[:2100]

    for i in range(700):
        pt = event_data[i*3]
        eta = event_data[i*3 + 1]
        phi = event_data[i*3 + 2]

        if pt > 0:
            particles.append([pt, eta, phi])

    if len(particles) == 0:
        return np.zeros((1,3))

    return np.array(particles)


########################################
# COMPUTE MET
########################################

def compute_MET(particles):

    px = particles[:,0] * np.cos(particles[:,2])
    py = particles[:,0] * np.sin(particles[:,2])

    MET_x = -np.sum(px)
    MET_y = -np.sum(py)

    MET = np.sqrt(MET_x**2 + MET_y**2)

    return MET, MET_x, MET_y


########################################
# DELTA R CALCULATION
########################################

def delta_r(eta1, phi1, eta2, phi2):

    d_eta = eta1 - eta2
    d_phi = abs(phi1 - phi2)

    if d_phi > np.pi:
        d_phi = 2*np.pi - d_phi

    return np.sqrt(d_eta**2 + d_phi**2)


def periodic_dphi(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    """
    Computes minimum angular difference wrapped onto [0, pi] on the S^1 circle.
    Differentiable and continuous across the -pi/+pi branch cut seam.
    """
    dphi = torch.abs(phi1 - phi2)
    return torch.min(dphi, 2.0 * torch.pi - dphi)


def compute_kinematic_recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    pt_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Physically rigorous reconstruction loss for collider coordinates (pT, eta, phi).
    Respects S^1 periodic Riemannian geometry on azimuthal angle phi.

    Args:
        recon: Tensor of shape [N, 3] representing [ln(1+pT), eta, phi]
        target: Tensor of shape [N, 3] representing true [ln(1+pT), eta, phi]
        pt_weight: Loss weight for pT component
        ang_weight: Loss weight for angular (eta, phi) component

    Returns:
        total_loss, loss_pt, loss_ang
    """
    pt_err = (recon[:, 0] - target[:, 0]).pow(2)
    eta_err = (recon[:, 1] - target[:, 1]).pow(2)
    phi_err = periodic_dphi(recon[:, 2], target[:, 2]).pow(2)

    loss_pt = pt_err.mean()
    loss_ang = 0.5 * (eta_err.mean() + phi_err.mean())
    total_loss = pt_weight * loss_pt + ang_weight * loss_ang
    return total_loss, loss_pt, loss_ang


########################################
# BUILD ΔR GRAPH (VECTORIZED)
########################################

def build_deltaR_edges(particles, threshold=DELTA_R_THRESHOLD):
    """
    Vectorized computation of delta R graph edges:
      dr = sqrt((d_eta)^2 + (d_phi)^2) with periodic phi wrap-around.
    Excludes self-loops and returns a [2, num_edges] long tensor.
    If threshold <= 0.0, immediately returns an empty edge index [2, 0].
    """
    if threshold <= 0.0:
        return torch.zeros((2, 0), dtype=torch.long)

    num_nodes = len(particles)
    if num_nodes <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)

    eta = particles[:, 1]
    phi = particles[:, 2]

    # Pairwise differences
    d_eta = eta[:, None] - eta[None, :]
    d_phi = np.abs(phi[:, None] - phi[None, :])
    d_phi = np.where(d_phi > np.pi, 2 * np.pi - d_phi, d_phi)

    dr = np.sqrt(d_eta**2 + d_phi**2)

    # Filter by threshold and exclude self-loops
    mask = (dr < threshold) & ~np.eye(num_nodes, dtype=bool)

    edges = np.argwhere(mask)
    if len(edges) == 0:
        return torch.tensor([[0], [0]], dtype=torch.long)

    return torch.tensor(edges.T, dtype=torch.long).contiguous()


########################################
# EVENT → GRAPH
########################################

def event_to_graph(
    event,
    log_pt=True,
    delta_r_threshold=DELTA_R_THRESHOLD,
    include_met_node=True,
    phi_met_mode="atan2",
):
    """
    Transforms a raw collision event into a PyG Data graph object.
    
    1. Extracts visible particles [pT, eta, phi].
    2. Computes event Missing Transverse Energy (MET).
    3. Adds a global MET node (if include_met_node=True) with azimuthal orientation
       determined by phi_met_mode ('atan2' for physical atan2(MET_y, MET_x), or 'zero'/'none').
    4. Computes topological delta-R edges (< delta_r_threshold).
    5. Applies log1p compression to pT to balance dynamic ranges (if log_pt=True).
    """
    particles = extract_particles(event)

    MET, MET_x, MET_y = compute_MET(particles)

    if include_met_node:
        if phi_met_mode == "atan2":
            phi_met = float(np.arctan2(MET_y, MET_x)) if MET > 0 else 0.0
        else:
            phi_met = 0.0
        MET_node = np.array([[MET, 0.0, phi_met]])
        all_particles = np.vstack([particles, MET_node])
    else:
        all_particles = particles

    # Build spatial proximity edges based on (eta, phi)
    edge_index = build_deltaR_edges(all_particles, threshold=delta_r_threshold)

    # Feature scaling: log1p(pT) balances scale with eta and phi
    features = all_particles.copy()
    if log_pt:
        features[:, 0] = np.log1p(features[:, 0])

    x = torch.tensor(features, dtype=torch.float)

    # Attach MET vector as global target for the physics prediction head
    y = torch.tensor([MET_x, MET_y], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def event_to_flat_features(event, input_dim=64, log_pt=True):
    """
    Extracts fixed-size flat tabular features for baseline MLP autoencoder.
    Truncates to input_dim features (default 64) and applies log1p scaling to particle pT columns (0, 3, 6, ...).
    """
    feat = np.array(event[:input_dim], dtype=np.float32).copy()
    if log_pt:
        for c in range(0, input_dim, 3):
            feat[c] = np.log1p(feat[c])
    return torch.tensor(feat, dtype=torch.float32)


def chunk_to_flat_features(chunk_array, input_dim=64, log_pt=True):
    """
    Vectorized extraction of flat tabular features for a 2D chunk of collision events.
    Applies column-sliced log1p scaling to particle pT columns (0, 3, 6, ...) without Python loops.
    """
    chunk_np = np.asarray(chunk_array)
    feat = np.array(chunk_np[:, :input_dim], dtype=np.float32, copy=True)
    if log_pt:
        feat[:, 0:input_dim:3] = np.log1p(feat[:, 0:input_dim:3])
    return torch.from_numpy(feat)


def stream_filtered_events(data_path, is_signal=None, chunk_size=1000):
    """
    Streams raw event chunks from an HDF5 file with optional truth filtering.
    If the file contains a truth column (index 2100 in 2101-column files):
      - is_signal=True yields only truth == 1 rows
      - is_signal=False yields only truth == 0 rows
      - is_signal=None yields all rows
    If the file has 2100 columns, all rows are background.
    """
    h5file = tables.open_file(data_path, mode='r')
    try:
        node = h5file.get_node("/df/block0_values")
        total_rows = node.nrows
        for start in range(0, total_rows, chunk_size):
            stop = min(start + chunk_size, total_rows)
            chunk = node[start:stop]
            if is_signal is not None and chunk.shape[1] > 2100:
                expected_val = 1.0 if is_signal else 0.0
                mask = (chunk[:, 2100] == expected_val)
                chunk = chunk[mask]
                if len(chunk) == 0:
                    continue
            yield chunk
    finally:
        h5file.close()


########################################
# GNN MODEL
########################################

class EventGNN(torch.nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.conv1 = GCNConv(input_dim,64)
        self.conv2 = GCNConv(64,64)

        self.lin = torch.nn.Linear(64,2)

    def forward(self,x,edge_index,batch):

        x = F.relu(self.conv1(x,edge_index))
        x = F.relu(self.conv2(x,edge_index))

        x = global_mean_pool(x,batch)

        x = self.lin(x)

        return x


########################################
# PRODUCTION GNN MODEL (Multi-Task GAT)
########################################

class NodeGNN(torch.nn.Module):
    """
    Multi-Task Graph Attention Autoencoder for Anomaly Detection.
    
    Architecture:
      Encoder: GATConv(3→32, 4 heads) [conv1] → ELU → GATConv(128→32, 1 head) [conv2] → ELU
      Decoder Head 1: Node Reconstruction (32→16→3) [decoder]
      Decoder Head 2: MET Prediction via global_mean_pool (32→16→2) [met_head]

    State-dict layer keys match models/discovery_model.pt:
      conv1.*, conv2.*, decoder.*, met_head.*
    """

    def __init__(self, input_dim=3, hidden_dim=32, num_heads=4):
        super().__init__()

        # Encoder: Graph Attention Layers (named conv1 and conv2 to match saved checkpoints)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)

        # Decoder Head 1: Node-Level Reconstruction (named decoder to match saved checkpoints)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, input_dim)
        )

        # Decoder Head 2: Global MET Prediction
        self.met_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, 2)  # Predicts [MET_x, MET_y]
        )

    # Aliases for backwards compatibility with scripts using gat1 / gat2 / recon_head
    @property
    def gat1(self):
        return self.conv1

    @gat1.setter
    def gat1(self, val):
        self.conv1 = val

    @property
    def gat2(self):
        return self.conv2

    @gat2.setter
    def gat2(self, val):
        self.conv2 = val

    @property
    def recon_head(self):
        return self.decoder

    @recon_head.setter
    def recon_head(self, val):
        self.decoder = val

    def encode(self, x, edge_index, batch):
        """Encodes particle nodes and returns node-level z and pooled global_z."""
        z = F.elu(self.conv1(x, edge_index))
        z = F.elu(self.conv2(z, edge_index))
        global_z = global_mean_pool(z, batch)
        return z, global_z

    def forward(self, x, edge_index, batch):

        # Encode
        z, global_z = self.encode(x, edge_index, batch)

        # Decode: Reconstruct node features
        recon_x = self.decoder(z)

        # Decode: Predict global MET
        pred_met = self.met_head(global_z)

        return recon_x, pred_met


########################################
# BASELINE MLP AUTOENCODER
########################################

class BaselineMLP(nn.Module):
    """
    Baseline Tabular Multi-Layer Perceptron Autoencoder.
    Layer dimensions and structure reproduce models/autoencoder_bg.pt:
      encoder.0: Linear(64, 32)
      encoder.1: ELU
      encoder.2: Linear(32, 8)
      decoder.0: Linear(8, 32)
      decoder.1: ELU
      decoder.2: Linear(32, 64)
    """

    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x


########################################
# CONTROLLED ARCHITECTURE 3: NODE-WISE AUTOENCODER (NO MESSAGE PASSING)
########################################

class NodeWiseAE(nn.Module):
    """
    Node-Wise Feedforward Autoencoder with Zero Relational Message Passing.
    Processes each particle x_i in the collision event independently (self-loops only).
    Architecture matches parameter budget of NodeGNN (~5.8k parameters):
      Encoder: Linear(3 -> 128) -> ELU -> Linear(128 -> 32) -> ELU
      Decoder: Linear(32 -> 16) -> ELU -> Linear(16 -> 3)
      Optional MET Head: Linear(32 -> 16) -> ELU -> Linear(16 -> 2) from global_mean_pool(z)
    """

    def __init__(self, input_dim=3, hidden_dim=32, wide_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, wide_dim),
            nn.ELU(),
            nn.Linear(wide_dim, hidden_dim),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, input_dim),
        )
        self.met_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, 2),
        )

    def encode(self, x, edge_index=None, batch=None):
        """Encodes particle nodes independently, returning node-level z and pooled global_z."""
        z = self.encoder(x)
        if batch is not None:
            global_z = global_mean_pool(z, batch)
        else:
            global_z = z.mean(dim=0, keepdim=True)
        return z, global_z

    def forward(self, x, edge_index=None, batch=None):
        z, global_z = self.encode(x, edge_index, batch)
        recon_x = self.decoder(z)
        pred_met = self.met_head(global_z)
        return recon_x, pred_met


########################################
# CONTROLLED ARCHITECTURE 4: PERMUTATION-INVARIANT POINT CLOUD (DEEPSETS)
########################################

class DeepSetsPointNet(nn.Module):
    """
    Permutation-Invariant Point Cloud Autoencoder (DeepSets / Particle PointNet).
    Aggregates per-particle embeddings via global mean pooling, then conditions each
    particle's reconstruction on the global event context. Zero pairwise edge operations.
    Architecture matches parameter budget (~5.6k parameters):
      Particle Encoder: Linear(3 -> 64) -> ELU -> Linear(64 -> 32) -> ELU
      Global Aggregator: global_mean_pool(phi(x_i)) -> rho in R^32
      Context Concatenation: z_i = [phi(x_i), rho] in R^64
      Conditioned Decoder: Linear(64 -> 32) -> ELU -> Linear(32 -> 16) -> ELU -> Linear(16 -> 3)
      MET Head: Linear(32 -> 16) -> ELU -> Linear(16 -> 2) from rho
    """

    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.particle_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, hidden_dim),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, input_dim),
        )
        self.met_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, 2),
        )

    def encode(self, x, edge_index=None, batch=None):
        """Encodes particles and conditions on global pooled representation."""
        phi = self.particle_encoder(x)
        if batch is not None:
            rho = global_mean_pool(phi, batch)
            rho_expanded = rho[batch]
        else:
            rho = phi.mean(dim=0, keepdim=True)
            rho_expanded = rho.expand(phi.size(0), -1)
        z = torch.cat([phi, rho_expanded], dim=-1)
        return z, rho

    def forward(self, x, edge_index=None, batch=None):
        z, rho = self.encode(x, edge_index, batch)
        recon_x = self.decoder(z)
        pred_met = self.met_head(rho)
        return recon_x, pred_met


########################################
# CONTROLLED ARCHITECTURE 5: HYBRID RELATIONAL POINTNET-GAT
########################################

class PointNetGATHybrid(nn.Module):
    """
    Hybrid Relational-PointNet Autoencoder for Collider Anomaly Detection.
    
    Fuses local pairwise relational attention (subjet substructure & energy flow)
    with global permutation-invariant point cloud context (event shape & angular moments).
    
    Architecture (~7.9k parameters):
      1. Local Relational GAT Branch:
         conv1: GATConv(3 -> 16, heads=4, concat=True) -> ELU (64 dim)
         conv2: GATConv(64 -> 32, heads=1, concat=False) -> ELU (32 dim)
      2. Global PointNet Branch:
         point_phi: Linear(3 -> 32) -> ELU -> Linear(32 -> 32) -> ELU (32 dim)
         rho: global_mean_pool(phi, batch) (32 dim)
      3. Multi-Scale Context Fusion:
         h_fused = [h_gat (32), phi (32), rho_expanded (32)] (96 dim)
      4. Conditioned Particle Decoder:
         Linear(96 -> 32) -> ELU -> Linear(32 -> 16) -> ELU -> Linear(16 -> 3)
      5. Global MET Prediction Head:
         Linear(32 -> 16) -> ELU -> Linear(16 -> 2) from rho
    """

    def __init__(self, input_dim=3, hidden_dim=32, num_heads=4):
        super().__init__()
        # 1. Local Relational GAT Branch
        self.conv1 = GATConv(input_dim, 16, heads=num_heads, concat=True)
        self.conv2 = GATConv(16 * num_heads, hidden_dim, heads=1, concat=False)

        # 2. Global PointNet Branch
        self.point_phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # 3. Conditioned Particle Reconstruction Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, input_dim),
        )

        # 4. Global MET Head
        self.met_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ELU(),
            nn.Linear(16, 2),
        )

    def encode(self, x, edge_index, batch=None):
        # Local Relational GAT
        h_gat = F.elu(self.conv1(x, edge_index))
        h_gat = F.elu(self.conv2(h_gat, edge_index))

        # Global PointNet
        phi = self.point_phi(x)
        if batch is not None:
            rho = global_mean_pool(phi, batch)
            rho_expanded = rho[batch]
        else:
            rho = phi.mean(dim=0, keepdim=True)
            rho_expanded = rho.expand(phi.size(0), -1)

        # Multi-scale fusion
        h_fused = torch.cat([h_gat, phi, rho_expanded], dim=-1)
        return h_fused, rho

    def forward(self, x, edge_index, batch=None):
        h_fused, rho = self.encode(x, edge_index, batch)
        recon_x = self.decoder(h_fused)
        pred_met = self.met_head(rho)
        return recon_x, pred_met


########################################
# MODEL FACTORY
########################################

def create_model(config):
    """
    Model factory returning an instantiated nn.Module matching any experiment configuration.
    Supports:
      - 'baseline_mlp': BaselineMLP (tabular MLP autoencoder)
      - 'gat_autoencoder': NodeGNN (GAT autoencoder with optional MET head)
      - 'node_wise_ae' / 'nodewise_ae': NodeWiseAE (independent per-particle autoencoder)
      - 'deepsets_pointnet' / 'pointnet': DeepSetsPointNet (permutation-invariant point cloud)
      - 'pointnet_gat_hybrid' / 'hybrid_pointnet_gat': PointNetGATHybrid (fused relational + point cloud)
      - 'score_decorrelated_hybrid': ScoreDecorrelatedNodeGNN wrapping PointNetGATHybrid
      - 'adversarial_gat': AdversarialNodeGNN (GAT autoencoder with GRL adversary head)
      - 'score_decorrelated_gat': ScoreDecorrelatedNodeGNN (GAT with score-level DisCo/GRL)
    """
    if hasattr(config, "model"):
        m_cfg = config.model
        arch = m_cfg.architecture_type
        input_dim = m_cfg.input_dim
        hidden_dim = m_cfg.hidden_dim
        num_heads = m_cfg.num_heads if hasattr(m_cfg, "num_heads") else 4
    elif isinstance(config, dict):
        m_cfg = config.get("model", {})
        arch = m_cfg.get("architecture_type", "gat_autoencoder")
        input_dim = m_cfg.get("input_dim", 3)
        hidden_dim = m_cfg.get("hidden_dim", 32)
        num_heads = m_cfg.get("num_heads", 4)
    else:
        raise ValueError(f"Unknown config format: {type(config)}")

    if arch == "baseline_mlp":
        return BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=8)
    elif arch == "gat_autoencoder":
        return NodeGNN(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads)
    elif arch in ("node_wise_ae", "nodewise_ae"):
        return NodeWiseAE(input_dim=input_dim, hidden_dim=hidden_dim)
    elif arch in ("deepsets_pointnet", "pointnet"):
        return DeepSetsPointNet(input_dim=input_dim, hidden_dim=hidden_dim)
    elif arch in ("pointnet_gat_hybrid", "hybrid_pointnet_gat"):
        return PointNetGATHybrid(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads)
    elif arch in ("score_decorrelated_hybrid", "hybrid_decorrelated"):
        from src.adversarial_unlearning_ae import ScoreDecorrelatedNodeGNN
        decorr_type = getattr(m_cfg, "decorrelation_type", "disco") if hasattr(config, "model") else m_cfg.get("decorrelation_type", "disco")
        score_type = getattr(m_cfg, "score_type", "weighted") if hasattr(config, "model") else m_cfg.get("score_type", "weighted")
        base = PointNetGATHybrid(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        return ScoreDecorrelatedNodeGNN(base, decorrelation_type=decorr_type, hidden_dim=hidden_dim, score_type=score_type)
    elif arch == "adversarial_gat":
        from src.adversarial_unlearning_ae import AdversarialNodeGNN
        return AdversarialNodeGNN(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads)
    elif arch == "score_decorrelated_gat":
        from src.adversarial_unlearning_ae import ScoreDecorrelatedNodeGNN
        decorr_type = getattr(m_cfg, "decorrelation_type", "grl") if hasattr(config, "model") else m_cfg.get("decorrelation_type", "grl")
        score_type = getattr(m_cfg, "score_type", "pt_only") if hasattr(config, "model") else m_cfg.get("score_type", "pt_only")
        base = NodeGNN(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        return ScoreDecorrelatedNodeGNN(base, decorrelation_type=decorr_type, hidden_dim=hidden_dim, score_type=score_type)
    else:
        raise ValueError(f"Unsupported architecture_type: {arch}")


########################################
# RUN SAMPLE TEST
########################################

if __name__ == "__main__":

    device = DEVICE
    print(f"Using device: {device}")

    model = EventGNN(input_dim=3).to(device)

    print("Model initialized")


    for df_chunk in stream_lhco_events():

        dataset = []

        for i in range(len(df_chunk)):

            event = df_chunk.iloc[i].values

            graph = event_to_graph(event)

            dataset.append(graph)

        loader = DataLoader(dataset, batch_size=8)

        for batch in loader:

            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)

            print("Output shape:", out.shape)

        break


    print("Sample pipeline run completed.")



