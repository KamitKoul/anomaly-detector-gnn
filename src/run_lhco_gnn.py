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

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

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


########################################
# BUILD ΔR GRAPH
########################################

def build_deltaR_edges(particles):

    edges = []

    num_nodes = len(particles)

    for i in range(num_nodes):
        for j in range(num_nodes):

            if i == j:
                continue

            eta1, phi1 = particles[i][1], particles[i][2]
            eta2, phi2 = particles[j][1], particles[j][2]

            dr = delta_r(eta1, phi1, eta2, phi2)

            if dr < DELTA_R_THRESHOLD:
                edges.append([i, j])

    if len(edges) == 0:
        edges.append([0,0])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


########################################
# EVENT → GRAPH
########################################

def event_to_graph(event):

    particles = extract_particles(event)

    MET, MET_x, MET_y = compute_MET(particles)

    MET_node = np.array([[MET,0,0]])

    particles = np.vstack([particles, MET_node])

    x = torch.tensor(particles, dtype=torch.float)

    edge_index = build_deltaR_edges(particles)

    # Attach MET vector as global target for the physics prediction head
    y = torch.tensor([MET_x, MET_y], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data


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
      Encoder: GATConv(3→32, 4 heads) → ELU → GATConv(128→32, 1 head) → ELU
      Decoder Head 1: Node Reconstruction (32→16→3)
      Decoder Head 2: MET Prediction via global_mean_pool (32→16→2)
    """

    def __init__(self, input_dim=3, hidden_dim=32, num_heads=4):
        super().__init__()

        # Encoder: Graph Attention Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)

        # Decoder Head 1: Node-Level Reconstruction
        self.recon_head = nn.Sequential(
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

    def forward(self, x, edge_index, batch):

        # Encode
        z = F.elu(self.gat1(x, edge_index))
        z = F.elu(self.gat2(z, edge_index))

        # Decode: Reconstruct node features
        recon_x = self.recon_head(z)

        # Decode: Predict global MET
        global_z = global_mean_pool(z, batch)
        pred_met = self.met_head(global_z)

        return recon_x, pred_met


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



