import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


########################################
# CONFIG
########################################

DATA_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"

MAX_PARTICLES = 700
DELTA_R_THRESHOLD = 0.4


########################################
# STREAM DATASET
########################################

def stream_lhco_events(chunk_size=100):

    store = pd.HDFStore(DATA_PATH)

    key = store.keys()[0]

    total_rows = store.get_storer(key).nrows

    print("Total events in dataset:", total_rows)

    for start in range(0, total_rows, chunk_size):

        stop = min(start + chunk_size, total_rows)

        df_chunk = store.select(key, start=start, stop=stop)

        yield df_chunk

    store.close()


########################################
# EXTRACT PARTICLES
########################################

def extract_particles(event):

    particles = []

    for i in range(MAX_PARTICLES):

        pt = event[i * 3]
        eta = event[i * 3 + 1]
        phi = event[i * 3 + 2]

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

    return MET


########################################
# DELTA R CALCULATION
########################################

def delta_r(eta1, phi1, eta2, phi2):

    d_eta = eta1 - eta2
    d_phi = np.abs(phi1 - phi2)

    if d_phi > np.pi:
        d_phi = 2*np.pi - d_phi

    return np.sqrt(d_eta**2 + d_phi**2)


########################################
# BUILD DELTA R EDGES
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

    MET = compute_MET(particles)

    MET_node = np.array([[MET,0,0]])

    particles = np.vstack([particles, MET_node])

    x = torch.tensor(particles, dtype=torch.float)

    edge_index = build_deltaR_edges(particles)

    data = Data(x=x, edge_index=edge_index)

    return data


########################################
# STREAM GRAPHS
########################################

def stream_graph_dataset(chunk_size=100):

    for df_chunk in stream_lhco_events(chunk_size):

        graphs = []

        for i in range(len(df_chunk)):

            event = df_chunk.iloc[i].values

            graph = event_to_graph(event)

            graphs.append(graph)

        yield graphs