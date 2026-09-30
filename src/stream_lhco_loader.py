import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import tables


########################################
# CONFIG
########################################

DATA_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"

MAX_PARTICLES = 700
DELTA_R_THRESHOLD = 0.4


########################################
# STREAM DATASET
########################################

def stream_lhco_events(data_path=DATA_PATH, chunk_size=100):

    h5file = tables.open_file(data_path, mode='r')

    try:
        node = h5file.get_node("/df/block0_values")
        total_rows = node.nrows

        print("Total events in dataset:", total_rows)

        for start in range(0, total_rows, chunk_size):

            stop = min(start + chunk_size, total_rows)

            chunk = node[start:stop]

            yield pd.DataFrame(chunk)

    finally:
        h5file.close()


from src.run_lhco_gnn import (
    extract_particles,
    compute_MET,
    delta_r,
    build_deltaR_edges,
    event_to_graph,
)


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