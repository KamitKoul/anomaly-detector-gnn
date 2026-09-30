"""
Real-Time LHC Detector Stream Simulation
=========================================
Simulates a live particle detector data stream by loading a trained
NodeGNN model and performing end-to-end inference on streamed collision
events. Measures per-event latency, calculates throughput statistics,
and flags anomalous events that exceed a reconstruction error threshold.

Outputs:
    - Console: live throughput metrics and colored anomaly alerts
    - Plot:   realtime_latency_distribution.png
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.run_lhco_gnn import stream_lhco_events, event_to_graph, NodeGNN, DEVICE


########################################
# CONFIG
########################################

MODEL_PATH = "models/discovery_model.pt"
DATA_PATH = "data/events_LHCO2020_backgroundMC_Pythia.h5"

# Number of events to process in the simulation run
NUM_EVENTS = 1000

# Anomaly score threshold – events above this MSE are flagged
ANOMALY_THRESHOLD = 0.05

# ANSI color codes for terminal alerts
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_GREEN = "\033[92m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


########################################
# LOAD TRAINED MODEL
########################################

def load_model(model_path=MODEL_PATH):
    """
    Loads a trained NodeGNN model from a saved state dict.

    Args:
        model_path: Path to the saved .pt state dict file.

    Returns:
        model: NodeGNN in eval mode on the target DEVICE.
    """
    model = NodeGNN().to(DEVICE)

    if not os.path.exists(model_path):
        print(f"{COLOR_RED}ERROR: Model not found at {model_path}{COLOR_RESET}")
        print("Run anomaly_detection_ae.py first to train and save the model.")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"{COLOR_GREEN}Model loaded from {model_path}{COLOR_RESET}")
    print(f"Device: {DEVICE}")

    return model


########################################
# COMPUTE ANOMALY SCORE
########################################

def compute_anomaly_score(model, graph):
    """
    Runs inference on a single event graph and returns the
    node-level reconstruction MSE as the anomaly score.

    Args:
        model: Trained NodeGNN in eval mode.
        graph: PyG Data object for a single event.

    Returns:
        score: float, mean squared reconstruction error.
        latency: float, inference time in seconds.
    """
    graph = graph.to(DEVICE)

    # Create a batch vector for single-graph inference
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=DEVICE)

    t_start = time.perf_counter()

    with torch.no_grad():
        recon_x, _ = model(graph.x, graph.edge_index, batch)
        score = F.mse_loss(recon_x, graph.x).item()

    t_end = time.perf_counter()
    latency = t_end - t_start

    return score, latency


########################################
# ANOMALY ALERT
########################################

def print_anomaly_alert(event_idx, score):
    """
    Prints a colored terminal alert for a flagged anomalous event.

    Args:
        event_idx: Integer index of the event in the stream.
        score:     Anomaly score (MSE) that exceeded the threshold.
    """
    print(
        f"\n{COLOR_BOLD}{COLOR_RED}"
        f"╔══════════════════════════════════════════════╗\n"
        f"║  ⚠  ANOMALY DETECTED  ⚠                     ║\n"
        f"║  Event #{event_idx:<8d}                          ║\n"
        f"║  Score: {score:<10.6f}  (threshold: {ANOMALY_THRESHOLD})    ║\n"
        f"╚══════════════════════════════════════════════╝"
        f"{COLOR_RESET}\n"
    )


########################################
# LATENCY DISTRIBUTION PLOT
########################################

def plot_latency_distribution(latencies, scores, output_path="realtime_latency_distribution.png"):
    """
    Generates and saves a latency distribution histogram with
    throughput statistics annotated.

    Args:
        latencies: List of per-event latency values (seconds).
        scores:    List of per-event anomaly scores.
        output_path: File path for the saved plot.
    """
    latencies_ms = np.array(latencies) * 1000.0  # Convert to milliseconds

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Latency Distribution ---
    ax1 = axes[0]
    ax1.hist(latencies_ms, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.axvline(np.median(latencies_ms), color="red", linestyle="--", linewidth=1.5,
                label=f"Median: {np.median(latencies_ms):.2f} ms")
    ax1.axvline(np.percentile(latencies_ms, 95), color="orange", linestyle="--", linewidth=1.5,
                label=f"P95: {np.percentile(latencies_ms, 95):.2f} ms")
    ax1.set_xlabel("Inference Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Per-Event Inference Latency Distribution")
    ax1.legend()

    # --- Panel 2: Anomaly Score Timeline ---
    ax2 = axes[1]
    event_indices = np.arange(len(scores))
    colors = ["red" if s > ANOMALY_THRESHOLD else "steelblue" for s in scores]
    ax2.scatter(event_indices, scores, c=colors, s=8, alpha=0.6)
    ax2.axhline(ANOMALY_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                label=f"Threshold: {ANOMALY_THRESHOLD}")
    ax2.set_xlabel("Event Index")
    ax2.set_ylabel("Anomaly Score (MSE)")
    ax2.set_title("Real-Time Anomaly Score Stream")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Latency distribution plot saved to {output_path}")


########################################
# THROUGHPUT REPORT
########################################

def print_throughput_report(latencies, scores, total_wall_time):
    """
    Prints a formatted summary of throughput and anomaly statistics.

    Args:
        latencies:       List of per-event latency values (seconds).
        scores:          List of per-event anomaly scores.
        total_wall_time: Total elapsed wall-clock time (seconds).
    """
    latencies_ms = np.array(latencies) * 1000.0
    scores = np.array(scores)
    num_events = len(latencies)
    num_anomalies = int(np.sum(scores > ANOMALY_THRESHOLD))
    throughput = num_events / total_wall_time if total_wall_time > 0 else 0.0

    print(f"\n{COLOR_BOLD}{'='*50}")
    print(f"  REAL-TIME SIMULATION REPORT")
    print(f"{'='*50}{COLOR_RESET}")
    print(f"  Events Processed:    {num_events}")
    print(f"  Total Wall Time:     {total_wall_time:.2f} s")
    print(f"  Throughput:          {throughput:.1f} events/sec")
    print(f"  {'-'*46}")
    print(f"  Latency (median):    {np.median(latencies_ms):.2f} ms")
    print(f"  Latency (mean):      {np.mean(latencies_ms):.2f} ms")
    print(f"  Latency (P95):       {np.percentile(latencies_ms, 95):.2f} ms")
    print(f"  Latency (P99):       {np.percentile(latencies_ms, 99):.2f} ms")
    print(f"  Latency (max):       {np.max(latencies_ms):.2f} ms")
    print(f"  {'-'*46}")
    print(f"  Anomalies Flagged:   {num_anomalies} / {num_events} "
          f"({100.0 * num_anomalies / num_events:.1f}%)")
    print(f"  Mean Anomaly Score:  {np.mean(scores):.6f}")
    print(f"  Max Anomaly Score:   {np.max(scores):.6f}")
    print(f"{'='*50}\n")


########################################
# MAIN SIMULATION LOOP
########################################

def run_realtime_simulation():
    """
    Main entry point for the real-time detector stream simulation.

    Loads the trained NodeGNN model, streams LHC background events,
    converts each to a graph, performs inference, tracks latency,
    and flags anomalous events with a visual terminal alert.
    """
    print(f"\n{COLOR_BOLD}{COLOR_YELLOW}=== LHC Real-Time Detector Stream Simulation ==={COLOR_RESET}\n")

    # Load trained model
    model = load_model()

    latencies = []
    scores = []
    event_count = 0

    print(f"Streaming {NUM_EVENTS} events from {DATA_PATH}...")
    print(f"Anomaly threshold: {ANOMALY_THRESHOLD}\n")

    wall_start = time.perf_counter()

    pbar = tqdm(total=NUM_EVENTS, desc="Simulating detector stream", unit="evt")

    for df_chunk in stream_lhco_events(DATA_PATH):

        for i in range(len(df_chunk)):

            if event_count >= NUM_EVENTS:
                break

            event = df_chunk.iloc[i].values

            # Convert raw event → PyG graph
            graph = event_to_graph(event)

            # Run inference and measure latency
            score, latency = compute_anomaly_score(model, graph)

            latencies.append(latency)
            scores.append(score)

            # Flag anomalous events
            if score > ANOMALY_THRESHOLD:
                print_anomaly_alert(event_count, score)

            event_count += 1
            pbar.update(1)

            # Live throughput in progress bar
            elapsed = time.perf_counter() - wall_start
            if elapsed > 0:
                pbar.set_postfix({
                    "score": f"{score:.5f}",
                    "lat_ms": f"{latency*1000:.1f}",
                    "evt/s": f"{event_count/elapsed:.1f}"
                })

        if event_count >= NUM_EVENTS:
            break

    pbar.close()
    wall_end = time.perf_counter()
    total_wall_time = wall_end - wall_start

    # Print summary statistics
    print_throughput_report(latencies, scores, total_wall_time)

    # Generate latency distribution plot
    plot_latency_distribution(latencies, scores)

    print(f"{COLOR_GREEN}Simulation complete.{COLOR_RESET}")


########################################
# ENTRY POINT
########################################

if __name__ == "__main__":
    run_realtime_simulation()
