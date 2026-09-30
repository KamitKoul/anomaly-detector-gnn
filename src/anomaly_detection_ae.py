import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

sys.path.append(os.getcwd())

from src.run_lhco_gnn import stream_lhco_events, event_to_graph, NodeGNN, DEVICE

class LHCODataset(Dataset):
    def __init__(self, data_path, limit=5000):
        self.data_path = data_path
        self.limit = limit
        self.events = []
        
        print(f"Loading {limit} events from {data_path} into memory (raw)...")
        count = 0
        for df_chunk in stream_lhco_events(data_path):
            for i in range(len(df_chunk)):
                # Store raw event to save memory (graphs are bulky)
                self.events.append(df_chunk.iloc[i].values)
                count += 1
                if count >= limit: break
            if count >= limit: break
            
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        # Convert to graph on-the-fly to keep RAM low
        return event_to_graph(self.events[idx])

def run_research_training():
    device = DEVICE
    print(f"--- RESEARCH TRAINING ACTIVE ON: {device} ---")
    
    model = NodeGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    criterion_recon = nn.MSELoss()
    criterion_met = nn.MSELoss()
    
    bg_path = "data/events_LHCO2020_backgroundMC_Pythia.h5"
    sig_path = "data/W_prime_signal.h5"

    # 1. Load Training Data (Only Background)
    train_dataset = LHCODataset(bg_path, limit=5000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. Training Loop
    print("Starting Multi-Task Discovery Training...")
    model.train()
    num_epochs = 20 # Reduced epochs for faster feedback, adjust as needed
    
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass - DataLoader handles 'batch' correctly
            recon_x, pred_met = model(data.x, data.edge_index, data.batch)
            
            # Loss A: Reconstruction
            loss_recon = criterion_recon(recon_x, data.x)
            
            # Loss B: MET Prediction
            loss_met = criterion_met(pred_met, data.y)
            
            loss = loss_recon + 0.1 * loss_met
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Complete | Avg Loss: {total_loss/len(train_loader):.6f}")

    # 3. Save Master Model
    if not os.path.exists("models"): os.makedirs("models")
    torch.save(model.state_dict(), "models/discovery_model.pt")
    print("Master Discovery Model saved to models/discovery_model.pt")

    # 4. Final Scientific Evaluation
    print("\nEvaluating Discovery Power...")
    model.eval()
    bg_scores, sig_scores = [], []
    
    # Eval Dataset
    eval_bg_dataset = LHCODataset(bg_path, limit=500)
    eval_sig_dataset = LHCODataset(sig_path, limit=500)
    
    eval_bg_loader = DataLoader(eval_bg_dataset, batch_size=1, shuffle=False)
    eval_sig_loader = DataLoader(eval_sig_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        print("Scoring Background...")
        for data in tqdm(eval_bg_loader):
            data = data.to(device)
            recon_x, _ = model(data.x, data.edge_index, data.batch)
            bg_scores.append(F.mse_loss(recon_x, data.x).item())
            
        print("Scoring Signal...")
        for data in tqdm(eval_sig_loader):
            data = data.to(device)
            recon_x, _ = model(data.x, data.edge_index, data.batch)
            sig_scores.append(F.mse_loss(recon_x, data.x).item())

    ratio = np.mean(sig_scores)/np.mean(bg_scores)
    print(f"RESEARCH SUCCESS! Discovery Separation: {ratio:.2f}x")
    
    plt.figure(figsize=(10,6))
    plt.hist(bg_scores, bins=50, color='cyan', alpha=0.5, label='Background', density=True)
    plt.hist(sig_scores, bins=50, color='magenta', alpha=0.5, label='W\' Signal', density=True)
    plt.title(f"Node-Level Separation (Ratio: {ratio:.2f}x)")
    plt.xlabel("Anomaly Score (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("anomaly_comparison_plot.png")
    print("Plot saved to anomaly_comparison_plot.png")


if __name__ == "__main__":
    run_research_training()
