# Technical Optimization Plan: GPU Acceleration & Pipeline Decoupling

**Document ID:** `reports/gpu_optimization/optimization_plan.md`  
**Target:** Elimination of CPU Starvation & Maximization of NVIDIA GPU Compute Utilization  
**Author:** Antigravity Performance Engineering & Scientific Verification Agent  

---

## 1. Architectural Strategy & Phased Roadmap

To transform the LHC Olympics 2020 GNN training pipeline from a CPU-bound, serialized workflow into a high-performance, asynchronous streaming architecture, the optimization plan is structured into five concrete phases:

```
[ Phase 1: CPU Vectorization ] ---> [ Phase 2: Async DataLoader ] ---> [ Phase 3: Hardware AMP ]
              |                                     |                                   |
              v                                     v                                   v
   28x Faster Extraction                 Overlap Graph & GPU Compute             Tensor Core Speedup
   Optimized Delta-R Edges               Persistent Multi-Workers                Dynamic GradScaler
```

---

## 2. Phase-by-Phase Technical Specifications

### Phase 1: CPU Preprocessing & Graph Construction Vectorization
* **Target File:** `src/run_lhco_gnn.py`
* **Objective:** Eliminate CPython interpreter loops in particle parsing and edge construction.
* **Technical Implementation:**
  1. **Vectorized Particle Extraction:** Replace `for i in range(700):` with NumPy array reshaping and boolean indexing:
     ```python
     data = raw_data[:2100].reshape(-1, 3)
     mask = data[:, 0] > 0
     particles = data[mask]
     ```
  2. **Fast Delta-R Edges:**
     - Replace $\sqrt{(\Delta \eta)^2 + (\Delta \phi)^2} < \text{threshold}$ with $(\Delta \eta)^2 + (\Delta \phi)^2 < \text{threshold}^2$, avoiding millions of square root calculations.
     - Replace allocation of `~np.eye(num_nodes, dtype=bool)` with in-place diagonal infinity masking: `np.fill_diagonal(dr_sq, np.inf)`.
     - Use `np.nonzero(dr_sq < threshold_sq)` directly instead of `np.argwhere` and transposition.
  3. **Particle Reuse in Graph Builder:** Update `event_to_graph` to accept optional `particles=None` parameter, preventing redundant second particle extractions.

### Phase 2: Asynchronous Multi-Worker Prefetching
* **Target Files:** `src/train_experiment.py`, `src/streaming_data_loader.py`
* **Objective:** Decouple CPU graph construction from GPU execution by overlapping batch generation with GPU forward/backward computation.
* **Technical Implementation:**
  1. Configure `PyGDataLoader` and `TorchDataLoader` with:
     - `num_workers = 2` (dynamically defaulted when `device.type == 'cuda'`, or user-configured via `--num-workers`).
     - `persistent_workers = True`: prevents worker processes from being re-spawned at every epoch boundary.
     - `prefetch_factor = 2`: maintains a constant buffer of 2 pre-constructed batches per worker, guaranteeing the GPU never starves.
  2. Enable CLI override: `--num-workers <int>`.

### Phase 3: Asynchronous Host-to-Device Memory Transfers
* **Target Files:** `src/train_experiment.py`, `src/streaming_data_loader.py`
* **Objective:** Eliminate CPU stalls during PCIe DMA memory transfers.
* **Technical Implementation:**
  1. Configure `pin_memory = True` in data loaders when CUDA is active. Pinned host memory allocates page-locked memory in RAM, allowing the GPU DMA engine to copy memory directly without host OS paging.
  2. Use non-blocking tensor transfers:
     ```python
     batch = batch.to(device, non_blocking=True)
     ```
     This allows the CPU to queue the DMA transfer and immediately begin preparing the next batch without waiting for the PCIe copy to complete.

### Phase 4: Automatic Mixed Precision (AMP) with Gradient Scaling
* **Target File:** `src/train_experiment.py`
* **Objective:** Exploit NVIDIA Tensor Cores on Turing (Tesla T4) and Ampere/Hopper architectures for accelerated 16-bit floating point matrix multiplication while preventing numerical gradient underflow.
* **Technical Implementation:**
  1. Initialize modern gradient scaler:
     ```python
     try:
         scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
     except Exception:
         scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
     ```
  2. Wrap forward pass and loss computation in `torch.amp.autocast(device_type="cuda", enabled=use_amp)`.
  3. Scale loss, unscale gradients prior to gradient clipping (`clip_grad_norm_`), and step optimizer via scaler:
     ```python
     if use_amp and scaler is not None:
         scaler.scale(loss).backward()
         scaler.unscale_(optimizer)
         total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
         scaler.step(optimizer)
         scaler.update()
     ```

### Phase 5: Standalone Diagnostic Tooling & Kaggle Notebook Integration
* **Target Files:** `scripts/diagnose_gpu_pipeline.py`, `notebooks/phase3_kaggle.ipynb`, `scripts/setup_kaggle.sh`
* **Objective:** Give researchers and Kaggle users an automated tool to diagnose, benchmark, and verify their GPU pipeline in one command.
* **Technical Implementation:**
  1. Create `scripts/diagnose_gpu_pipeline.py` offering hardware audit, component profiling, duty cycle calculation, and scientific equivalence verification.
  2. Embed diagnostic cell in `notebooks/phase3_kaggle.ipynb`.

---

## 3. Scientific Invariance Mandate & Constraints

Every code modification must strictly uphold the following scientific invariances:
1. **Model Parameter Budget:** Exactly 7,957 parameters for `PointNetGATHybrid`.
2. **Graph Neighborhood Definition:** Edges exist if and only if $\Delta R = \sqrt{(\Delta \eta)^2 + (\Delta \phi)^2} < 0.4$, excluding self-loops.
3. **Azimuthal Coordinate Wrapping:** $\Delta \phi$ is periodically wrapped on $[0, \pi]$ using minimum circular arc distance.
4. **Kinematic Feature Scaling:** $p_T \to \ln(1 + p_T)$, preserving dynamic range.
5. **Loss Objectives:**
   - Riemannian periodic reconstruction loss on $(\ln(1+p_T), \eta, \phi)$.
   - Missing Transverse Energy (MET) regression target: $(\text{MET}_x, \text{MET}_y)$.
   - Invariant mass distance correlation (DisCo) penalty with stable unbiased matrix centering.
