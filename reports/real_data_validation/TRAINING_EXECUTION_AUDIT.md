# Scientific Training Execution & Optimization Audit

**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Model Architecture:** `PointNetGATHybrid` (Frozen, 7,957 parameters)  
**Objective:** Pure Kinematic Reconstruction ($\mathcal{L}_{\text{recon}}$, $\lambda_{\text{MET}}=0.0, \lambda_{\text{adv}}=0.0$, DisCo disabled)  
**Optimizer:** Adam ($\text{lr} = 0.001$, $\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$, $\text{weight\_decay}=0.0$, $\text{clip\_norm}=50.0$)  
**Batching:** Fixed $B=32$ with strict `drop_last = True`  
**Date:** 2026-10-31  

---

## 1. Experiment Mathematical Contracts & Budgets

The data-scaling study investigates two distinct scientific questions governing representation learning in graph neural networks:

### Experiment A: Fixed Epochs (Dataset-Proportional Compute)
*Scientific Question:* "How does anomaly discrimination scale when each collision event is presented a fixed number of times ($E=20$) while the background sample size $D$ expands?"

With mini-batch size $B=32$ and strict `drop_last = True`, the exact number of batches per epoch is $\lfloor D / 32 \rfloor$. The exact optimizer updates $U(D)$ across all tiers are mathematically deterministic:

$$U(D) = E \times \left\lfloor \frac{D}{B} \right\rfloor = 20 \times \left\lfloor \frac{D}{32} \right\rfloor$$

| Tier | Events ($D$) | Batches/Epoch | Dropped Remainder/Epoch | Exact Expected Updates $U(D)$ | Total Event Presentations |
|---|---:|---:|---:|---:|---:|
| **D0** | 5,000 | 156 | 8 | **3,120** | 99,840 |
| **D1** | 25,000 | 781 | 8 | **15,620** | 499,840 |
| **D2** | 50,000 | 1,562 | 16 | **31,240** | 999,680 |
| **D3** | 100,000 | 3,125 | 0 | **62,500** | 2,000,000 |
| **D4** | 250,000 | 7,812 | 16 | **156,240** | 4,999,680 |
| **D5** | 500,000 | 15,625 | 0 | **312,500** | 10,000,000 |
| **D6** | 900,000 | 28,125 | 0 | **562,500** | 18,000,000 |

*Audit Verification:* Previously, synthetic mode ran only 36 updates ($2 \times \lfloor 600/32 \rfloor$) for all tiers. Under real-data execution, D0 executes exactly 156 updates per epoch ($3,120$ updates over 20 epochs).

---

### Experiment B: Fixed Optimizer Updates (Compute-Normalized Scaling)
*Scientific Question:* "How does anomaly discrimination scale when total gradient updates are held strictly constant ($U = 10,000$) while the diversity of the background pool varies from 5k to 900k events?"

Total event presentations are fixed:
$$P = U \times B = 10,000 \times 32 = 320,000 \text{ collision events}$$

Across different tiers $D$, events are recycled cyclically:
$$\text{Mean Event Reuse Frequency } R(D) = \frac{P}{D} = \frac{320,000}{D}$$
$$\text{Dataset Coverage Fraction } C(D) = \min\left(1.0, \frac{320,000}{D}\right)$$

| Tier | Background Pool ($D$) | Target Updates | Total Presentations | Unique Events Observed | Mean Reuse Frequency ($R$) | Pool Coverage ($C$) |
|---|---:|---:|---:|---:|---:|---:|
| **D0** | 5,000 | 10,000 | 320,000 | 5,000 | **64.00x** | 100.0% |
| **D1** | 25,000 | 10,000 | 320,000 | 25,000 | **12.80x** | 100.0% |
| **D2** | 50,000 | 10,000 | 320,000 | 50,000 | **6.40x** | 100.0% |
| **D3** | 100,000 | 10,000 | 320,000 | 100,000 | **3.20x** | 100.0% |
| **D4** | 250,000 | 10,000 | 320,000 | 250,000 | **1.28x** | 100.0% |
| **D5** | 500,000 | 10,000 | 320,000 | 320,000 | **1.00x** | 64.0% |
| **D6** | 900,000 | 10,000 | 320,000 | 320,000 | **1.00x** | 35.6% |

---

## 2. Checkpoint Selection & Scientific Isolation

1. **Unsupervised Metric Exclusivity:**
   - Checkpoints (`checkpoint_best.pt`) are selected strictly by minimum validation kinematic reconstruction loss:
     $$\mathcal{L}_{\text{val}} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \mathcal{L}_{\text{recon}}(x_i, \hat{x}_i)$$
   - Evaluated over the full 50,000 background validation partition (`data/splits/bg_val_50k.npy`).
   - The slicing bug (`val_indices[:10000]`) has been eliminated.
2. **Strict Signal Blindness:**
   - No signal collision events or signal labels exist in the validation loop.
   - Signal evaluation occurs strictly post-training on frozen test partitions.
3. **Dual Checkpoint Preservation:**
   - `checkpoint_best.pt`: Minimum validation reconstruction loss model state.
   - `checkpoint_final.pt`: Terminal training step model state.
   - Each checkpoint payload captures: model weights, optimizer states, GradScaler state, complete Python/NumPy/PyTorch RNG states, dataset manifest SHA-256, and Git commit SHA.
