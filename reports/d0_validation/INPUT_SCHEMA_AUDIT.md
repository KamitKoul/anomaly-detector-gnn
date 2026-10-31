# Input Schema & Feature Mapping Audit: Background vs. Signal Datasets

**Date:** 2026-10-31  
**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Datasets:**
- Background: `LHC Olympics 2020 Background MC Pythia.h5` (SHA-256: `97edb2ac7a894102...`)
- Signal: `data/W_prime_signal.h5` (SHA-256: `624afc85740010fd...`)
**Status:** **AUDITED & EQUIVALENT**

---

## 1. HDF5 Storage Architecture & Column Analysis

Both datasets were created according to the official LHC Olympics 2020 R&D challenge specifications (Kasieczka et al.):

| Property | Background Dataset (`events_LHCO2020_backgroundMC_Pythia.h5`) | Signal Dataset (`W_prime_signal.h5`) | Equivalence Status |
|---|---|---|---|
| **HDF5 Group / Node** | `/df/block0_values` | `/df/block0_values` | Identical Pandas HDF5 format |
| **Storage Structure** | `tables.carray.CArray` (chunked) | `tables.array.Array` | Compatible NumPy-backed array |
| **Row Dimension (Events)** | 1,000,000 | 100,000 | Distinct physics populations |
| **Column Dimension** | **2,100** | **2,101** | Column 2100 is truth label bit |
| **Data Type** | `float64` | `float64` | Identical precision |
| **Particle Slots** | 700 particles | 700 particles | Identical |
| **Features per Particle** | 3 ($p_T, \eta, \phi$) | 3 ($p_T, \eta, \phi$) | Identical |

---

## 2. Feature Mapping & The 2,101st Column

Inspection of the raw arrays confirms:
- In the background dataset, columns 0 through 2099 represent 700 particle triplets $(p_T, \eta, \phi)$. There is no 2101st column.
- In the signal dataset, columns 0 through 2099 represent the exact same 700 particle triplets $(p_T, \eta, \phi)$. Column index 2100 (the 2101st value) contains the benchmark ground-truth label bit: `1.0` across all signal rows.

### Column Mapping Table

| Column Index Range | Particle Slot | Feature Meaning | Background Representation | Signal Representation | Model Input? |
|---|---|---|---|---|---|
| `[0]` | Particle 0 | Transverse momentum ($p_T$) [GeV] | $p_T \ge 0.0$ | $p_T \ge 0.0$ | **YES** (transformed via $\log(1+p_T)$) |
| `[1]` | Particle 0 | Pseudorapidity ($\eta$) | $\eta \in [-5, 5]$ | $\eta \in [-5, 5]$ | **YES** (untransformed) |
| `[2]` | Particle 0 | Azimuthal angle ($\phi$) [rad] | $\phi \in [-\pi, \pi]$ | $\phi \in [-\pi, \pi]$ | **YES** (untransformed) |
| `[3i + 0]` | Particle $i$ ($1 \le i \le 699$) | Transverse momentum ($p_T$) [GeV] | $p_T \ge 0.0$ (or $0.0$ if padded) | $p_T \ge 0.0$ (or $0.0$ if padded) | **YES** (if $p_T > 0$) |
| `[3i + 1]` | Particle $i$ ($1 \le i \le 699$) | Pseudorapidity ($\eta$) | $\eta$ (or $0.0$ if padded) | $\eta$ (or $0.0$ if padded) | **YES** (if $p_T > 0$) |
| `[3i + 2]` | Particle $i$ ($1 \le i \le 699$) | Azimuthal angle ($\phi$) [rad] | $\phi$ (or $0.0$ if padded) | $\phi$ (or $0.0$ if padded) | **YES** (if $p_T > 0$) |
| **`[2100]`** | **None** | **Truth Label Bit** | **NOT PRESENT (Implicit 0.0)** | **`1.0` (Truth Signal Bit)** | **NO (STRICTLY EXCLUDED)** |

---

## 3. Preprocessing Trace & Leakage Verification

### A. Particle Extraction (`src/run_lhco_gnn.py:extract_particles`)
```python
def extract_particles(event):
    if isinstance(event, np.ndarray):
        raw_data = event[:2100]
    elif hasattr(event, "values"):
        raw_data = event.values[:2100]
    else:
        raw_data = np.asarray(event[:2100], dtype=np.float32)

    # Reshape into (700, 3) where columns are [pt, eta, phi]
    data = raw_data.reshape(-1, 3)
    mask = data[:, 0] > 0
    particles = data[mask]
    ...
```
1. **Explicit Slicing `event[:2100]`:** Regardless of whether the row has 2,100 or 2,101 columns, only the first 2,100 columns are selected. Column 2100 is discarded before any computation.
2. **Zero-Padding Removal:** Events are padded to 700 particle slots. Physical particles have $p_T > 0$, while padded slots have $(p_T=0, \eta=0, \phi=0)$. The boolean mask `mask = data[:, 0] > 0` strips all padded slots.
3. **Padded Slot Invariance:** Zero-padded slots do not enter the graph structure or model forward pass.

### B. Graph Construction (`src/run_lhco_gnn.py:event_to_graph`)
For both classes:
1. Visible particles $(N, 3)$ are extracted identically.
2. Missing Transverse Energy ($\text{MET}$) is computed from visible particles:
   $$\vec{p}_{T,\text{total}} = \sum_{i=1}^N \left(p_{T,i}\cos\phi_i, \, p_{T,i}\sin\phi_i\right), \quad \text{MET} = \|\vec{p}_{T,\text{total}}\|, \quad \phi_{\text{MET}} = \text{atan2}(-\sum p_{y,i}, -\sum p_{x,i})$$
3. A global MET node $[\text{MET}, 0.0, \phi_{\text{MET}}]$ is appended, yielding $(N+1, 3)$ particles.
4. Proximity edges are constructed with threshold $\Delta R < 0.4$:
   $$\Delta R = \sqrt{(\eta_i - \eta_j)^2 + (\Delta\phi_{ij})^2}, \quad \Delta\phi_{ij} = \min\left(|\phi_i - \phi_j|, \, 2\pi - |\phi_i - \phi_j|\right)$$
5. Node features: $p_T$ is transformed via $\log(1 + p_T)$, while $\eta$ and $\phi$ are left unscaled. Resulting node tensor has shape $(N+1, 3)$, dtype `torch.float32`.

### C. Information Leakage Guarantee
- Signal labels are strictly isolated to evaluation scoring (computation of ROC-AUC and background rejection).
- No truth label, event ID, or anomalous metadata enters the model input or graph features.
- Model inputs for background and signal collision events are physics-equivalent representations.
