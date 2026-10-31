# Dataset & Partition Integrity Verification Report

**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Dataset Format:** HDF5 PyTables CArray / Array  
**Audit Verification:** Physical Materialization, Cryptographic Hashes, and Partition Disjointness  
**Date:** 2026-10-31  

---

## 1. Physical Dataset Verification

Both primary datasets have been verified on local storage with allocated physical blocks:

| Dataset Identifier | Physical Path | Allocated Blocks | File Size | Cryptographic SHA-256 Digest | Status |
|---|---|---|---|---|---|
| **LHC Olympics 2020 Background MC Pythia** | `data/events_LHCO2020_backgroundMC_Pythia.h5` | 5,268,904 | 2,697,677,287 B (2.51 GB) | `97edb2ac7a894102a5c8377e489c73ae871d487d114430e917b9f684dea36ed4` | **VERIFIED** |
| **W' $\to$ XY Anomaly Signal** | `data/W_prime_signal.h5` | 3,284,456 | 1,681,638,808 B (1.57 GB) | `624afc85740010fd4e0668ca772784785918891b5bd0e22ca07a4a5ffc7346dc` | **VERIFIED** |

### Internal HDF5 Node Architecture

- **Background File:**
  - Table Node: `/df/block0_values`
  - Dimensions: `(1,000,000, 2100)`
  - Storage Type: `tables.carray.CArray` (chunked, blosc compressed)
  - Data Type: `float64`
  - Total Events: Exactly 1,000,000 collision events (700 particles $\times 3$ features: $p_T, \eta, \phi$)
  - Data Integrity: Zero NaNs, zero Infs across verified rows.

- **Signal File:**
  - Table Node: `/df/block0_values`
  - Dimensions: `(100,000, 2101)`
  - Storage Type: `tables.array.Array`
  - Data Type: `float64`
  - Total Events: Exactly 100,000 signal collision events (last feature: truth bit = 1.0)
  - Data Integrity: Zero NaNs, zero Infs across verified rows.

---

## 2. Collision Physics Diagnostics

Batch-level particle kinematics extracted from the physical background dataset:

| Kinematic Variable | Minimum | Maximum | Mean | Median | Standard Deviation | Physical Integrity Check |
|---|---|---|---|---|---|---|
| **Multiplicity ($N_{\text{particles}}$)** | 55 | 329 | 179.73 | 174.50 | 54.12 | Realistic dijet jet fragmentation |
| **Padding Fraction** | — | — | 74.32% | — | — | Sparsely packed 700-slot tensors |
| **Particle $p_T$ [GeV]** | 0.10 | 1367.25 | 16.20 | 3.05 | 54.79 | Heavy-tailed transverse momentum spectrum |
| **Pseudorapidity ($\eta$)** | -5.00 | 4.99 | 0.04 | 0.06 | 1.49 | Symmetric barrel/endcap coverage |
| **Azimuthal Angle ($\phi$)** | -3.14 | 3.14 | 0.04 | 0.00 | 1.80 | Uniform cylinder geometry |

---

## 3. Data Partition Hierarchy & Leakage Verification

All 12 partition `.npy` files in `data/splits/` were verified against the 1,000,000 background and 100,000 signal event pools:

### Background Partitions (Total Pool = 1,000,000 events)

| Partition File | Exact Event Count | SHA-256 Digest | Disjointness from Val/Test | Hierarchy Guarantee |
|---|---|---|---|---|
| `bg_train_d0_5k.npy` | 5,000 | `c232c45b77b40129ee0ab21f4d08acb474dd8ddab2f013ee3e0912c9a28aaa87` | 0 overlap | Base tier ($D_0$) |
| `bg_train_d1_25k.npy` | 25,000 | `2c075d3c25a229cf73b13103509cff13abd0cd58d6fee4f59524980186a5ada2` | 0 overlap | $D_0 \subset D_1$ |
| `bg_train_d2_50k.npy` | 50,000 | `a0643ab9b4d6b96cf3b285b40c7f49c34a4b8071d27c1519784151ba7b834015` | 0 overlap | $D_1 \subset D_2$ |
| `bg_train_d3_100k.npy` | 100,000 | `e43b37a77f6ca99ac1219b0d10812d5d5344a7c7a9c34f72823f7f4a5e7200f7` | 0 overlap | $D_2 \subset D_3$ |
| `bg_train_d4_250k.npy` | 250,000 | `c8ee32b54870660c6f9a1462e839ba3027fcd727c994cb5c61fa69c08a799b8a` | 0 overlap | $D_3 \subset D_4$ |
| `bg_train_d5_500k.npy` | 500,000 | `59a45640c78f2cae98faf4fbce710d78ee173001ad755dde7b724a781805b1c9` | 0 overlap | $D_4 \subset D_5$ |
| `bg_train_d6_900k.npy` | 900,000 | `d67b360862889cd17800f2b3ae1bd38085378dbfdde2d5b6690d13a62c296f20` | 0 overlap | $D_5 \subset D_6 = \text{Full Train Pool}$ |
| `bg_val_50k.npy` | 50,000 | `1e72579a39ebdf1228cf13f8f0e653935d77e188bc6501af39054588395daa74` | 0 overlap with Train/Test | Frozen Validation Set |
| `bg_test_50k.npy` | 50,000 | `a5b6a4cc115758acf677713b54f6b809681873b4b855b567d090fc5f41ffeeb9` | 0 overlap with Train/Val | Frozen Test Set |

**Integrity Verification:**
$$\text{Pool} = \text{Train}_{900\text{k}} \cup \text{Val}_{50\text{k}} \cup \text{Test}_{50\text{k}} = 1,000,000$$
$$\text{Train}_{900\text{k}} \cap \text{Val}_{50\text{k}} = \emptyset, \quad \text{Train}_{900\text{k}} \cap \text{Test}_{50\text{k}} = \emptyset, \quad \text{Val}_{50\text{k}} \cap \text{Test}_{50\text{k}} = \emptyset$$
Strict nested ladder verified: $D_0 \subset D_1 \subset D_2 \subset D_3 \subset D_4 \subset D_5 \subset D_6$.

### Signal Partitions (Total Pool = 100,000 events)

| Partition File | Exact Event Count | SHA-256 Digest | Disjointness |
|---|---|---|---|
| `sig_val_25k.npy` | 25,000 | `f8017f23f4b2b4adfb5b0fd1ffab269722c599485267a3343aebc9b67f37b78d` | 0 overlap with Test/Reserve |
| `sig_test_25k.npy` | 25,000 | `1bb9f10dd56fe207f0f408a95a6b56845e1d77cce7820396796e8a02b4ff0e07` | 0 overlap with Val/Reserve |
| `sig_reserve_50k.npy` | 50,000 | `98c62ddd32051a65e5d6541d354099ef5ba284bbefabb628b276f35d0d11893e` | 0 overlap with Val/Test |

**Signal Containment Guarantee:** Zero signal events are indexed in background partitions. Unsupervised background training is strictly uncontaminated.
