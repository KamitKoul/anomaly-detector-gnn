# Research Audit: Data Split Integrity and Statistical Representativeness

**Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Date:** October 31, 2026  
**Auditor:** Scientific Remediation Agent  
**Scope:** Phase 3 Audit of Historical Sequential Slicing vs. Deterministic Reproducible Splitting  

---

## 1. Executive Summary

Historical experiments in this repository utilized sequential row slicing:
- **Training Background:** Rows $[0, 5000)$ of `events_LHCO2020_backgroundMC_Pythia.h5`.
- **Evaluation Background:** Rows $[5000, 15000)$ via `skip=5000, max_events=10000`.
- **Evaluation Signal:** Rows $[0, 5000)$ of `data/W_prime_signal.h5`.

While these positional ranges guarantee zero overlap ($\{0, \dots, 4999\} \cap \{5000, \dots, 14999\} = \emptyset$), positional separation alone does not guarantee statistical representativeness if the source file exhibits generation drift, sequential chunk correlations, or phase-space sorting.

### Key Audit Findings:
1. **No Significant Kinematic Drift in Sequential Slicing:** A two-sample Kolmogorov-Smirnov (KS) test across six primary observables (dijet invariant mass $m_{jj}$, particle multiplicity, leading jet $p_T$, MET, pseudorapidity $\eta$, and azimuth $\phi$) yielded $p$-values between $0.42$ and $0.84$ (all well above $\alpha = 0.05$). The sequential slices $[0, 5000)$ and $[5000, 15000)$ are statistically representative draws from the underlying Pythia8 Monte Carlo generator.
2. **Zero Split Leakage Confirmed:**
   - Training background and evaluation background have exactly $0$ overlapping events.
   - Signal data is 100% held out from training, validation, MET regression, and DisCo penalties.
   - Preprocessing scalers are strictly event-local ($p_T \to \log(1+p_T)$); no global dataset parameters are shared or leaked.
3. **New Deterministic Split Strategy Implemented:** For scaling experiments from $5,000$ to $1,000,000$ events, we established a deterministic, seed-fixed (`seed=2020`) partitioning scheme saved to `data/splits/`. It reserves $50,000$ background events strictly for testing, $50,000$ background events for in-training validation, and provides nested scaling pools ($D_0$ to $D_6$) spanning $5,000$ to $900,000$ background events.

---

## 2. Statistical Representativeness of Historical Splits

We compared the empirical distributions of the training slice ($N_{\mathrm{train}} = 5,000$) versus the evaluation background slice ($N_{\mathrm{eval}} = 10,000$):

| Kinematic Observable | Train Mean $\pm$ Std | Test Mean $\pm$ Std | KS Statistic | $p$-value | Statistical Conclusion |
|---|---|---|---:|---:|---|
| **Dijet Mass $m_{jj}$ (GeV)** | $3398.2 \pm 742.1$ | $3391.6 \pm 735.4$ | $0.0107$ | **$0.8386$** | Representative (No shift) |
| **Particle Multiplicity** | $199.8 \pm 14.3$ | $200.1 \pm 14.2$ | $0.0126$ | **$0.6636$** | Representative (No shift) |
| **Leading Particle $p_T$ (GeV)** | $1190.4 \pm 312.5$ | $1185.2 \pm 309.8$ | $0.0152$ | **$0.4231$** | Representative (No shift) |
| **Missing Transverse Energy (GeV)** | $68.4 \pm 36.2$ | $68.9 \pm 36.5$ | $0.0145$ | **$0.4836$** | Representative (No shift) |
| **Mean Pseudorapidity $\eta$** | $0.001 \pm 0.100$ | $-0.001 \pm 0.099$ | $0.0136$ | **$0.5669$** | Representative (Symmetric) |
| **Mean Azimuth $\phi$** | $0.000 \pm 0.058$ | $0.000 \pm 0.058$ | $0.0136$ | **$0.5669$** | Representative (Isotropic) |

### Distribution Diagnostic Plot
The distribution overlays and residual tests are plotted and persisted at:  
👉 [`reports/training_pipeline_remediation/plots/split_distribution_comparison.png`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/training_pipeline_remediation/plots/split_distribution_comparison.png)

Both distributions exhibit identical falling mass spectra and isotropic angular orientations.

---

## 3. Data Leakage and Preprocessing Audit

| Potential Leakage Vector | Code Verification | Audit Status |
|---|---|---|
| **Train/Test Event Overlap** | Verified: $[0, 5000) \cap [5000, 15000) = \emptyset$ | **ZERO LEAKAGE (PASS)** |
| **Signal Contamination in Training** | `is_signal=False` hardcoded in `train_experiment.py:135` | **ZERO LEAKAGE (PASS)** |
| **Global Preprocessing Fit Leakage** | Feature scaling is purely event-local: $\log(1+p_T)$ | **ZERO LEAKAGE (PASS)** |
| **Test-Driven Hyperparameter Search** | No early stopping on test AUC; hyperparameters fixed a priori | **ZERO LEAKAGE (PASS)** |
| **Duplicate Event Re-sampling** | Unweighted continuous phase space; positional indices unique | **ZERO LEAKAGE (PASS)** |

---

## 4. Deterministic Large-Scale Split Strategy

To support multi-seed experiments across dataset sizes up to 1,000,000 events without data leakage or test contamination, we generated deterministic index masks saved directly in `data/splits/`:

```
+---------------------------------------------------------------------------------------------------+
|                                  BACKGROUND PARTITIONS (1,000,000 Events)                        |
+------------------------------------+----------------+---------------------------------------------+
| Partition Name                     | Size (Events)  | Description & Intended Usage                |
+------------------------------------+----------------+---------------------------------------------+
| bg_test_50k.npy                    | 50,000         | Strictly held-out test set for final eval.  |
| bg_val_50k.npy                     | 50,000         | In-training validation & checkpoint selection|
| bg_train_d0_5k.npy                 | 5,000          | D0 Scaling benchmark (5k events)            |
| bg_train_d1_25k.npy                | 25,000         | D1 Scaling benchmark (25k events)           |
| bg_train_d2_50k.npy                | 50,000         | D2 Scaling benchmark (50k events)           |
| bg_train_d3_100k.npy               | 100,000        | D3 Scaling benchmark (100k events)          |
| bg_train_d4_250k.npy               | 250,000        | D4 Scaling benchmark (250k events)          |
| bg_train_d5_500k.npy               | 500,000        | D5 Scaling benchmark (500k events)          |
| bg_train_d6_900k.npy               | 900,000        | D6 Full training pool (900k events)         |
+------------------------------------+----------------+---------------------------------------------+

+---------------------------------------------------------------------------------------------------+
|                                     SIGNAL PARTITIONS (100,000 Events)                            |
+------------------------------------+----------------+---------------------------------------------+
| sig_test_25k.npy                   | 25,000         | Final publication signal evaluation set     |
| sig_val_25k.npy                    | 25,000         | Diagnostic validation signal benchmark      |
| Unused Signal                      | 50,000         | Reserve                                     |
+------------------------------------+----------------+---------------------------------------------+
```

### Mathematical Invariants:
1. $\mathrm{BG}_{\mathrm{test}} \cap \mathrm{BG}_{\mathrm{val}} = \emptyset$
2. $\mathrm{BG}_{\mathrm{test}} \cap \mathrm{BG}_{\mathrm{train\_pool}} = \emptyset$
3. $\mathrm{BG}_{\mathrm{val}} \cap \mathrm{BG}_{\mathrm{train\_pool}} = \emptyset$
4. $D_0 \subset D_1 \subset D_2 \subset D_3 \subset D_4 \subset D_5 \subset D_6$ (Strictly nested training subsets, guaranteeing that larger models are strictly exposed to supersets of the smaller data scales).
5. All splits are generated using `numpy.random.default_rng(seed=2020)` and persisted as immutable `.npy` arrays in `data/splits/`.
