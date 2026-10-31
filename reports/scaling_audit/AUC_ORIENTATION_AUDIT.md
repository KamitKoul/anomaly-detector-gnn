# ROC-AUC Orientation, Score Distributions & Physics Validity Audit

**Target Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Audit Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Auditor:** Independent Scientific Audit Team  
**Date:** October 31, 2026  
**Status:** **INVESTIGATION COMPLETE — CAUSE OF AUC ~0.49 RESOLVED**  

---

## 1. Executive Summary

All scaling reports cited a matched-window ROC-AUC of **$0.4856 \pm 0.0349$** across all tiers, concluding:
> *"The PointNet–GAT hybrid achieves matched ROC-AUC values hovering near 0.49, statistically consistent with random guessing... establishing that unsupervised particle-level autoencoding with a compact 7,957-parameter model quickly learns to reconstruct typical QCD dijets and multi-pronged W' decay topologies with identical reconstruction error."*

This audit forensically investigated whether:
1. The score was inverted (i.e., lower error was treated as more anomalous, flipping an AUC of 0.5144 to 0.4856).
2. The class labels were inverted (e.g., Background=1, Signal=0).
3. The model genuinely failed to discriminate $W' \to XY$ from QCD dijets.
4. An implementation or data substitution artifact generated the result.

### Core Audit Verdict:
**THE REPORTED AUC IS AN ARTIFACT OF SYNTHETIC UNIFORM NOISE EVALUATION.**  
The score calculation, class labels, and metric orientation in `evaluation/evaluate_scaling_study.py` are mathematically correct. However, because the evaluation harness was executed with `--synthetic`, both background and signal events were drawn from **identical independent uniform random distributions** (`np.random.uniform(-3, 3)`). The reported $\text{AUC} \approx 0.4856$ is simply the finite-sample empirical AUC of random noise evaluated against random noise.

---

## 2. Verification of Anomaly Score Orientation & ROC Implementation

### 2.1 Class Label Assignment:
From [`evaluation/evaluate_scaling_study.py:231`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L231):
```python
    n_bg = len(bg_events)
    n_sig = len(sig_events)
    y_all = np.array([0] * n_bg + [1] * n_sig)
```
- **Background Events:** Label = `0` (Negative class).
- **Signal Events ($W'$):** Label = `1` (Positive / Anomaly class).
- **Label Standard:** Fully conforms to HEP anomaly detection conventions.

### 2.2 Score Definition & Directionality:
From [`evaluation/evaluate_scaling_study.py:172-179`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L172-L179):
```python
    err_pt = (recon_x[:, 0] - batched.x[:, 0]).pow(2)
    err_eta = (recon_x[:, 1] - batched.x[:, 1]).pow(2)
    dphi = torch.abs(recon_x[:, 2] - batched.x[:, 2])
    err_phi = torch.min(dphi, 2.0 * np.pi - dphi).pow(2)
    err_ang = 0.5 * (err_eta + err_phi)
    err_w = 0.8 * err_pt + 0.2 * err_ang
```
- **Reconstruction MSE:** Evaluated constituent-wise.
- **Hypothesis:** BSM resonant signals ($W' \to XY$) contain atypical multi-pronged substructures not present in QCD dijet training data, resulting in higher reconstruction error.
- **Score Assignment:** Higher error $\implies$ higher anomaly score $\implies$ higher probability of being Signal (Label 1).
- **Score Inversion Check:** The score direction is correct. Lower reconstruction error is not treated as signal.

### 2.3 ROC-AUC Function:
From [`evaluation/evaluate_scaling_study.py:46`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L46):
```python
    auc = roc_auc_score(y_true[boot_idx], scores[boot_idx])
```
`sklearn.metrics.roc_auc_score` assumes higher score values predict the positive class (`y_true == 1`). This matches the score definition.

---

## 3. Forensic Trace: Why Did AUC Equal 0.4856?

In [`evaluation/evaluate_scaling_study.py:321-331`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L321-L331):
```python
    if args.synthetic:
        np.random.seed(42)
        n_bg = min(args.limit_bg, 500)
        n_sig = min(args.limit_sig, 250)
        bg_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_bg)]
        sig_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_sig)]
        bg_raw_m = np.random.uniform(2500, 4500, n_bg)
        sig_raw_m = np.random.normal(3500, 80, n_sig)
```

### Mathematical Mechanism:
1. `bg_events` and `sig_events` were drawn from the **exact same generator**: `np.random.uniform(-3, 3, 2101)`.
2. There was zero kinematic, substructure, or clustering difference between background and signal graphs.
3. The neural network decoder reconstructed both sets with statistically identical mean squared error distributions:
   - Background Mean Reconstruction Loss: $\mu_{\text{bg}} = 1.748 \pm 0.012$
   - Signal Mean Reconstruction Loss: $\mu_{\text{sig}} = 1.747 \pm 0.017$
4. In the matched mass window ($m_{jj} \in [3300, 3700]$ GeV):
   - Background events in window: $\approx 100$ events (uniform over $[2500, 4500]$ GeV).
   - Signal events in window: $\approx 245$ events (Gaussian $\mu=3500, \sigma=80$).
5. Evaluating two identical distributions on a finite sample size yields an empirical AUC centered around $0.50$ with standard error $\sigma \approx \frac{1}{\sqrt{12 \cdot N_{\text{eff}}}} \approx 0.035$.
6. For seed 42, the empirical realization was **0.48555** (rounded in reports to **0.4856**).

---

## 4. Scientific Consequences of the Finding

1. **The "Capacity Saturation" Hypothesis was Based on False Evidence:**
   The claim in [`FINAL_SCALING_REPORT.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/FINAL_SCALING_REPORT.md) that *"the PointNet–GAT architecture achieves matched ROC-AUC values hovering near 0.49 because the 7,957-parameter model learns low-order QCD kinematics and cannot separate multi-pronged W' decays"* is an unsubstantiated narrative built to explain a synthetic mock artifact.
2. **Actual PointNet–GAT Performance on Real LHC Olympics Data Remains Unknown for this Matrix:**
   Because real HDF5 test events were never evaluated by the scaling harness, the true discrimination performance of the PointNet–GAT model under this protocol remains to be measured on Kaggle GPU.

---

## 5. Audit Verdict on AUC Orientation
* **Score Implementation:** **CORRECT.** Label 1 = Signal, higher score = higher error.
* **Reported Metric Validity:** **INVALID.** The reported 0.4856 AUC is a measurement of synthetic uniform noise, not LHC physics.
