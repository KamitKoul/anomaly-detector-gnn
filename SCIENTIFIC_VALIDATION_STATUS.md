# SCIENTIFIC VALIDATION STATUS: Metric Definitions, Implementations, and Issues

**Generated:** October 30, 2026  
**Repository:** `anomaly-detector-gnn`  
**All definitions sourced from actual evaluation script implementations**

---

## 1. Core Metric Implementations

### 1.1 ROC-AUC
**Source:** `evaluation/evaluate_single.py:36-63`, `sklearn.metrics.roc_auc_score`

```python
def compute_roc_metrics(labels, scores):
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    auc = float(roc_auc_score(labels_arr, scores_arr))  # Default: trapezoidal rule
    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    ...
```

- **Implementation:** `sklearn.metrics.roc_auc_score` with default settings
- **Average method:** Not specified (binary classification → single AUC)
- **Uncertainty:** **NOT COMPUTED** — no bootstrap/DeLong confidence intervals
- **Status:** ✅ Implemented, ❌ No uncertainty quantification

---

### 1.2 Background Rejection at Fixed Signal Efficiency
**Source:** `evaluation/evaluate_single.py:47-62`

```python
# Background rejection (1/FPR) at 30% and 50% signal efficiency
idx_30 = np.where(tpr >= 0.30)[0]
if len(idx_30) > 0 and fpr[idx_30[0]] > 0:
    rejection_30 = float(1.0 / fpr[idx_30[0]])

idx_50 = np.where(tpr >= 0.50)[0]
if len(idx_50) > 0 and fpr[idx_50[0]] > 0:
    rejection_50 = float(1.0 / fpr[idx_50[0]])
```

- **Definition:** `Rejection = 1 / FPR` at the threshold where `TPR ≥ target_efficiency`
- **Operating points:** 30% and 50% signal efficiency
- **Edge case:** Returns `-1.0` if FPR=0 (infinite rejection)
- **Uncertainty:** **NOT COMPUTED**
- **Status:** ✅ Implemented

---

### 1.3 Signal Efficiency
**Definition:** True Positive Rate (TPR) at a given threshold
- Computed via `roc_curve()` from sklearn
- Used implicitly for rejection calculation
- **Status:** ✅ Available via ROC curve

---

### 1.4 Background Efficiency
**Definition:** False Positive Rate (FPR) at a given threshold
- `FPR = 1 - Specificity = FP / (FP + TN)`
- Used for rejection calculation: `Rejection = 1/FPR`
- **Status:** ✅ Available via ROC curve

---

### 1.5 Separation Ratio
**Source:** `evaluation/evaluate_single.py:66-86`, `src/ultimate_discovery_proof.py:184-217`

```python
def compute_separation_statistics(bg_scores, sig_scores):
    mu_bg = np.mean(bg_arr)
    sigma_bg = np.std(bg_arr)
    mu_sig = np.mean(sig_arr)
    ratio = mu_sig / mu_bg if mu_bg > 0 else 0.0
    significance = (mu_sig - mu_bg) / sigma_bg
    return {"separation_ratio": ratio, "significance_sigma": significance}
```

- **Definition:** `Ratio = Mean(Signal Scores) / Mean(Background Scores)`
- **Significance:** `(μ_sig - μ_bg) / σ_bg` (Gaussian approximation)
- **Status:** ✅ Implemented

---

## 2. Mass-Related Metrics

### 2.1 Mass-Matched ROC-AUC
**Source:** `evaluation/evaluate_mass_correlation.py:230-269`

```python
# Signal resonance window: [3300, 3700] GeV
win_low, win_high = 3300.0, 3700.0
bg_win_mask = (bg_raw_m >= win_low) & (bg_raw_m <= win_high)
sig_win_mask = (sig_raw_m >= win_low) & (sig_raw_m <= win_high)

# Inclusive ROC
y_all = np.array([0] * len(bg_sc) + [1] * len(sig_sc))
sc_all = np.concatenate([bg_sc, sig_sc])
auc_inclusive = roc_auc_score(y_all, sc_all)

# Mass-Matched ROC
bg_sc_win = bg_sc[bg_win_mask]
sig_sc_win = sig_sc[sig_win_mask]
y_win = np.array([0] * len(bg_sc_win) + [1] * len(sig_sc_win))
sc_win = np.concatenate([bg_sc_win, sig_sc_win])
auc_matched = roc_auc_score(y_win, sc_win)
```

- **Window:** [3300, 3700] GeV (±200 GeV around 3.5 TeV W' resonance)
- **Method:** Subset both background and signal to same mass window, then compute AUC
- **Purpose:** Test if discrimination survives when mass is controlled
- **Status:** ✅ Implemented

---

### 2.2 Micro-Fine Mass-Window ROC-AUC
**Source:** `evaluation/evaluate_controlled_ablation.py:256-258`, `evaluation/investigate_deep_physics.py:202-238`

```python
win_micro_bg = (bg_raw_m >= 3450.0) & (bg_raw_m <= 3550.0)  # ±50 GeV
win_micro_sig = (sig_raw_m >= 3450.0) & (sig_raw_m <= 3550.0)
...
y_micro = np.array([0] * int(np.sum(win_micro_bg)) + [1] * int(np.sum(win_micro_sig)))
s_micro = np.concatenate([bg_s[win_micro_bg], sig_s[win_micro_sig]])
auc_micro = roc_auc_score(y_micro, s_micro)
```

- **Windows tested:** ±200, ±150, ±100, **±50 GeV** around 3.5 TeV
- **100-GeV slices:** 3000-4000 GeV in 100 GeV bins
- **Status:** ✅ Implemented

---

### 2.3 Pearson Correlation (r)
**Source:** `evaluation/evaluate_mass_correlation.py:213-214`, `evaluation/evaluate_controlled_ablation.py:303-304`, `scipy.stats.pearsonr`

```python
r_val, _ = pearsonr(bg_raw_m, bg_s)  # bg_s = anomaly scores
```

- **Definition:** Linear correlation coefficient between anomaly score and invariant mass
- **Sample:** Background events only (10k or 995k depending on eval)
- **Status:** ✅ Implemented

---

### 2.4 Spearman Correlation (ρ)
**Source:** `evaluation/evaluate_mass_correlation.py:214-215`, `scipy.stats.spearmanr`

```python
rho_val, _ = spearmanr(bg_raw_m, bg_s)
```

- **Definition:** Rank correlation (monotonic relationship)
- **Status:** ✅ Implemented

---

### 2.5 Ridge Probe R² (Mass Predictability)
**Source:** `evaluation/evaluate_mass_correlation.py:284-294`

```python
ridge = Ridge(alpha=1.0)
n_train = int(len(z_bar) * 0.8)
ridge.fit(z_bar[:n_train], m_target[:n_train])
r2_latent = ridge.score(z_bar[n_train:], m_target[n_train:])

# Also on anomaly score
ridge_sc = Ridge(alpha=1.0)
ridge_sc.fit(sc[:n_train].reshape(-1, 1), m_target[:n_train])
r2_score = ridge_sc.score(sc[n_train:].reshape(-1, 1), m_target[n_train:])
```

- **Definition:** Coefficient of determination from Ridge regression (α=1.0)
- **Train/test split:** 80/20 temporal (not random — first 80% train, last 20% test)
- **Targets:** 
  - `z_bar` → mass (pooled latent predictability)
  - `score` → mass (anomaly score predictability)
- **Mass normalization:** `m_target = bg_raw_m / 1000.0` (TeV)
- **Status:** ✅ Implemented
- **Note:** No cross-validation; single 80/20 split

---

## 3. Jensen-Shannon Divergence (JSD) & Inverse JSD

### 3.1 JSD Implementation
**Source:** `evaluation/evaluate_mass_sculpting.py:45-57`

```python
def compute_jsd(p_samples: np.ndarray, q_samples: np.ndarray, bins: np.ndarray) -> float:
    from scipy.spatial.distance import jensenshannon
    
    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)
    
    eps = 1e-12
    p_norm = (p_hist + eps) / np.sum(p_hist + eps)
    q_norm = (q_hist + eps) / np.sum(q_hist + eps)
    
    js_dist = jensenshannon(p_norm, q_norm, base=np.e)  # Natural log
    return float(js_dist ** 2)  # JSD = (JS distance)²
```

**Source (with bootstrap):** `evaluation/evaluate_controlled_ablation.py:58-98`

```python
def compute_jsd_smoothness(masses_full, masses_cut, bins=40, range_gev=(2000, 5000), num_bootstraps=500):
    p_counts, edges = np.histogram(masses_full, bins=bins, range=range_gev)
    q_counts, _ = np.histogram(masses_cut, bins=bins, range=range_gev)
    
    p = p_counts.astype(np.float64) / (np.sum(p_counts) + 1e-12)
    q = q_counts.astype(np.float64) / (np.sum(q_counts) + 1e-12)
    
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()
    
    js_div = float(jensenshannon(p, q, base=2.0) ** 2)  # Base-2 log here!
    inv_jsd = 1.0 / js_div if js_div > 1e-9 else 9999.0
    
    # Bootstrap uncertainty
    rng = np.random.RandomState(42)
    boot_invs = []
    for _ in range(num_bootstraps):
        resample = rng.choice(masses_cut, size=n_cut, replace=True)
        q_b, _ = np.histogram(resample, bins=bins, range=range_gev)
        qb = q_b.astype(np.float64) / (np.sum(q_b) + 1e-12)
        qb = np.clip(qb, 1e-12, 1.0)
        qb /= qb.sum()
        d = float(jensenshannon(p, qb, base=2.0) ** 2)
        if d > 1e-9:
            boot_invs.append(1.0 / d)
    
    unc = float(np.std(boot_invs)) if boot_invs else 0.0
    return js_div, inv_jsd, unc
```

---

### 3.2 JSD Configuration Parameters

| Parameter | `evaluate_mass_sculpting.py` | `evaluate_controlled_ablation.py` |
|---|---|---|
| **Mass range** | `[2000, 5000] GeV` (default bins=40) | `[2000, 5000] GeV` (bins=40, fixed) |
| **Number of bins** | 40 (uniform, Δm=75 GeV) | 40 (uniform, Δm=75 GeV) |
| **Reference dist (P)** | All background events (N=20,000) | All background events |
| **Selected dist (Q)** | Events with score > τ (top 10%/5%/1%) | Same |
| **Normalization** | Density=True → then renormalize to sum=1 | Counts → normalize to sum=1 |
| **Empty bin handling** | `eps=1e-12` added before normalization | `clip(1e-12, 1.0)` then renormalize |
| **Logarithm base** | `base=np.e` (natural log) | `base=2.0` (base-2) ⚠️ **INCONSISTENCY** |
| **JSD formula** | `js_dist²` where `js_dist = jensenshannon(p, q)` | Same |
| **Bootstrap** | Not implemented | 500 resamples, seed=42 |
| **Uncertainty** | Not reported | Std of bootstrap 1/JSD |

---

### 3.3 Inverse JSD (Background Smoothness)
**Definition:** `Smoothness = 1 / JSD(P || Q)`

- **Interpretation:** Higher = less mass sculpting (selected distribution closer to reference)
- **Scale:** 9999.0 used as cap when JSD < 1e-9 (or 1e-7 in mass_sculpting.py)
- **Reported at:** Top 10% (90th percentile), Top 5% (95th), Top 1% (99th)
- **Both:** Raw dijet mass and RSD groomed mass (m_RSD)

---

### 3.4 Critical JSD Implementation Issues

| Issue | Location | Severity | Impact |
|---|---|---|---|
| **Logarithm base inconsistency** | `evaluate_mass_sculpting.py:56` uses `base=np.e`; `evaluate_controlled_ablation.py:80` uses `base=2.0` | **High** | JSD values differ by factor `(ln 2)² ≈ 0.48`; 1/JSD differs by ~2.08x |
| **Bootstrap only on Q** | Both implementations resample only the selected distribution | **Medium** | Underestimates uncertainty; reference P treated as fixed truth |
| **Cap value arbitrary** | 9999.0 (controlled) vs 9999.0 (sculpting) for JSD < 1e-9/1e-7 | **Low** | Affects only near-perfect smoothness cases |
| **Binning fixed at 40** | No systematic study of binning sensitivity | **Medium** | 75 GeV bins; detector resolution ~3.5% (≈100 GeV at 3 TeV) |
| **Inverse JSD not a direct smoothness measure** | 1/JSD is monotonic but non-linear; not calibrated to physical significance | **High** | Cannot interpret "427 vs 303" as 1.4× better in physical terms |

---

## 4. Statistical Significance & Uncertainty

### 4.1 Bootstrap Uncertainty (JSD only)
- **Implemented in:** `evaluate_controlled_ablation.py:83-97`
- **Method:** 500 Poisson bootstrap resamples of selected events (masses_cut)
- **Random seed:** Fixed at 42 (`np.random.RandomState(42)`)
- **Reported:** `inv_jsd_unc` as standard deviation of bootstrap 1/JSD values
- **Missing:** Bootstrap on reference distribution; bootstrap on ROC AUC; bootstrap on correlations

### 4.2 Seed-to-Seed Variation
- **Framework:** `controlled_structure` configs with seeds 42, 123, 999
- **Status:** Only 1 of 12 configs trained (nodewise_ae_s999)
- **Not yet measurable:** Need all 3 seeds × 4 architectures trained

### 4.3 Statistical Significance (Discovery)
**Source:** `src/ultimate_discovery_proof.py:211-215`

```python
significance = (mu_sig - mu_bg) / sigma_bg  # Gaussian σ
```

- **Definition:** `(Mean_signal - Mean_background) / Std_background`
- **Assumption:** Background scores approximately Gaussian
- **Not:** p-value, CLs, or profile likelihood
- **Reported in BASELINE.md:** 0.37σ for frozen baseline

---

## 5. Metric Validation Status

| Metric | Implemented | Unit Tested | Uncertainty | Cross-Validated | Issues |
|---|---|---|---|---|---|
| ROC AUC | ✅ | ✅ (test_mass_correlation) | ❌ | N/A | No CI |
| Background Rejection | ✅ | ❌ | ❌ | N/A | - |
| Separation Ratio | ✅ | ❌ | ❌ | N/A | - |
| Significance (σ) | ✅ | ❌ | ❌ | N/A | Gaussian approx |
| Mass-Matched ROC AUC | ✅ | ✅ (test_mass_correlation) | ❌ | N/A | Population mismatch |
| Micro-Fine ROC AUC | ✅ | ✅ (test_deep_physics) | ❌ | N/A | Low stats at ±50 GeV |
| Pearson r | ✅ | ✅ (test_mass_correlation) | ❌ | N/A | - |
| Spearman ρ | ✅ | ✅ (test_mass_correlation) | ❌ | N/A | - |
| Ridge R² | ✅ | ✅ (test_mass_correlation) | ❌ | ❌ (single 80/20 split) | No CV |
| JSD | ✅ | ❌ | ⚠️ (controlled only) | N/A | Log base inconsistency |
| Inverse JSD | ✅ | ❌ | ⚠️ (controlled only) | N/A | Not physical smoothness |
| Bootstrap 1/JSD unc | ✅ (controlled) | ❌ | N/A | N/A | Only resamples Q |

---

## 6. Population Mismatch Issues

| Evaluation | Background Events | Signal Events | Notes |
|---|---|---|---|
| **Architecture metrics.json** | 995,000 | 100,000 | Full dataset (skipping first 5k train) |
| **Mass sculpting (20k)** | 20,000 | N/A | Events 5,000-25,000 |
| **Mass correlation** | 10,000 | 5,000 | Subset for speed |
| **Deep physics** | 10,000 | 5,000 | Same as mass correlation |
| **Controlled ablation** | 10,000 | 5,000 | Same population |

**Critical:** Metrics from different evaluations **CANNOT be directly compared** — they use different event populations. The architecture ROC AUC (0.7815 on 995k) ≠ mass correlation AUC (0.7076 on 10k).

---

## 7. Mathematical Implementation Problems

### 7.1 Angular Periodicity in Loss
**Location:** All reconstruction losses use `MSE(recon_x, x)` on φ directly
**Problem:** φ ∈ [-π, π] with wrap-around; MSE penalizes φ=3.1 vs φ=-3.1 as large error (6.2) instead of true Δφ ≈ 0.08
**Impact:** Angular reconstruction error inflated; may explain inverted angular AUC
**Status:** ❌ NOT FIXED — no periodic loss (von Mises, wrapped MSE, or sin/cos encoding)

### 7.2 Adversarial Decorrelation Bypass
**Location:** `src/adversarial_unlearning_ae.py:308-320` (AdversarialNodeGNN.forward)
**Problem:** 
```python
reversed_z = grad_reverse(global_z, alpha)
pred_mass = self.adversary(reversed_z)
# But anomaly score uses node decoder:
recon_x = self.decoder(z)  # z is node-level, NOT conditioned on global_z
```
**Impact:** Adversary constrains `global_z` but node decoder `Dec(z_i)` still encodes mass via pT reconstruction
**Evidence:** R²(score→mass) = 0.0043 (adv) vs 0.0025 (production) — **worse**

### 7.3 MET Target vs Physical MET
**Observation:** Mean true MET ≈ 75-80 GeV for both bg and sig (fully hadronic → true MET=0)
**Implementation:** `event_to_graph` computes MET from visible particles, uses as regression target
**Issue:** MET prediction learns detector noise pattern, not physics conservation law
**Evidence:** MET correlation with mass: +0.0665 (production) vs +0.0858 (standalone) — helps but not fundamental

### 7.4 Inverse JSD Interpretation
**Claim in reports:** "Background Smoothness = 1/JSD"
**Reality:** 1/JSD is a monotonic transformation of JSD; not calibrated to physical smoothness
**Problem:** Cannot say "427 is 41% smoother than 303" — the relationship is non-linear
**Recommendation:** Report JSD directly; use 1/JSD only as relative ranking

---

## 8. Validation Gaps

| Gap | Description | Priority |
|---|---|---|
| ROC AUC confidence intervals | No bootstrap/DeLong implementation | High |
| Angular periodic loss | φ MSE ignores 2π wrap-around | High |
| Adversarial score decorrelation | Proposed but not trained/evaluated | High |
| Multi-seed statistical validation | 11/12 controlled configs untrained | High |
| Signal topology generalization | Only W'→XY→4q at 3.5 TeV tested | Medium |
| JSD binning sensitivity | Fixed 40 bins, no systematic study | Medium |
| Mass-matched population consistency | Different eval populations across scripts | Medium |
| Score-level decorrelation training | 9 configs proposed, 0 executed | High |
| Background-only vs mixed signal eval | Signal file contains 1M bg + 100k sig; truth filtering works but not validated for mass correlation | Low |

---

## 9. Verified vs Unverified Claims

| Claim | Status | Evidence |
|---|---|---|
| "GAT outperforms MLP" | ✅ **VERIFIED** | AUC 0.7815 vs 0.4848 on 995k/100k |
| "pT-only score better than full" | ✅ **VERIFIED** | Matched AUC 0.7676 vs 0.6804 |
| "Angular score inverted" | ✅ **VERIFIED** | AUC 0.4172 < 0.50 |
| "Mass-matched discrimination survives" | ✅ **VERIFIED** | AUC 0.6804 at [3.3,3.7] TeV; 0.6719 at ±50 GeV |
| "GAT sculpting mass" | ✅ **VERIFIED** | 1/JSD: 303 (standalone) vs 427 (production) |
| "MET regularizes mass" | ✅ **VERIFIED** | 1/JSD improves from 303→427; r drops 0.0858→0.0665 |
| "Adversarial decorrelates mass" | ❌ **REFUTED** | 1/JSD: 351 (adv) vs 427 (production); R² score→mass: 0.0043 vs 0.0025 |
| "Graph message passing helps" | ⚠️ **COMPLEX** | Inference-time: disconnected (0.793) > relational (0.680); Scrambled kills it (0.471) |
| "NodeWiseAE matches GAT" | ❓ **UNTESTED** | Only 1 seed trained; not evaluated |
| "DeepSets matches GAT" | ❓ **UNTESTED** | 0 seeds trained |
| "Score-GRL/DisCo works" | ❓ **UNTESTED** | 9 configs proposed, 0 trained |

---

**End of Scientific Validation Status**