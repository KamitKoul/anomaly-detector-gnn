# Final Scientific Readiness Gate: D0 Validation & Kaggle GPU Migration

**Date:** 2026-10-31  
**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Execution Target:** Kaggle GPU Infrastructure (`notebooks/phase4_scaling.ipynb`)  
**Verdict:** **B — PARTIALLY VERIFIED** (Gated for Kaggle GPU Execution)

---

## 1. Compliance Audit Across Readiness Gates (G1 – G12)

| Gate | Scientific Requirement | Status | Evidence / Outcome |
|---|---|---|---|
| **G1** | D0–D6 tier definitions reconciled | **PASS** | Proved in `DATASET_TIER_RECONCILIATION.md`. Partition file `bg_train_d6_900k.npy` holds 900,000 unique events; 320,000 is Experiment B's fixed event presentation budget ($10{,}000 \times 32$). |
| **G2** | Background and signal schemas verified | **PASS** | Proved in `INPUT_SCHEMA_AUDIT.md`. Background: `(1000000, 2100)`, Signal: `(100000, 2101)`. Column 2100 is the truth label bit (`1.0`). |
| **G3** | Model input schemas equivalent | **PASS** | Proved in `INPUT_SCHEMA_AUDIT.md`. Particle extraction strictly slices `event[:2100]`. Column 2100 is completely excluded from graph construction and model inputs. Zero label leakage. |
| **G4** | Loss normalization audited | **PASS** | Proved in `LOSS_NORMALIZATION_AUDIT.md`. All 5 controlled tensor unit tests pass in `tests/test_loss_normalization.py`. The 1.23246 vs 0.11032 initial gap is proved to be the Adam convergence trajectory. |
| **G5** | Angular score implementation verified | **PASS** | Proved in `SCORE_DEFINITION_AUDIT.md` and `ANGULAR_SCORE_INVESTIGATION.md`. Strictly uses periodic Riemannian geodesic $\min(\|\Delta\phi\|, 2\pi - \|\Delta\phi\|)$. |
| **G6** | All score definitions mathematically documented | **PASS** | Proved in `SCORE_DEFINITION_AUDIT.md`. Documented $S_{pT}$, $S_{\text{ang}}$, $S_{\text{weighted}}$, and $S_{\text{full}}$ equations and numerical scale ratios ($6.2\times - 8.9\times$). |
| **G7** | Frozen evaluation verified | **PASS** | Proved in `EVALUATION_AUDIT.md`. Evaluates 50,000 held-out background and 25,000 held-out signal events. Bootstrap CIs strictly separated from model-seed variation. |
| **G8** | Full-budget D0 executed with 3,120 updates | **GATED** | Gated to Kaggle GPU per user mandate. Local M4 execution stopped. Execution automated via Cell 5 of `notebooks/phase4_scaling.ipynb`. |
| **G9** | Best/final checkpoints saved and reloaded | **GATED** | Automated in `notebooks/phase4_scaling.ipynb` for CUDA GPU. |
| **G10** | Full-budget evaluation artifacts saved | **GATED** | Automated in `notebooks/phase4_scaling.ipynb` for CUDA GPU. |
| **G11** | No synthetic production fallback | **PASS** | All production entry points assert physical dataset existence and fail closed on missing or dataless APFS stubs. |
| **G12** | Reproducibility manifest complete | **PASS** | SHA-256 manifests, partition tables, and git commit tracking established. |

---

## 2. Scientific Readiness Verdict

```text
========================================================================================
                          SCIENTIFIC READINESS VERDICT: B
                              (PARTIALLY VERIFIED)
========================================================================================
  [X] G1: Tier definitions reconciled (900k pool vs 320k presentations).
  [X] G2-G3: Input schemas audited; 2,101st column isolated; zero label leakage.
  [X] G4: Loss normalization audited; 5/5 unit tests passed.
  [X] G5-G6: Angular score verified; coordinate scale disparity documented.
  [X] G7: Frozen test protocol verified on real LHC collision events.
  [X] G11-G12: Fail-closed guards active; zero synthetic fallbacks in production.
  [!] G8-G10: Full-budget D0 training (3,120 updates), pilot (6 runs), and matrix
      (42 runs) are GATED to Kaggle GPU execution per user instruction.
========================================================================================
```

---

## 3. Kaggle GPU Execution Protocol

To execute the validated real-data scaling study on Kaggle:
1. **Push/Sync Workspace:** Ensure working tree changes are committed and pushed to GitHub.
2. **Open Notebook in Kaggle:** Upload or open [`notebooks/phase4_scaling.ipynb`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/notebooks/phase4_scaling.ipynb).
3. **Attach Datasets:** Attach the LHC Olympics 2020 background and signal datasets to the Kaggle session (`/kaggle/input/`).
4. **Select Accelerator:** Under Notebook Options $\to$ Accelerator, select **GPU T4 x2** or **GPU P100**.
5. **Execute Step-by-Step:**
   - **Cell 2:** Environment & GPU verification (`!nvidia-smi` and dependency check).
   - **Cell 4:** Dataset auto-discovery and symlinking.
   - **Cell 6:** Split verification (`!python src/data/scaling_split.py`).
   - **Cell 8:** Complete unit test suite (`!python -m unittest discover -s tests -p "test_*.py"`).
   - **Cell 10:** **Full-Budget D0 Training (3,120 updates) and Test Evaluation** (`experiments/scaling/experiment_a_fixed_epochs.py --tier D0 --seed 42 --epochs 20`).
   - **Cell 12:** Six-run scaling pilot (`scripts/run_scaling_pilot.py`).
   - **Cell 14:** Full 42-run matrix (`scripts/run_scaling_matrix.py --experiment both --tier all --seed all`).
   - **Cell 16:** Download results bundle (`scaling_results_bundle.tar.gz`).
