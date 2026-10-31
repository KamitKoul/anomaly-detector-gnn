# Research Report: Validation Architecture and Checkpoint Selection Protocol

**Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Date:** October 31, 2026  
**Auditor:** Scientific Remediation Agent  
**Scope:** Phase 4 Protocol for Unsupervised Validation and Generalization Monitoring  

---

## 1. Executive Summary

In the historical training pipeline ([`src/train_experiment.py#L418-L446`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/train_experiment.py#L418-L446)), models were trained for a fixed 20 epochs across 5,000 events without an in-training validation dataset:
1. No validation loss or generalization metrics were computed during training.
2. The model state saved to `checkpoint.pt` was unconditionally the weights at the end of epoch 20, regardless of whether late-epoch overfitting, gradient conflict, or loss divergence had occurred.
3. No early stopping mechanism existed to arrest optimization collapse.

This remediation establishes an independent background validation pipeline and an unsupervised checkpoint selection policy implemented in [`src/validation_pipeline.py`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/validation_pipeline.py).

---

## 2. Validation Dataset Architecture

### Partition Independence
The validation dataset is sourced from the newly established deterministic partition array:
- **Source File:** `data/splits/bg_val_50k.npy`
- **Total Events:** $50,000$ pure QCD background Pythia8 events.
- **Independence Guarantee:**
  $$\mathrm{BG}_{\mathrm{val}} \cap \mathrm{BG}_{\mathrm{train}} = \emptyset, \quad \mathrm{BG}_{\mathrm{val}} \cap \mathrm{BG}_{\mathrm{test}} = \emptyset$$
- **Signal Isolation:** Strictly zero signal events are present in the validation set.

---

## 3. Metric Tracking and Checkpoint Selection Policy

### Monitored Metrics per Epoch:
At the conclusion of each training epoch, the model evaluates on the validation set under `torch.no_grad()`:
- `val_loss_recon`: Mean particle reconstruction MSE (Riemannian periodic).
- `val_loss_met`: Mean MET prediction MSE on held-out background.
- `val_loss_adv`: Validation distance correlation penalty $\mathrm{dCor}^2(S, m_{jj})$.
- `val_loss_total`: Weighted validation composite loss.
- `train_loss_recon`, `train_loss_met`, `train_loss_adv`: Corresponding training metrics.
- `grad_norm`: Optimizer step gradient norm.
- `lr`: Active learning rate.
- `epoch_time_sec`: Time elapsed per epoch.

### Checkpoint Selection Criteria
> [!IMPORTANT]
> **Strict Scientific Contract:** Checkpoint selection must be based **strictly on unsupervised background metrics**.
> Under no circumstances may final test set metrics or signal evaluation AUC be used to select model checkpoints, as doing so constitutes label leakage into an unsupervised model.

- **Primary Selection Metric:** Minimum `val_loss_recon` (background reconstruction fidelity).
- **Secondary Selection Metric (for DisCo models):** Minimum `val_loss_total = val_loss_recon + \lambda_{\mathrm{DisCo}} \mathrm{val\_loss\_adv}`.
- **Persistence Convention:**
  - `checkpoint_best.pt`: Overwritten only when the primary validation metric achieves a new minimum (with tolerance $\Delta \ge 10^{-5}$).
  - `checkpoint_last.pt`: Saved at the final epoch for auditability and post-training trajectory inspection.
- **Early Stopping:** Configurable patience ($P = 10$ epochs by default). If validation reconstruction loss fails to improve for 10 consecutive epochs, training terminates to prevent over-fitting.

---

## 4. Implementation Details and Verification

Implemented in [`src/validation_pipeline.py`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/validation_pipeline.py):
1. `evaluate_validation_epoch`: Clean validation loop guaranteeing model is in `eval()` mode and no gradients accumulate.
2. `CheckpointManager`: Handles `checkpoint_best.pt`, `checkpoint_last.pt`, metric logging, and early stopping.

Unit tests in [`tests/test_validation_pipeline.py`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/tests/test_validation_pipeline.py) confirmed:
- Zero gradient pollution during validation evaluation.
- Correct detection of improvements, patience tracking, and automatic triggering of early stopping.

```
Ran 2 tests in 0.079s — OK.
```
