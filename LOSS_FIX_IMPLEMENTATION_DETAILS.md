# loss_fix Branch — End-to-End Implementation Details

This document records the implementation details of all changes developed on the **loss_fix** branch (before merging main). It serves as a reference to re-implement or port these features onto the post-merge codebase.

---

## 1. Rollout Loss (Flow Matching)

### 1.1 Overview

- **Purpose:** Add an auxiliary training loss that compares *autoregressively rolled-out* model predictions to ground-truth future chunks, encouraging multi-step consistency.
- **Total loss:** `L_total = L_fm + Σ_k w_k * L_rollout_k` (flow-matching loss plus weighted rollout losses).

### 1.2 FlowMatching API (genMoPlan/models/generative/flow_matching.py)

**New constructor parameters:**

- `use_rollout_loss: bool = False` — Enable rollout loss.
- `rollout_steps: int = 1` — Number of rollout steps (K). Dataset provides `rollout_steps - 1` target chunks.
- `rollout_weighting` — List of weights `[w_1, ..., w_{K-1}]` or dict:
  - `type: "exp"`, `decay: float` → weights `decay^k`;
  - `type: "linear"`, `start`, `end` → linear schedule.
- `rollout_loss_type: str = "l2"` — `"l2"` or `"manifold"` (manifold distance when `self.manifold` is set).
- `rollout_sample_kwargs: dict` — Sampling for rollout: `n_timesteps`, `integration_method`, `batch_frac` (subsample batch to control compute).
- `rollout_operator: str = "fixed"` — `"fixed"` or `"adaptive_stride"`.
- `adaptive_rollout: dict` — If `enabled: True`, sets `rollout_operator` from `mode` (e.g. `"adaptive_stride"`).

**compute_loss(x_target, cond, ..., rollout_targets=None):**

- When `use_rollout_loss` is True and `rollout_targets` is provided (non-empty tensor or dict with `targets`):
  - **Fixed operator:** `_compute_rollout_loss(...)` — expects `rollout_targets` tensor.
  - **Adaptive operator:** `_compute_adaptive_rollout_loss(...)` — expects `rollout_targets` dict with `targets`, `target_lengths`, `valid_mask`, optionally `shared_shifts`.
- Rollout loss is added to the base FM loss; rollout metrics are merged into `info`.

### 1.3 Fixed Rollout Loss (_compute_rollout_loss)

- **Inputs:** `x_target` [B, T, D], cond, global_query, local_query, mask, `rollout_targets` [B, K-1, H, D] or [K-1, H, D] (expanded to batch).
- **Loop k = 0 .. K-2:**
  - History = last `history_length` steps of current trajectory (or from previous rollout prediction).
  - `conditional_sample_grad(cond from history, ...)` → predicted trajectory (gradients enabled).
  - Predicted horizon = trajectory[:, history_length:, :]; compare to `rollout_targets[:, k, :, :]`.
  - Step loss: L2 or manifold distance; apply weight `rollout_weights[k]`.
  - Update “current history” for next step from predicted trajectory (or GT for non-sampled indices if `batch_frac < 1`).
- **Output:** Sum of weighted step losses; info dict with `rollout_{k+1}_loss`, `rollout_{k+1}_weighted_loss`, `rollout_total_unweighted`, `rollout_weighted_loss`.

### 1.4 Adaptive Rollout Loss (_compute_adaptive_rollout_loss)

- **Inputs:** `rollout_targets` dict:
  - `targets` [B, K-1, max_span, D] — ground-truth chunks (variable length via `valid_mask`).
  - `target_lengths` [B, K-1] — actual length per step/sample.
  - `valid_mask` [B, K-1, max_span] — which timesteps are valid.
  - `shared_shifts` (optional) [B, K-1] — anchor shift per step for history update.
- **Loop k = 0 .. K-2:**
  - Condition from current history; `conditional_sample_grad(...)` → pred horizon.
  - Compare pred horizon to `targets[:, k, :, :]` using `_compute_masked_step_loss(pred, gt, valid_mask[:, k, :])`.
  - Weight and accumulate; update history using `shared_shifts[:, k]` (or default 1) to pick which predicted timestep starts the next history window.
- **Output:** Total weighted loss; info with `adaptive_rollout_step_{k+1}_loss`, `adaptive_rollout_span_{k+1}`, `adaptive_rollout_shared_shift_{k+1}`, etc.

### 1.5 Differentiable Sampling (conditional_sample_grad)

- Same as `conditional_sample` but **without** `@torch.no_grad()`, so the ODE solve is differentiable for rollout loss backprop.
- Used only during training when computing rollout loss.

### 1.6 Base Class (genMoPlan/models/generative/base.py)

- `loss(..., rollout_targets=None)` and `validation_loss(..., rollout_targets=None)` pass `rollout_targets` through to the method implementation (e.g. FlowMatching).
- Empty tensor `rollout_targets` is treated as None; dict payloads are supported for adaptive rollout.

---

## 2. Dataset & Data Pipeline

### 2.1 Dataset Config (base / method_kwargs)

- **flow_matching.method_kwargs:**  
  `use_rollout_loss`, `rollout_steps`, `rollout_weighting`, `rollout_loss_type`, `rollout_sample_kwargs`, `adaptive_rollout`.
- **base.dataset_kwargs** (or equivalent):
  - `rollout_steps` — must match method’s `rollout_steps`.
  - `rollout_target_mode`: `"gt_future"` (fixed future chunks) or `"adaptive_stride"` (adaptive stride with shared shift schedule).
  - For adaptive: `adaptive_rollout.enabled`, `mode`, `shared_shift_schedule` (e.g. `type: "arithmetic"`, `start`, `delta`), `base_span`, `max_span`.

### 2.2 Trajectory Dataset (genMoPlan/datasets)

- Dataset loads trajectories and, when `rollout_steps` > 1 and `rollout_target_mode` is set, produces:
  - **Fixed (`gt_future`):** Tensor of shape `[rollout_steps-1, horizon_length, dim]` or batched, representing the next K-1 horizon chunks after the current window.
  - **Adaptive (`adaptive_stride`):** Dict with `targets`, `target_lengths`, `valid_mask`, `shared_shifts` derived from the adaptive stride schedule and trajectory length.
- Collate passes this through as `rollout_targets` in the batch.

### 2.3 Data Processing (genMoPlan/utils/data_processing.py)

- **compute_actual_length(length, stride):** `1 + (length - 1) * stride` — used for horizon/step math in rollout and inference.
- Used by flow_matching and trajectory generator for consistent step/length calculations.

---

## 3. Trainer (genMoPlan/utils/trainer.py)

### 3.1 Rollout Targets in Training Step

- Batch may include `rollout_targets` (tensor or dict).
- Trainer passes it to `model.loss(..., rollout_targets=rollout_targets)` so that FlowMatching can compute rollout loss when enabled.

### 3.2 Logging

- Rollout-related keys in `infos` (e.g. `rollout_*`, `adaptive_rollout_*`) are separated and logged together for visibility:
  - `rollout_1_loss`, `rollout_1_weighted_loss`, …; `rollout_total_unweighted`, `rollout_weighted_loss`;
  - or `adaptive_rollout_step_1_loss`, `adaptive_rollout_span_1`, etc.

### 3.3 Detailed Evaluation

- **detailed_eval_freq:** Every N epochs, run a “detailed evaluation” block:
  - Final-state evaluation (e.g. rollout from initial conditions, compare final state to target).
  - Optional: sequential or full-trajectory metrics.
- **evaluate_final_states():** Uses system and trajectory generator to run rollout and compute final state MAE (and per-dim MAE if state names available); returns `final_rollout_mae` and related infos.

### 3.4 Validation

- `validate()` passes batch (including `rollout_targets` if present) to `model.validation_loss(..., rollout_targets=...)` so validation can optionally include rollout loss.
- Return signature (loss_fix): `(val_loss, final_state_loss, real_data_loss, real_fraction)`.

---

## 4. Trajectory Generator (genMoPlan/utils/trajectory_generator.py)

- Used at inference and in trainer’s final-state evaluation.
- Integrates with **stride**, **history_length**, **horizon_length**, and **max_path_length**.
- **compute_actual_length** (or equivalent) used for step counting and path length.
- May accept **max_path_length** and **validation_kwargs** for validation-time rollouts.

---

## 5. Training Scripts

### 5.1 train_trajectory.py

- **initialize_from_checkpoint(gen_model, trainer, args):**
  - **init_from:** Directory containing checkpoint (e.g. stage-1 run).
  - **init_state_name:** `"best.pt"` (or similar).
  - **init_use_ema:** Load EMA weights if True.
  - **init_strict:** Strict load_state_dict.
  - **init_reset_optimizer** / **init_reset_scheduler:** If False, load optimizer and LR scheduler state from checkpoint.
  - After load, `trainer.reset_parameters()` to sync EMA with loaded model.

- Dataset loader args (loss_fix) included: `horizon_length`, `history_length`, `stride`, `observation_dim`, `trajectory_normalizer`, `normalizer_params`, `trajectory_preprocess_fns`, `preprocess_kwargs`, `dataset_kwargs` (with `rollout_steps`, `rollout_target_mode`, `adaptive_rollout` when used).

- Entry point: `evaluate` from `scripts.evaluate` (main) vs `estimate_roa` from `scripts.estimate_roa` (loss_fix) — script wiring difference.

### 5.2 adaptive_train_trajectory.py

- Adaptive training loop (e.g. uncertainty-based data selection, iterative dataset growth). Exact CLI and args aligned with rollout/adaptive rollout config.

### 5.3 resume_train_trajectory.py

- Resume from checkpoint; respects same config (rollout, init_from, etc.) where applicable.

---

## 6. Config Overrides (Cartpole / Experiments)

The following overrides were present in **config/cartpole_pybullet.py** on loss_fix (removed when accepting main’s config). They can be re-added as needed.

### 6.1 Base / Validation

- **init_from**, **init_state_name**, **init_use_ema**, **init_strict**, **init_reset_optimizer**, **init_reset_scheduler** — for two-stage training.
- **detailed_eval_freq** — every N epochs run detailed evaluation (final state, etc.).
- **eval_freq** — basic eval logging frequency.
- **inference_mask_strategy** — e.g. `"first_step_only"` for history masking at inference.

### 6.2 Rollout Loss Overrides

- **rollout_loss_2_steps / 3_steps:** `method_kwargs.use_rollout_loss=True`, `rollout_steps`, `rollout_weighting`, `rollout_loss_type`, `rollout_sample_kwargs`; `dataset_kwargs.rollout_steps`, `rollout_target_mode: "gt_future"`.
- **rollout_loss_exp_decay:** Same with `rollout_weighting: { type: "exp", decay: 0.5 }`.
- **rollout_loss_3_steps_exp:** `decay: 0.7`, `batch_frac: 1.0`.

### 6.3 Adaptive Rollout Overrides

- **adaptive_rollout_v1:**  
  `rollout_operator: "adaptive_stride"`, `rollout_steps: 4`, `rollout_weighting: [1.0, 0.6, 0.4]`, `adaptive_rollout` with `enabled`, `mode: "adaptive_stride"`, `shared_shift_schedule: { type: "arithmetic", start: 2, delta: 1 }`, `base_span`, `max_span`.  
  Matching `dataset_kwargs` with `rollout_target_mode: "adaptive_stride"` and same `adaptive_rollout` schedule.

- **adaptive_rollout_stride1_constshift:** Same idea with `shared_shift_schedule: { start: 1, delta: 0 }`.

- **adaptive_rollout_step1_only / step2_only / step3_only:** Same adaptive config with single non-zero weight in `rollout_weighting` (e.g. `[1,0,0]`, `[0,1,0]`, `[0,0,1]`).

### 6.4 Two-Stage Training

- **stage1_base_fast:** `use_rollout_loss: False`, `rollout_steps: 1`; `dataset_kwargs.rollout_steps: 1`; `detailed_eval_freq: 10`.
- **stage2_adaptive_ft:** Lower LR (e.g. 5e-5), `use_rollout_loss: True`, adaptive rollout config, `detailed_eval_freq: 10`. Use `--init_from <stage1_run>` to load stage-1 checkpoint.

### 6.5 Stride / Horizon / Path Length

- **stride_2**, **stride_3**, **stride_5**, **stride_10**, **stride_19_horizon_31_single_step**, **stride_25_horizon_31_single_step**.
- **horizon_7**, **horizon_15**, **horizon_31**.
- **path_length_150**, **path_length_300**, **path_length_450**, **path_length_613**; **max_path_572**.
- Combined: **stride_2_horizon_15**, **stride_3_horizon_15**, **stride_1_horizon_15_path_300**, etc.

### 6.6 History Masking / Padding

- **history_mask_2 / history_mask_5**, **history_padding_2 / history_padding_5** — history_length and padding/mask toggles; **final_state_evaluation: False** where needed.

---

## 7. Tests

- **tests/genMoPlan/utils/test_trainer.py:** Assertions on `final_rollout_mae` and `final_rollout_mae_*` in loss dicts.
- **tests/genMoPlan/utils/test_trajectory_generator.py:** Trajectory generator behavior (stride, horizon, path length).
- **tests/genMoPlan/datasets/test_utils.py:** Dataset utils (e.g. masking, rollout target construction if present).

---

## 8. Summary Table

| Component | Change |
|-----------|--------|
| **FlowMatching** | Rollout loss (fixed + adaptive), `conditional_sample_grad`, rollout_weighting, rollout_operator, adaptive_rollout dict. |
| **Base generative** | Pass `rollout_targets` into `loss` and `validation_loss`. |
| **Dataset** | `rollout_steps`, `rollout_target_mode` (gt_future / adaptive_stride), `adaptive_rollout` schedule; batch `rollout_targets`. |
| **Data processing** | `compute_actual_length(length, stride)`. |
| **Trainer** | Pass `rollout_targets` to model.loss; rollout logging; detailed_eval_freq; evaluate_final_states (final_rollout_mae). |
| **Trajectory generator** | Stride/horizon/path length and max_path_length for rollout eval. |
| **train_trajectory.py** | init_from checkpoint loading; dataset_kwargs for rollout; optional estimate_roa. |
| **Config** | Rollout and adaptive rollout overrides; two-stage (stage1_base_fast, stage2_adaptive_ft); stride/horizon/path overrides. |

---

*Document generated to capture loss_fix implementation before merging main. Re-implement these pieces on top of the post-merge codebase (systems, eval, main configs) as needed.*
