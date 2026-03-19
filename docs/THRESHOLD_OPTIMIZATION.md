# Threshold Optimization and Conformal Prediction

This document describes the threshold optimization system used in the adaptive ROA pipeline. It covers what the thresholds are, how they are computed, the available optimization variants, and how they drive predictions.

## Table of Contents

1. [Overview](#overview)
2. [Core Threshold Parameters](#core-threshold-parameters)
3. [Probability Estimation (Monte Carlo Sampling)](#probability-estimation-monte-carlo-sampling)
4. [Threshold Optimization (Stage 1)](#threshold-optimization-stage-1)
5. [q_hat Calibration (Stage 2)](#q_hat-calibration-stage-2)
6. [Prediction Rules](#prediction-rules)
7. [Evaluation Variants in `evaluate_full_roa_fast`](#evaluation-variants-in-evaluate_full_roa_fast)
8. [Configuration Reference](#configuration-reference)
9. [End-to-End Pipeline Flow](#end-to-end-pipeline-flow)

---

## Overview

The threshold system provides **uncertainty-aware classification** of initial states in a dynamical system as SUCCESS (converges to an attractor), FAILURE (does not converge), UNCERTAIN (insufficient confidence), or INVALID (endpoint not physically meaningful).

The system operates in two stages:

1. **Threshold Optimization**: Find optimal decision boundary parameters (lambda_star, delta_star) on a validation set.
2. **q_hat Calibration**: Compute a calibration threshold on a held-out calibration set that provides formal coverage guarantees via conformal prediction.

Key source files:
- `adaptive_roa/conformal/config.py` — `ConformalConfig` dataclass
- `adaptive_roa/conformal/probability_estimator.py` — `ProbabilityEstimator` (MC sampling)
- `adaptive_roa/conformal/lambda_optimizer.py` — `LambdaOptimizer` (grid search)
- `adaptive_roa/conformal/calibrator.py` — `Calibrator` (q_hat computation + prediction sets)
- `adaptive_roa/adaptive_v2/eval/full_roa.py` — `evaluate_full_roa_fast` (evaluation)
- `adaptive_roa/adaptive_v2/types.py` — `ThresholdState` dataclass

---

## Core Threshold Parameters

### ThresholdState

Defined in `adaptive_roa/adaptive_v2/types.py`:

```python
@dataclass
class ThresholdState:
    lambda_star: float              # Optimal decision boundary center
    delta_star: float               # Uncertainty half-width
    q_hat: float | None = None      # Single-class calibration threshold (training)
    q_hat_eval: float | None = None              # Evaluation-time q_hat (held-out cal set)
    q_hat_success_eval: float | None = None      # Per-class success threshold (eval)
    q_hat_failure_eval: float | None = None      # Per-class failure threshold (eval)
```

### What Each Parameter Means

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| `lambda_star` | λ* | (0, 1) | The central decision boundary on p(success). Separates confident failure from uncertain from confident success. |
| `delta_star` | δ* | (0, 0.5) | Half-width of the uncertainty band around λ*. Points with p(success) in [λ*−δ*, λ*+δ*] are classified as UNKNOWN. |
| `q_hat` | q̂ | ℝ⁺ | Non-conformity score threshold for conformal prediction sets. A label is included in the prediction set if its score ≤ q̂. Provides (1−α) coverage guarantee. |
| `q_hat_success` | q̂_s | ℝ⁺ | Per-class threshold for the SUCCESS label (multi-class variant). |
| `q_hat_failure` | q̂_f | ℝ⁺ | Per-class threshold for the FAILURE label (multi-class variant). |

### Decision Regions (λ±δ band)

Given p_success for a state:

```
p_success < λ* − δ*       →  FAILURE  (confident)
λ* − δ* ≤ p_success ≤ λ* + δ*  →  UNKNOWN  (uncertain)
p_success > λ* + δ*       →  SUCCESS  (confident)
```

With the additional **p_invalid veto**: if p_invalid ≥ λ*−δ*, the point is marked INVALID regardless of p_success.

### Prediction Labels

| Label | Value | Meaning |
|-------|-------|---------|
| SUCCESS | 1 | Confidently predicted to converge |
| FAILURE | 0 | Confidently predicted to not converge |
| UNCERTAIN | -1 | Model is not confident enough to decide |
| INVALID | -2 | Endpoint is not physically meaningful (high p_invalid) |

---

## Probability Estimation (Monte Carlo Sampling)

Before thresholds can be optimized, the model's uncertainty must be quantified. This is done by the `ProbabilityEstimator` (`adaptive_roa/conformal/probability_estimator.py`).

### How It Works

For each initial state x:

1. Sample K latent vectors z₁, z₂, ..., z_K ~ N(0, I).
2. For each z_k, run the flow matcher to predict an endpoint: ê_k = FlowMatcher(x, z_k).
3. Classify each endpoint using the system's attractor classifier: label_k = classify_attractor(ê_k, radius).
4. Compute empirical probabilities:
   - **p_success(x)** = count(label_k = 1) / K
   - **p_failure(x)** = count(label_k = −1) / K
   - **p_invalid(x)** = count(label_k = 0) / K

### Parameters

- `num_mc_samples` (K): Number of MC forward passes per state (default: 100).
- `mc_batch_size`: States processed in parallel per GPU batch (default: 1024).
- `attractor_radius`: Radius for `classify_attractor()` (default: 0.2).

### Invalid Endpoint Refinement

When `refine_invalids=True`, endpoints classified as invalid (label=0) are refined by re-running the ODE from a random t_start ~ U[t_min, t_max] to t=1.0, using the invalid endpoint as the warm-start. This is repeated up to `refine_max_attempts` times until the endpoint resolves to success or failure.

---

## Threshold Optimization (Stage 1)

The `LambdaOptimizer` (`adaptive_roa/conformal/lambda_optimizer.py`) finds optimal λ* and/or δ* via grid search on a validation set.

### Optimization Modes

Controlled by `config.optimize_mode`:

#### Mode: `"lambda"` — Optimize λ* with fixed δ

- Searches λ ∈ [δ, 1−δ] over `lambda_grid_size` points.
- δ is fixed at `config.delta`.
- Returns: (λ*, config.delta).

#### Mode: `"delta"` — Optimize δ* with fixed λ=0.5

- Searches δ ∈ [delta_min, delta_max] over `delta_grid_size` points.
- λ is fixed at 0.5.
- Returns: (0.5, δ*).

#### Mode: `"joint"` — Optimize both λ* and δ* simultaneously

- 2D grid search over (λ, δ) pairs.
- λ ∈ [delta_min, 1−delta_min] with `lambda_grid_size` points.
- δ ∈ [delta_min, delta_max] with `delta_grid_size` points.
- Only evaluates valid combinations where λ−δ > 0 and λ+δ < 1.
- Returns: (λ*, δ*).

### Optimization Objectives

Controlled by `config.optimize_objective`:

#### `"loss"` (default) — Weighted misclassification + unknown rate

```
Loss = w × misclassification_rate + (1−w) × unknown_rate
```

- `w` (default 0.9): Weight for misclassification penalty.
- `misclassification_rate`: Fraction of confident predictions that are wrong.
- `unknown_rate`: Fraction of all points classified as UNKNOWN.
- High w penalizes errors more; low w penalizes uncertainty more.

#### `"jstat"` — J-statistic based

```
Loss = w × (1 − J) + (1−w) × unknown_rate
```

- J = Youden's J-statistic = TPR + TNR − 1 (ranges from −1 to 1, 1 is perfect).
- Class-balanced: penalizes both false positives and false negatives equally.
- Useful when class imbalance makes raw misclassification rate misleading.

#### `"f1"` — F1 target with minimum separatrix

Uses `optimize_lambda_delta_for_f1_targets()` in `full_roa.py`:

1. Grid search over (λ, δ) pairs.
2. Find all pairs achieving F1 ≥ `target_f1`.
3. Among those, pick the pair with minimum separatrix_pct.
4. If no pair achieves the target F1, returns the best F1 found.

#### `"fixed"` — No optimization

- Uses `config.fixed_lambda_star` and `config.fixed_delta_star` directly.
- Bypasses the grid search entirely.

### Decision Rules

Controlled by `config.decision_rule`:

#### `"one_sided"` — Uses only p_success

```
INVALID  if p_invalid ≥ λ−δ
SUCCESS  if p_success > λ+δ  (and not invalid)
FAILURE  if p_success < λ−δ  (and not invalid)
UNKNOWN  otherwise
```

Used for systems where failure is simply "not success" (e.g., pendulum).

#### `"two_sided"` — Uses both p_success and p_failure

```
INVALID  if p_invalid ≥ λ−δ
SUCCESS  if p_success > λ+δ  (and not invalid)
FAILURE  if (1 − p_failure) < λ−δ  (and not invalid)
UNKNOWN  otherwise
```

Used for systems with explicit failure conditions (e.g., CartPole where falling over is distinct from not reaching the target).

The two-sided rule also handles the rare case where both success and failure conditions are met simultaneously — such points are classified as UNKNOWN.

---

## q_hat Calibration (Stage 2)

After λ* and δ* are found, the `Calibrator` (`adaptive_roa/conformal/calibrator.py`) computes q̂ on a separate calibration set to provide formal coverage guarantees.

### Non-Conformity Scores

The non-conformity score measures how "strange" it would be to assign a particular label to a state, given its estimated probabilities. Lower scores mean the label is more conforming.

#### One-Sided Scores (p_success only)

| True Label | Score Formula | Interpretation |
|------------|---------------|----------------|
| SUCCESS (1) | max(0, (λ+δ) − p_success) | Low p_success makes "SUCCESS" strange |
| FAILURE (−1) | max(0, p_success − (λ−δ)) | High p_success makes "FAILURE" strange |
| UNKNOWN (0) | max(0, (λ−δ) − p_s, p_s − (λ+δ)) | Being outside [λ−δ, λ+δ] makes "UNKNOWN" strange |

#### Two-Sided Scores (p_success and p_failure)

Uses derived thresholds u = λ+δ (success threshold on p_s) and v = 1−λ+δ (failure threshold on p_f):

| True Label | Score Formula | Interpretation |
|------------|---------------|----------------|
| SUCCESS (1) | max(0, u − p_success) | Low p_success makes "SUCCESS" strange |
| FAILURE (−1) | max(0, v − p_failure) | Low p_failure makes "FAILURE" strange |
| UNKNOWN (0) | max(0, p_s − u, p_f − v) | High confidence in either class makes "UNKNOWN" strange |

### Calibration Algorithm

Given a calibration set {(x_i, y_i)} with estimated probabilities:

1. Compute non-conformity score for each calibration point using its **true** label:
   ```
   s_i = nonconformity_score(p_success_i, y_true_i, λ*, δ*)
   ```

2. Compute the finite-sample quantile level:
   ```
   quantile_level = min((1 − α) × (n + 1) / n, 1.0)
   ```
   where α is the significance level (e.g., 0.1 for 90% coverage) and n is the calibration set size.

3. Set q̂ to the quantile of the scores:
   ```
   q_hat = quantile(scores, quantile_level)
   ```

### Coverage Guarantee

By conformal prediction theory:
```
P(y_true ∈ prediction_set) ≥ 1 − α
```

This guarantee holds marginally (over the randomness of the calibration set) with no distributional assumptions beyond exchangeability of calibration and test data.

### Prediction Sets

For a new test point with estimated probabilities (p_success, p_failure):

1. For each candidate label y ∈ {−1, 0, 1}:
   - Compute non-conformity score s_y
   - Include y in the prediction set if s_y ≤ q̂

2. Map the prediction set to a decision:
   - Singleton {SUCCESS} → predict SUCCESS
   - Singleton {FAILURE} → predict FAILURE
   - Multiple labels in set → UNCERTAIN
   - Empty set → UNCERTAIN
   - UNKNOWN in set → UNCERTAIN

### Per-Class Calibration (Multi-Class Variant)

`calibrate_per_class()` splits the calibration set by true label and computes separate thresholds:

- **q̂_success**: Computed only on calibration points with y_true = 1
- **q̂_failure**: Computed only on calibration points with y_true = −1

Each uses its own finite-sample correction:
```
quantile_level_s = min((1 − α) × (n_success + 1) / n_success, 1.0)
quantile_level_f = min((1 − α) × (n_failure + 1) / n_failure, 1.0)
```

Either threshold can be None if no calibration points exist for that class.

#### Unknown Handling Modes

When using per-class q̂ values, the UNKNOWN label needs a threshold. Three modes are available:

| Mode | Unknown Threshold | Effect |
|------|-------------------|--------|
| `"min"` | min(q̂_s, q̂_f) | Conservative: UNKNOWN enters set only with very low scores |
| `"max"` | max(q̂_s, q̂_f) | Liberal: UNKNOWN enters set more easily |
| `"skip"` | N/A | UNKNOWN is never included in any prediction set |

---

## Evaluation Variants in `evaluate_full_roa_fast`

The main evaluation function (`adaptive_roa/adaptive_v2/eval/full_roa.py`) computes multiple prediction variants simultaneously on a held-out test set:

### 1. `lambda_delta` — Primary λ±δ band

Uses the optimized λ* and δ* directly:
- SUCCESS if p_success > λ*+δ*
- FAILURE if p_success < λ*−δ* (one-sided) or (1−p_failure) < λ*−δ* (two-sided)
- INVALID if p_invalid ≥ λ*−δ*
- UNKNOWN otherwise

### 2. `fixed_threshold` — Baseline with hardcoded thresholds

- SUCCESS if p_success > 0.6
- FAILURE if p_failure > 0.6
- INVALID if p_invalid > 0.5
- UNKNOWN otherwise

Serves as a naive baseline for comparison.

### 3. `lambda_only` — Degenerate case with δ=0

- SUCCESS if p_success > λ*
- FAILURE if p_success < λ* (one-sided) or p_failure > 1−λ* (two-sided)
- INVALID if p_invalid ≥ λ*

No uncertainty band; every non-invalid point gets a confident prediction.

### 4. `qhat_prediction_sets` — Conformal sets with single q̂

Requires q̂ from calibration. For each point:
1. Compute non-conformity scores for labels {−1, 0, 1}
2. Include label in set if score ≤ q̂
3. Singleton sets → confident prediction; otherwise → UNCERTAIN

Reports coverage (fraction of points whose true label is in the prediction set).

### 5. `qhat_multi_class_{min,max,skip}` — Per-class q̂ variants

Requires q̂_success and q̂_failure. Evaluated for all three unknown modes:
- `qhat_multi_class_min`
- `qhat_multi_class_max`
- `qhat_multi_class_skip`

### 6. Conservative Metrics

For both `lambda_delta` and `qhat_prediction_sets`, conservative metrics are also computed where all uncertain/invalid points are treated as FAILURE. This gives a safety-aware view: precision is unchanged, but recall drops for true ROA points hiding in the separatrix.

### Outputs Per Variant

Each prediction variant produces:
- **Classification metrics**: accuracy, precision, recall, specificity, F1, TP/TN/FP/FN
- **Separatrix metrics**: invalid_pct, uncertain_pct, separatrix_pct (invalid + uncertain)
- **Coverage** (q̂ variants only): fraction of true labels in prediction sets
- **Endpoint error statistics**: Mean geodesic distance between predicted and true endpoints, partitioned by prediction region (certain_success, certain_failure, uncertain, invalid, separatrix)
- **MC sample error statistics**: Per-point error variability across MC samples

---

## Configuration Reference

All threshold-related parameters are in `ConformalConfig` (`adaptive_roa/conformal/config.py`), loaded from `configs/conformal/default.yaml`:

```yaml
# Decision boundary parameters
delta: 0.05          # Initial δ (also used as fixed δ when optimize_mode="lambda")
w: 0.9               # Loss weight: w × error + (1-w) × unknown_rate

# Coverage parameters
alpha: 0.1            # Significance level (α=0.1 → 90% coverage guarantee)

# Monte Carlo sampling
num_mc_samples: 100   # K forward passes per state
mc_batch_size: 1024   # GPU batch size for MC sampling

# Classification
attractor_radius: 0.2 # Endpoint classification radius

# Optimization mode: "lambda", "delta", or "joint"
optimize_mode: lambda

# Decision rule: "one_sided" or "two_sided"
decision_rule: one_sided

# Grid search parameters
lambda_grid_size: 100
delta_grid_size: 100
delta_min: 0.01
delta_max: 0.49

# p_invalid veto during threshold optimization
use_p_invalid_veto: true

# Optimization objective: "loss", "jstat", "f1", or "fixed"
optimize_objective: loss

# F1-based optimization target (only when optimize_objective="f1")
target_f1: 0.90

# Fixed thresholds (only when optimize_objective="fixed")
fixed_lambda_star: 0.5
fixed_delta_star: 0.1

# Invalid endpoint refinement
refine_invalids: false
refine_t_min: 0.7
refine_t_max: 0.9
refine_num_steps: 100
refine_max_attempts: 5
```

---

## End-to-End Pipeline Flow

The following shows how thresholds are computed and used within one epoch of the adaptive engine (`adaptive_roa/adaptive_v2/engine.py`):

```
┌─────────────────────────────────────────────────────┐
│  1. TRAIN FLOW MATCHER                              │
│     FlowMatchingTrainer.fit(train_data)              │
│     → trained model                                  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  2. THRESHOLD OPTIMIZATION (Stage 1)                │
│     threshold_backend.bind_model(model)              │
│     threshold_state = threshold_backend.optimize(    │
│         X_val, y_val)                                │
│                                                      │
│     Internally:                                      │
│     a) ProbabilityEstimator.estimate(X_val)          │
│        → p_success, p_failure, p_invalid             │
│     b) LambdaOptimizer.optimize(p_s, y, p_f, p_inv) │
│        → λ*, δ*                                      │
│                                                      │
│     Output: ThresholdState(λ*, δ*, q_hat=None)       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  3. q_hat CALIBRATION (Stage 2)                     │
│     Calibrator.calibrate(                            │
│         p_success_d1, y_true_d1, λ*, δ*)            │
│                                                      │
│     a) Compute non-conformity scores using TRUE      │
│        labels on held-out calibration set D1         │
│     b) q_hat = quantile(scores,                      │
│                  (1−α)(n+1)/n)                       │
│                                                      │
│     Optional: calibrate_per_class → q̂_s, q̂_f       │
│                                                      │
│     Output: ThresholdState(λ*, δ*, q̂, q̂_s, q̂_f)    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  4. EVALUATION                                       │
│     evaluate_full_roa_fast(                          │
│         model, system, eval_file,                    │
│         lambda_star=λ*, delta=δ*,                    │
│         q_hat=q̂, q_hat_success=q̂_s,                 │
│         q_hat_failure=q̂_f)                           │
│                                                      │
│     Runs MC sampling on test states, then computes   │
│     all prediction variants:                         │
│     • lambda_delta (λ*±δ*)                           │
│     • fixed_threshold (baseline)                     │
│     • lambda_only (δ=0)                              │
│     • qhat_prediction_sets (if q̂ provided)           │
│     • qhat_multi_class_{min,max,skip}                │
│       (if q̂_s, q̂_f provided)                        │
│     • conservative variants                          │
│                                                      │
│     Output: Comprehensive metrics dict               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  5. SAVE ARTIFACTS                                   │
│     EpochArtifacts → artifacts_v2.json               │
│     Contains threshold_state, eval_metrics, etc.     │
└─────────────────────────────────────────────────────┘
```

### Summary

The threshold optimization system is a **two-stage conformal prediction pipeline**:

- **Stage 1** finds the optimal decision boundary (λ*, δ*) that balances classification accuracy against uncertainty, using grid search with configurable objectives (loss, J-statistic, F1, or fixed).
- **Stage 2** calibrates q̂ on held-out data to provide a formal (1−α) coverage guarantee via non-conformity scores.
- **Prediction** maps estimated probabilities to classification decisions through either the λ±δ band (direct thresholding) or conformal prediction sets (q̂-based set-valued predictions).
- **Evaluation** computes all variants simultaneously, enabling direct comparison of different thresholding strategies on the same MC samples.
