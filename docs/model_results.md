# Baseline Model Results

> **Synthetic India-inspired dataset; model performance is a prototype demonstration and is not evidence of real-world Indian bridge prediction accuracy.**

## Task
Predict `target_current_overall_condition` (min of deck/superstructure/substructure, 0–9) using only the 30 leakage-safe features from `docs/feature_engineering.md`. Treated as regression (continuous prediction), then rounded/clipped to [0,9] for accuracy-style metrics — condition is ordinal, so exact-class accuracy alone would understate a model that's "off by one," which is why MAE and within-1 accuracy are reported alongside it.

**Why MAE and within-1 accuracy matter here**: a prediction of 6 when the true rating is 7 is a minor, practically tolerable miss; a prediction of 2 when the true rating is 7 is a serious one. MAE captures this magnitude-of-error directly (unlike exact accuracy, which scores both misses as equally "wrong"), and within-1 accuracy reports the practically-relevant question — "is the model at least in the right neighborhood?" — which matters more for a maintenance-prioritization tool than exact-integer matching.

## Setup
- Seed: 42 | Preprocessing: fit on TRAIN only (`build_preprocessor()`), transform-only on val/test
- Train ≤2019 (23,852 rows) | Val 2020–2021 (4,154 rows) | Test 2022–2024 (5,800 rows)
- XGBoost: **not installed**, skipped per instructions (not installed to avoid unnecessary package installation) — RandomForest/GradientBoosting cover the tree-ensemble comparison

## Comparison

| Model | Val MAE | Test MAE | Val RMSE | Test RMSE | Within-1 (test) | Notes |
|---|---|---|---|---|---|---|
| MedianBaseline | 1.288 | 1.305 | 1.825 | 1.875 | 0.688 | naive constant (train median) predictor |
| Ridge | 0.465 | 0.487 | 0.589 | 0.617 | 0.981 | linear; can't capture the nonlinear decay curve |
| RandomForest | 0.396 | 0.415 | 0.505 | 0.523 | 0.993 | larger train→test MAE gap (0.350→0.415) — more overfit |
| **GradientBoosting** | **0.389** | **0.405** | 0.490 | 0.506 | 0.997 | **best val MAE, smallest train→test gap (0.388→0.405) → selected** |

**Selected model: GradientBoostingRegressor**, chosen by validation MAE only. Test metrics reported after selection, untouched: **MAE 0.405, RMSE 0.506, exact-accuracy 0.677, within-1 accuracy 0.997.**

All three real models comfortably beat the median baseline (MAE 1.29–1.31) without approaching suspicious near-perfect accuracy — exact accuracy tops out at 68–73% across models, which is the expected result of the generator's built-in unobserved latent-quality factor and inspector rating noise (`docs/synthetic_data.md`) making perfect prediction from observed features impossible by design.

## Confusion matrix (test, GradientBoosting, rows=actual/cols=predicted, 0–9)
Errors concentrate almost entirely on the diagonal ±1 (visible in the ordinal error distribution: -1: 985, 0: 3928, +1: 868, ±2: 19 total) — the model rarely confuses a "good" bridge for a "poor" one; its misses are graded, not random.

## Top feature importances (GradientBoosting)
1. `prev_overall_condition` (0.312)
2. `bridge_age` (0.204)
3. `prev_substructure_condition` (0.153)
4. `prev_superstructure_condition` (0.148)
5. `material_Timber` (0.129)
6. `years_since_rehab_filled` (0.023)
Everything else (other materials, structure types, climate, traffic) contributes <1% each. The model is leaning almost entirely on **prior condition + age**, which is exactly the intended behavior given how the synthetic generator builds condition trajectories (`docs/synthetic_data.md`).

## Saved artifacts
`models/best_model.joblib`, `models/preprocessor.joblib`, `models/model_metadata.json` (seed, feature list, target, split years, metrics, package versions).

## What to improve before +5/+10-year forecasting
- Climate/traffic features barely register (<1% importance each) — since `target_current` is dominated by `prev_overall_condition`, a longer-horizon target (+5/+10yr) is a better test of whether those features actually carry signal, since prior condition alone will predict a distant future value less well.
- RandomForest's larger train/test gap suggests its depth (10) is mildly over-fit relative to GradientBoosting's shallower trees (depth 3) — worth a lighter depth if RF is revisited.
- No hyperparameter tuning was done (by design, for a first baseline) — worth doing only once a +5/+10-year model is also in place, so tuning targets the harder task rather than over-fitting the easy one.
