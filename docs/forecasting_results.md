# Multi-Horizon Deterioration Forecasting Results

> **Synthetic India-inspired dataset; model performance is a prototype demonstration and is not evidence of real-world Indian bridge prediction accuracy.**

## Task
Two separate models: predict `overall_condition` at **Y+5** and **Y+10**, using only features known at Y (`docs/feature_engineering.md`). Targets come from Target B/C built by `feature_engineering.py`'s nearest-future-match logic; using a future value as a *label* is normal supervised learning, not leakage (see `docs/model_validation.md`).

## A real bug found and fixed before results were usable
Reusing the **global** temporal split (train ≤2019, val 2020–21, test 2022–24) crashed training: **zero rows** in val/test had a usable +10-year target, and effectively none had a usable +5-year target either. Cause: a target 5 or 10 years in the future needs a same-bridge record that far ahead, and the dataset ends at 2024 — so rows with a usable +5 target only exist up to inspection_year 2020, and +10 only up to 2016 (confirmed empirically, not assumed). This is a structural mismatch, not a style choice, so **horizon-specific splits** were computed the same way as the original global split (from the real cumulative row-count-by-year distribution of rows that actually have a usable target for that horizon) — `feature_engineering.py` and the current-condition model/split were **not** touched.

| Horizon | Feasible year range | Train | Val | Test |
|---|---|---|---|---|
| +5yr | 1995–2020 | ≤2014 (14,193 rows) | 2015–2017 (5,614) | 2018–2020 (4,060) |
| +10yr | 1995–2016 | ≤2012 (10,560 rows) | 2013–2014 (3,633) | 2015–2016 (2,357) |

## Results

| Model | Horizon | Val MAE | Test MAE | Val RMSE | Test RMSE | Within-1 (test) |
|---|---|---|---|---|---|---|
| MedianBaseline | +5 | 1.263 | 1.289 | 1.826 | 1.861 | 0.698 |
| Ridge | +5 | 0.481 | 0.492 | 0.607 | 0.621 | 0.983 |
| RandomForest | +5 | 0.439 | 0.457 | 0.552 | 0.573 | 0.989 |
| **GradientBoosting** | **+5** | **0.431** | **0.447** | 0.538 | 0.558 | 0.991 |
| MedianBaseline | +10 | 1.288 | 1.263 | 1.901 | 1.856 | 0.714 |
| Ridge | +10 | 0.497 | 0.519 | 0.627 | 0.651 | 0.980 |
| RandomForest | +10 | 0.453 | 0.465 | 0.577 | 0.592 | 0.988 |
| **GradientBoosting** | **+10** | **0.448** | **0.464** | 0.562 | 0.580 | 0.989 |

XGBoost: not installed, skipped (unchanged from the baseline phase). **GradientBoosting wins both horizons** on validation MAE, selected before touching test.

## Does performance degrade with horizon? Yes, and sensibly
| Task | Test MAE | Test Exact Acc |
|---|---|---|
| Current condition (previous phase) | 0.405 | 0.677 |
| +5 year | 0.447 | 0.627 |
| +10 year | 0.464 | 0.605 |

A small, steady degradation as the horizon grows — exactly what a genuinely-harder, non-leaking forecasting task should look like. No horizon showed suspiciously high (near-perfect) accuracy, so no leakage/over-determinism investigation was needed.

## Feature importance: current vs +5yr vs +10yr
| Feature | Current | +5yr | +10yr |
|---|---|---|---|
| prev_overall_condition | **0.312** | 0.046 | 0.035 |
| bridge_age | 0.204 | **0.271** | **0.231** |
| material_Timber | 0.129 | **0.515** | **0.551** |
| prev_substructure_condition | 0.153 | 0.040 | 0.039 |
| prev_superstructure_condition | 0.148 | 0.034 | 0.023 |
| years_since_rehab_filled | 0.023 | 0.028 | 0.029 |
| annual_rainfall_mm | ~0 | 0.006 | 0.008 |
| adt / heavy_vehicle_traffic | ~0 | ~0 | ~0 |

**How the problem changes with horizon**: for *current* condition, the strongest signal is simply "what was it recently" (prior condition dominates, 31%). As the horizon extends, prior condition's usefulness collapses (31%→5%→3.5%) — a condition rating from years ago says little about a decade from now — and the model shifts almost entirely onto **material and age**, especially `material_Timber` (13%→52%→55%). This matches the generator's design: timber has by far the fastest decay rate (0.035 vs 0.008–0.015 for other materials in `docs/synthetic_data.md`), so over a long horizon, *whether a bridge is timber at all* increasingly dominates where it ends up — a legitimate structural driver, not an artifact.

**Traffic and climate features stay negligible at every horizon** (<1% each). This is an honest limitation to flag before RUL: the generator's traffic_factor/climate_factor terms exist mathematically but contribute far less variance than material/age, so a real deployment (or a richer synthetic generator revision later) would need those signals to matter more before "risk factors" reporting could credibly include them.

## Is the model learning drivers, or copying previous condition?
**Both, but which one dominates changes with horizon.** At zero/short horizon it leans heavily on copying prior condition (reasonable — recent history is genuinely informative short-term). By +10yr it's learned the *actual generative structure* (age × material decay-rate), not just extrapolating the last reading — evidenced by prior-condition's importance collapsing while age/material's importance grows. This is the right pattern for a forecasting model, not a citation of laziness.

## Saved artifacts
`models/best_model_plus5.joblib`, `models/preprocessor_plus5.joblib`, `models/best_model_plus10.joblib`, `models/preprocessor_plus10.joblib`, `models/forecast_metadata.json`. Current-condition baseline models (`best_model.joblib`, `preprocessor.joblib`) untouched. `src/models/forecast.py` provides `predict_future_condition(bridge_features)` → `{"plus5": int, "plus10": int}`, smoke-tested successfully.

## Good enough to proceed to RUL?
Yes, with one caveat carried forward: RUL derivation should expect the same pattern — short-term estimates will lean on recent condition, long-term estimates will lean on material/age — and should not assume traffic/climate will contribute meaningfully to RUL uncertainty without first checking, the same way this phase found they don't for condition forecasting.
