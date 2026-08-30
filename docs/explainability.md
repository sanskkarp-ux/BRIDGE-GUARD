# SHAP Explainability

> **Synthetic India-inspired dataset; explanations describe what the model learned from synthetic data, not validated real-world engineering conclusions.**

## Method
`shap.TreeExplainer` on each GradientBoosting model (current/+5yr/+10yr — the winners selected in earlier phases), run on that task's own held-out test rows (2,000-row sample, `random_state=42`, where test set exceeds that). `FEATURE_COLS` is imported unchanged from `src/features/feature_engineering.py` — no new features were introduced, and target columns were never part of that list to begin with.

## Validation checks (all passed)
- **SHAP corresponds to actual predictions**: additivity check (`sum(shap) + base_value == model.predict()`) passed **exactly** (max error 0.000000) on all three tasks.
- **No target columns**: `FEATURE_COLS` is the same leakage-safe list used throughout the project; target columns were never part of it.
- **No future features**: same reused feature list, nothing added.
- **Feature-name mapping**: `pre.get_feature_names_out()` output mapped to human-readable labels via `FRIENDLY_NAMES` (e.g. `cat__material_Timber` → "material: Timber"); confirmed correct in the printed output below.
- **Reproducibility**: `TreeExplainer` is exact (not sampling-based) for tree models — deterministic given a fixed model and input; the only randomness (test-row subsampling above 2,000 rows) uses a fixed seed.

## Global importance: current vs +5yr vs +10yr (top from SHAP mean|value|)
| Feature | Current | +5yr | +10yr |
|---|---|---|---|
| bridge age | 0.504 (#1) | **0.648 (#1)** | **0.611 (#1)** |
| previous overall condition | 0.359 (#2) | 0.150 | 0.105 |
| material: Timber | 0.270 (#3) | **0.425 (#2)** | **0.478 (#2)** |
| previous substructure condition | 0.245 | 0.127 | 0.117 |
| previous superstructure condition | 0.203 | 0.107 | 0.082 |
| years since rehabilitation | 0.157 | 0.127 | 0.108 |
| material: Steel | 0.061 | 0.100 | 0.114 |
| ever rehabilitated | not top-10 | not top-10 | 0.099 |
| annual rainfall | not top-10 | 0.066 | not top-10 |
| flood/scour exposure | 0.043 | 0.064 | 0.078 |

**Agrees with the models' built-in `feature_importances_`** from `docs/forecasting_results.md` — same top-2 features (bridge age, material Timber) in the same rising order as horizon extends, same collapse of "previous condition" from dominant (current) to secondary (+10yr). One honest nuance: SHAP surfaces rainfall/flood-exposure a bit more visibly (small but present in top-10) than the impurity-based `feature_importances_` did (<1%, not in top-10 there) — a known general property of SHAP vs. impurity-based importance, not a contradiction.

## Specifically-requested feature check
| Feature | Present & ranked how |
|---|---|
| bridge age | **#1 at every horizon**, grows with horizon |
| previous condition | **#2 current, but collapses** by +10yr — recent history matters short-term, not long-term |
| rehabilitation (years since / ever) | present at every horizon, moderate rank |
| material (Timber) | **#2-#3, grows with horizon** — see limitation below |
| traffic (ADT) | **not in top-10 at any horizon** |
| heavy vehicle traffic (adtt_percent / heavy_vehicle_traffic) | **not in top-10 at any horizon** |
| rainfall | appears only at +5yr, small magnitude |
| temperature | **not in top-10 at any horizon** |
| flood/scour exposure (proxy) | present at every horizon, consistently small |
| structure type | appears only at +5yr (Arch), small magnitude |

## Timber dominance — explicitly a synthetic-generator property
Material (especially Timber) is the #2-#3 global driver and grows with horizon. This is a **direct, intentional consequence of `DECAY_RATE["Timber"]=0.035` in `src/data/synthetic_generator.py`** (vs 0.008–0.015 for other materials) — **not a validated real-world finding about timber bridges.** Documented here explicitly per instructions, consistent with the same finding already flagged in `docs/forecasting_results.md` and `docs/rul_results.md`.

## Individual bridge explanation (`explain_bridge()`, reusable function)
Example — first current-condition test row, prediction 7.13:
```
Factors pushing condition UP (better):
  - bridge age (SHAP +0.371)
  - years since rehabilitation (SHAP +0.313)
  - previous overall condition (SHAP +0.258)
  - previous substructure condition (SHAP +0.175)
  - previous superstructure condition (SHAP +0.131)

Factors pushing condition DOWN (worse):
  - material: Steel (SHAP -0.224)
  - annual rainfall (SHAP -0.060)
  - flood/scour exposure (SHAP -0.026)
  - structure type: Arch (SHAP -0.025)
  - material: Masonry (SHAP -0.024)
```
**Important nuance**: "bridge age" appears here as a positive (condition-improving) contributor for *this specific bridge* — that's correct SHAP behavior, not an error. SHAP explains a row relative to the population baseline; a bridge younger than the average test-set bridge, in combination with its other features, can have age push its prediction *up* even though age is globally the strongest downward driver on average. Local and global explanations aren't required to point the same direction for every row — this is flagged so the human-readable text isn't mistaken for a universal rule.

## RUL / survival model — SHAP deliberately NOT used
`shap.TreeExplainer` doesn't apply (Weibull-AFT isn't a tree model). The only SHAP-compatible alternative would be `KernelExplainer` — a slow, sampling-based approximation. **Not used**, because the Weibull-AFT model already has an exact, closed-form, more-interpretable-than-SHAP explanation: its own regression coefficients (log-scale, directly convertible to hazard/time-ratios), already reported in `docs/rul_results.md`'s Cox hazard-ratio table. Approximating with `KernelExplainer` would add computation without adding rigor over what the model already provides natively — the right call given the task's explicit permission to skip SHAP where inappropriate, not a shortcut.

## Saved artifacts
`reports/shap_importance_current.png`, `reports/shap_importance_plus5.png`, `reports/shap_importance_plus10.png` (top-10 bar charts). `src/explainability/shap_explainer.py` provides `explain_bridge(task, feature_row)` → prediction + top positive/negative contributors + human-readable text, reusable for any single bridge-year observation.

## Limitations
- Explanations describe the **synthetic generator's** learned relationships, not real bridge engineering (repeated from every prior doc in this project — worth restating here since explainability output is the piece most likely to be mistaken for real domain knowledge if shown out of context).
- Traffic and temperature contribute negligibly at every horizon — a real deployment would need to know why before trusting "risk factor" output that omits them.
- RUL/survival explanations rely on the AFT model's own coefficients, not SHAP — consistent across the model but not directly comparable on the same scale as the condition models' SHAP values.
