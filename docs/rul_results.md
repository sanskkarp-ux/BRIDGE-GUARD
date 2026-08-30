# RUL / Survival Analysis Results

> **Synthetic India-inspired dataset; model performance is a prototype demonstration and is not evidence of real-world Indian bridge prediction accuracy.**

## Event definition (reused, not invented)
**overall_condition ≤ 4** — the same "poor" threshold already fixed in `docs/synthetic_data.md` (matching FHWA's condition-measure convention from the Phase-1 research). Not redefined here.

## How survival data was built (`src/models/survival.py::build_survival_dataset`)
Built from **observed** inspection records (`bridges_synthetic.csv`), not the hidden ground-truth latent trajectory (`bridges_synthetic_ground_truth.csv` stays eval-only, as always). Per bridge:
- **start_year** = bridge's first inspection year (t=0 reference)
- **event=1** if any observed inspection ever recorded overall_condition ≤ 4; **event_year** = the first such year; **duration** = event_year − start_year
- **event=0** (censored) otherwise; **duration** = last observed inspection year − start_year (censored at the bridge's own last observation, the standard convention — not a single global cutoff)
- duration=0 (event at the very first inspection) is bumped to 0.5yr — Weibull-AFT requires strictly positive durations; applied identically to Cox for comparability

**Result**: 4,994 bridges — **1,338 events (26.8%)**, **3,656 censored (73.2%)**. This differs from the ground-truth file's 86% censoring, and correctly so: that number came from an "omniscient" latent-trajectory projection extended past the actual data; this number comes from what real inspection records would actually show, which is what a deployable model must work from.

Covariates: baseline (first-inspection) attributes only — material, structure_type, road_class, exposure_condition, ever_rehabilitated, bridge_age_at_start, adt, adtt_percent, annual_rainfall_mm, avg_temp_c, monsoon_intensity, flood_risk_score, num_spans, max_span_m, total_length_m, deck_width_m, num_lanes. **`prev_overall_condition` and other lag/history features are deliberately excluded** — they're undefined at a bridge's first inspection (our t=0), so including them wasn't possible for this baseline design. Numeric features standardized (train-fit mean/std, train-only, same rule as every other model in this project); categoricals one-hot encoded.

## Kaplan-Meier (population-level)
S(5yr)=0.870, S(10yr)=0.811, S(15yr)=0.731, S(20yr)=0.676, S(30yr)=0.473 (plateaus after). **Population median survival time: 27 years.**

## Cox PH — fit, but its core assumption is violated
Fit successfully (concordance 0.963–0.967 across train/val/test). But `check_assumptions()` flagged **the three most important covariates** — `bridge_age_at_start`, `material_Timber`, `ever_rehabilitated` — as failing the proportional-hazards test (p<0.05, `bridge_age_at_start` and `ever_rehabilitated` at p<5e-05). This is expected, not a fitting error: the generator's true process (`condition = 9·exp(−age·k)`, threshold-crossing age = ln(9/4)/k) is an **accelerated-failure-time** structure — covariates rescale *time*, not a constant hazard *ratio* — which Cox's proportional-hazards assumption doesn't match well for age in particular.

## Weibull-AFT — used as the more appropriate alternative
Per the task's explicit fallback instruction, fit `WeibullAFTFitter` on the same covariates (concordance 0.964–0.968 — comparable to Cox, slightly better, and doesn't require the violated PH assumption). **This is the model used for individual RUL estimates.**

## IMPORTANT: concordance ~0.96 is suspiciously high — investigated
Real bridge-survival studies typically report concordance in the 0.7–0.85 range. ~0.96 here was investigated rather than reported as a win:
- **Not a leakage bug**: all covariates are the bridge's own first-inspection attributes; none are derived from the event, from a later inspection, or from the hidden latent factor.
- **Cause**: the synthetic generator's condition trajectory is a low-noise, near-deterministic function of exactly age + material (+ smaller traffic/climate terms) — the same pattern already documented in `docs/forecasting_results.md`, where age and `material_Timber` dominated +5/+10yr condition-forecast importance (up to 55%). A survival model fed the same dominant covariates on the same generator will naturally separate bridges very cleanly.
- **Conclusion**: this reflects an **overly clean synthetic generator**, not real-world predictive power. Documented as a limitation, not a result to celebrate — matches the task's instruction directly.

## Individual bridge RUL — honesty over fake precision
Example (first test-set bridge, Weibull-AFT): **"Median RUL: not reached within observed horizon."** Its survival probability never dropped below 0.5 within the fitted model's horizon, so no median is reported — not fabricated. Same rule applies to the 25th–75th percentile range: reported only when the survival curve actually crosses those probabilities within the observed range, otherwise explicitly stated as not estimable. No output of the form "RUL = 11.347 years" is ever produced — `median_and_range()` in `survival.py` only returns whole-year values or `None`.

## Feature-influence check (Step 5, Cox hazard ratios — sign/magnitude, not real-world claims)
| Feature | In model? | Effect direction | Notes |
|---|---|---|---|
| bridge age | ✅ | ↑ hazard strongly (HR≈4.8) | dominant, as expected — PH-violated, AFT handles it better |
| material (esp. Timber) | ✅ | Timber HR≈38 (!), Steel≈5.9, RCC≈1.9 vs Masonry baseline | matches the generator's intentionally fastest Timber decay rate — **synthetic artifact, not a real-world material finding** |
| previous condition | ❌ not included | — | undefined at t=0 (first inspection); a future revision could re-frame as time-varying Cox if needed |
| rehabilitation history | ✅ (`ever_rehabilitated`) | protective (HR≈0.60, p<1e-12) | correctly reduces hazard — sanity-check passed |
| ADT | ✅ | not significant (p=0.31) | negligible, consistent with forecasting-phase finding |
| heavy vehicle traffic (adtt_percent) | ✅ | small but significant (HR≈1.24, p<1e-10) | present, unlike in the condition-forecast models — a genuine (if modest) difference worth noting |
| rainfall | ✅ | small but significant (HR≈1.21, p<1e-6) | same — survival framing picks up a bit more climate signal than point-condition regression did |
| temperature | ✅ | not significant | negligible |
| flood exposure (flood_risk_score) | ✅ | not significant (p=0.59) | negligible |
| scour exposure | proxy only | — | no dedicated scour field exists (same limitation as `docs/synthetic_data.md`); flood_risk_score serves as the only proxy |
| structure type | ✅ | Truss/Girder/Slab/Box all elevated vs baseline | secondary to material |

**Timber dominance confirmed as generator-driven**: HR≈38 is an extreme, unmistakable reflection of the generator's `DECAY_RATE["Timber"]=0.035` vs 0.008–0.015 for everything else — explicitly a synthetic-data property, not evidence about real timber bridges.

## Validation summary
Concordance index used throughout (not classification accuracy) — Cox and Weibull-AFT both ~0.96–0.97 across train/val/test with no meaningful train→test drop (no overfitting), but the absolute level is flagged as unrealistically high per the investigation above.

## Saved artifacts
`models/cox_survival_model.joblib`, `models/weibull_aft_survival_model.joblib`, `models/kaplan_meier_curve.csv`.

## Limitations
- Concordance is inflated by generator determinism, not real-world signal strength.
- No lag/history covariates (design limitation of the first-inspection-baseline framing).
- Scour has no dedicated field, only a flood-risk proxy.
- PH assumption violated for Cox on its top covariates — Weibull-AFT preferred for individual estimates, but AFT's own distributional (Weibull-shape) assumption hasn't been separately stress-tested.
- Random bridge-level 70/15/15 split (seed=42) used here, not the row-level temporal split from earlier phases — justified because this is a population-level model (one row per bridge, not a longitudinal forecast), so the earlier row-leakage concern doesn't apply the same way; documented as a deliberate, different design choice, not an oversight.
