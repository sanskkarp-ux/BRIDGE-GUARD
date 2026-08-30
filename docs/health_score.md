# BridgeGuard Prototype Health Score

> **"BridgeGuard Prototype Health Score" — a composite ML-derived risk indicator on synthetic India-inspired data. NOT a certified structural safety rating, NOT an official Indian bridge condition rating, NOT a replacement for physical inspection, NOT evidence of real-world Indian bridge performance.**

## Formula
```
health_score = round(
    0.40 * (current_condition / 9 * 100)
  + 0.20 * (predicted_condition_5yr / 9 * 100)
  + 0.15 * (predicted_condition_10yr / 9 * 100)
  + 0.25 * (survival_10yr_probability * 100)
)
```
clipped to [0, 100]. Missing components → remaining weights renormalized to sum to 1 (never silently treated as 0 or 100). Deterministic: identical inputs always give the identical output (no randomness anywhere in `compute_health_score()`).

## Components and why each is included
| Component | Weight | Why |
|---|---|---|
| Current condition | 40% | Highest-reliability signal in the project — lowest model error of anything built (test MAE 0.405, `docs/model_results.md`) |
| +5yr predicted condition | 20% | Genuinely different-horizon information, but reduced weight — SHAP (`docs/explainability.md`) shows the same underlying age/material signal drives current and +5yr predictions, so full independent weight would double-count it |
| +10yr predicted condition | 15% | Same reasoning, weighted even lower — SHAP shows prior-condition's influence keeps shrinking at +10yr, AND the +10yr model has the highest error of the three point-predictors (test MAE 0.464) |
| Survival 10yr probability | 25% | A structurally different model (baseline-only covariates, different construction) — real independent evidence, not pure double-counting — but capped below current_condition's weight because its ~0.96 concordance was explicitly flagged in `docs/rul_results.md` as likely inflated by generator determinism, not proof of superior reliability |

Combined weight on the three condition-level components is 40+20+15 = **75%, not 120%** — the deliberate anti-double-counting choice: three fully-independent full-weight signals would over-count the shared age/material driver three times.

## Normalization
- Condition (0–9) → 0–100 linearly: `(condition/9)*100`. Same direction as the NBI-derived scale used throughout this project (9=excellent→100, 0=worst→0).
- Survival probability (already 0–1) → 0–100 directly, no invented curve.

## A design bug found and fixed via the required validation testing
First version used **decline magnitude** (`current − predicted_5yr`) for the trend components, not absolute level. This masked absolute badness: a bridge already at condition 2–3 (already below this project's own poor-threshold of ≤4) that declined only slightly further still scored a "stable" 75–100 on the trend components, landing in **"Fair" category** — clearly wrong. Caught by testing example bridges #2 (old/high-risk) and #4 (poor-condition) exactly as instructed. **Fixed by using each horizon's absolute predicted level instead of decline magnitude** — the weight tapering (40/20/15%) already handles anti-double-counting, so switching to absolute levels doesn't reintroduce it; it just stops hiding badness. After the fix: bridge #2 → 28 (Poor), bridge #4 → 18 (Critical) — correctly reflecting their current status. No model was changed to fix this — only the scoring formula.

## Category thresholds
| Score | Category |
|---|---|
| 81–100 | Excellent |
| 61–80 | Good |
| 41–60 | Fair |
| 21–40 | Poor |
| 0–20 | Critical |

**Not blindly copied from the suggested Excellent/Good/Moderate/Poor/Critical bands** — evaluated and adjusted: the middle category is named **"Fair"**, not "Moderate," because it directly matches IRC:SP:35 — India's own bridge-inspection guideline (`docs/india_data_strategy.md` research) — which already uses an Excellent/Good/Fair/Poor/Critical 5-category scale. Reusing real Indian engineering terminology for the *labels* is defensible (a label isn't a data-accuracy claim), while the underlying *score* stays clearly marked synthetic. Even 20-point bands were kept (not recalibrated to percentiles of the population) because there's no real ground-truth category survey to calibrate against for this prototype — forcing a fancier calibration would fabricate precision the project doesn't have.

## Uncertainty handling
`confidence` field: `"high"` (all 4 components available), `"medium"` (1 missing), `"low"` (2+ missing). This is never hidden — a score computed from partial information is explicitly flagged as lower-confidence, not presented with the same certainty as a full-information score (example #5 below: 2 of 4 components missing → confidence="low").

## Example bridges (validation)
| Bridge | Current | +5yr | +10yr | S(10yr) | Score | Category | Confidence |
|---|---|---|---|---|---|---|---|
| 1. Healthy/new | 8 | 8 | 7 | 0.95 | **89** | Excellent | high |
| 2. Old/high-risk | 3 | 2 | 1 | 0.35 | **28** | Poor | high |
| 3. Recently rehabbed | 7 | 7 | 6 | 0.85 | **78** | Good | high |
| 4. Poor-condition | 2 | 1 | 1 | 0.20 | **18** | Critical | high |
| 5. Uncertain RUL (missing +10yr, missing survival) | 6 | 5 | — | — | **63** | Good | **low** |

## Validation results (all passed)
- All scores remain in [0, 100]: ✅
- Healthy bridge scores higher than old/high-risk and poor-condition bridges: ✅
- Recently-rehabbed scores higher than old/high-risk (rehab's benefit shows up): ✅
- Worsening predicted condition lowers the score (bridge 2 < bridge 1): ✅
- Bridges already below the poor-threshold correctly land in Poor/Critical, not Fair+ (regression check for the bug above): ✅
- Missing-data example shows reduced confidence, not false certainty: ✅

## Limitations
- Weights are reasoned from this project's own documented error/reliability findings, not fitted or optimized against any ground truth (none exists for a prototype health score) — they are a defensible, documented judgment call, not a proven-optimal formula.
- Category thresholds are even bands, not calibrated to a real population distribution.
- Survival-probability integration (`get_survival_10yr_probability()`) recomputes normalization statistics from the full synthetic population rather than the exact original train-only split used when the Weibull-AFT model was fit — a documented, negligible approximation (normalization stats only, not the model itself — no retraining occurred) rather than bit-exact reproduction.
- This score inherits every limitation already documented for its inputs: synthetic-generator determinism inflating the survival component's apparent reliability (`docs/rul_results.md`), Timber's outsized influence being a generator artifact (`docs/synthetic_data.md`), and traffic/climate features contributing little to any underlying model (`docs/forecasting_results.md`, `docs/explainability.md`).
- **Not a certified structural safety rating. Not an official Indian bridge condition rating. Not a replacement for physical inspection. Not evidence of real-world Indian bridge performance.**
