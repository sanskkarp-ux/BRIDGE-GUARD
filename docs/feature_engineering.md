# Feature Engineering

Pipeline: `src/features/feature_engineering.py`. Operates on the synthetic dataset (`docs/synthetic_data.md`) — not real data.

## Core rule
For a row (bridge B, year Y), **no feature may use information dated after Y**: no condition from Y+1 or later, no future traffic/rehab, no future-derived statistic. Features and targets are built by separate functions and saved to **separate files** (`features.csv`, `targets.csv`) precisely so a target can never be accidentally used as a model input.

## Feature list

**Categorical**: state, material, structure_type, road_class, exposure_condition
**Structural (static)**: num_spans, max_span_m, total_length_m, deck_width_m, num_lanes, latitude, longitude
**Age/rehab**: `bridge_age` (= inspection_year − year_built, cross-checked against the generator's own `bridge_age_years`), `ever_rehabilitated` (bool), `years_since_rehab_filled` (real years-since-rehab where rehabbed, else = bridge_age — a never-rehabbed bridge's "time since rehab" is defined as its whole life, same convention used for NBI's Item 106 = 0 in the earlier US-data phase)
**Traffic**: adt, adtt_percent, `heavy_vehicle_traffic` (= adt × adtt_percent/100)
**Environmental**: annual_rainfall_mm, avg_temp_c, monsoon_intensity, flood_risk_score (also serves as the scour-exposure proxy — no separate scour field exists in the synthetic generator, documented rather than duplicated as a fake distinct column)
**Historical condition** (all computed from strictly earlier rows of the same bridge via `groupby().shift()`): `prev_deck/superstructure/substructure/overall_condition` (lag-1), `overall_condition_trend` (prev − prev2), `num_previous_inspections` (count of rows before this one), `years_since_first_inspection`, `historical_deterioration_rate` (average yearly decline from the bridge's first-ever inspection up to its most recent past inspection)

30 features total. First inspection of a bridge has no history, so lag/trend/rate features are legitimately NaN there (~15–30% missingness, not a bug — see report below).

## Targets (kept separate from features)

- **Target A (current)**: `target_current_{deck,superstructure,substructure,overall}_condition` — this row's own condition, predicted from the other columns.
- **Target B (+5yr)**: nearest same-bridge future record within [Y+4, Y+6]; `has_plus5_target` flag + `plus5_actual_gap_years` for QA.
- **Target C (+10yr)**: nearest same-bridge future record within [Y+8, Y+12]; same flag/gap pattern.

Usable +5 targets: 23,867 / 33,806 rows. Usable +10 targets: 16,550 / 33,806 — fewer, as expected, since a +10 match needs a bridge to still be under observation a decade later.

## Preprocessing
`build_preprocessor()`: one-hot encode categoricals (median/most-frequent imputation for missing values), `handle_unknown="ignore"` so a category unseen in train doesn't crash on val/test — it's just encoded as all-zeros rather than requiring the encoder to have seen it. **Fit only on train rows**; val/test are transformed with the train-fitted object, never refit.
