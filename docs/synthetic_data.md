# Synthetic Data — Design & Disclaimer

> **Synthetic India-inspired data — for prototype/showcase purposes. Not real inspection data.**
> This dataset does not represent real Indian bridges, real inspection records, or real deterioration statistics. Every generated row carries this disclaimer in its own `data_source` column so it travels with the data wherever it's used.

BridgeGuard AI's current scope is a showcase/prototype (see project scope-lock decision). Real Indian bridge data (Uttarakhand PWD InfraMgt — see `docs/india_data_strategy.md`) is a future goal pending official permission, not used here.

## Generator
`src/data/synthetic_generator.py`, seed = 42 (reproducible), run via `python src/data/synthetic_generator.py`.

- **5,000 unique bridges**, 5–10 inspection years each (younger bridges get fewer, by design — not padded to a fixed count)
- **Actual output**: 33,806 bridge-year records (within the 25k–50k target)
- Outputs:
  - `data/processed/synthetic/bridges_synthetic.csv` — the modeling-safe panel
  - `data/processed/synthetic/bridges_synthetic_ground_truth.csv` — **eval-only**, never a model input
  - `data/processed/synthetic_sample.csv` — 50-row committed sample for reference

## Design summary (A–J)

**A. Variables**: bridge_id, state (15 Indian states spanning arid/coastal/hilly/flood-prone climate zones), district (synthetic label, not a real place), lat/long, road_class, year_built, material, structure_type, spans/lengths/widths/lanes, ADT/ADTT%, rainfall/temperature/monsoon-intensity/flood-risk, exposure_condition, last_rehab_year, inspection_year, deck/superstructure/substructure/overall condition (0–9).

**B. Distributions**: year_built triangular(1960, 2005, 2022); material conditioned on year_built (older skews masonry/timber); ADT lognormal by road_class; climate fields = per-state mean ± bridge-level noise (15-state lookup table with rainfall/temp/flood-risk/coastal flag).

**C. Longitudinal structure**: stable `bridge_id`; 5–10 inspections spaced 2–3 years apart, ending ≤2024, never before `year_built`.

**D/E. Condition & deterioration**: a hidden per-bridge `_latent_quality_factor` (N(0,1), unobserved heterogeneity — construction quality, local conditions not captured by any feature) combines with material decay rate × traffic factor × climate factor into a stress term; condition follows `9·exp(−stress)`. Deck/superstructure/substructure get correlated but distinct values (component offsets + independent noise). Rehab resets the deck's effective age more than the substructure's.

**F. RUL (ground truth only)**: solved analytically as years until the *unrounded* latent trajectory crosses condition ≤ 4, holding the stress-rate constant forward (a documented simplification — real future traffic/climate/decay won't stay constant).

**Correction (post-validation)**: the first version censored past an arbitrary fixed cap of 100 years, disconnected from the dataset's actual timeframe — with build years 1960–2022 and observation ending 2024 (≤64-year window), almost nothing could ever exceed 100, producing only 0.02% censored. Fixed to the textbook definition: a bridge is right-censored if its projected threshold-crossing year (`year_built + age_at_threshold`) falls **after `CURRENT_YEAR` (2024)**, i.e. it genuinely hasn't reached "poor" by the time we stop observing it. Result: **84.6% of rows / 86.1% of bridges are censored.** This is high but not forced — it falls directly out of `year_built` skewing toward 1990–2022 (most bridges are simply too young to have deteriorated to threshold yet within the observation window). Flagged as a real design tension: realistic for a young/expanding highway network, but it leaves relatively few "observed event" rows (~5,210) for any future RUL model to learn from.

**G. Noise/missing data**: ±1 inspector rating jitter, heteroscedastic noise (older bridges rated less consistently), 5–10% missingness on ADT/ADTT%/climate columns. The latent quality factor is the main defense against unrealistic near-100% model accuracy later — it's real variance no feature can explain.

**H. Leakage prevention**: ground-truth RUL and the latent factor live in a **separate file**, keyed by bridge_id + inspection_year, explicitly not merged into `bridges_synthetic.csv`. `years_since_rehab` only ever uses rehab years ≤ the inspection year.

**I. ML targets** (future phases, not computed here): deck/superstructure/substructure/overall condition are ready now; future condition at t+5/t+10 and predicted RUL will be derived at modeling time using proper temporal splits — not baked into the generator.

**J. Dashboard outputs** (future phases): health score, current + forecasted condition, 5/10/15-yr risk, RUL with interval, SHAP-based risk factors, and a persistent synthetic-data disclaimer banner.

## Validation pass (post-fix)
Also fixed during validation: `main()`'s merge duplicated `adt`/`adtt_percent`/climate columns into `_x`/`_y` pairs (they were generated in both the per-inspection records and the bridge-static table). Fixed by dropping them from the bridge-static side before merging — single source of truth now.

Checked and clean: 33,806 rows × 30 columns, 4,994 unique bridges (6 of the intended 5,000 produced no valid inspection year and were dropped — a young-bridge edge case, not a bug), 1–10 records/bridge (mean 6.8), no duplicate bridge-year rows, no out-of-range condition values, no negative ages, static fields (state/material/year_built/etc.) provably constant per bridge across years, and condition columns do genuinely vary within a bridge over time (not constant).

## What this is NOT
- Not real Indian bridge inspection data
- Not real Indian bridge deterioration statistics
- Not validated against any real-world model accuracy
- Not deployment-ready or a certified structural safety assessment
- The generating relationships (decay rates, climate multipliers, etc.) are plausibility-tuned by the developer, not fitted to any real dataset
