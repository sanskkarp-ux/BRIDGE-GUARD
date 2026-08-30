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

**F. RUL (ground truth only)**: solved analytically as years until the *unrounded* latent trajectory crosses condition ≤ 4, holding the stress-rate constant forward (a documented simplification — real future traffic/climate/decay won't stay constant). Right-censored past 100 years. **Observed result: 0% censored** in this run — with build years 1960–2022 and the current decay-rate ranges, every bridge's projected threshold-crossing age falls under 100 years, so the censoring mechanism exists in the code but didn't trigger on this dataset. Documented honestly rather than re-tuned to force censored cases.

**G. Noise/missing data**: ±1 inspector rating jitter, heteroscedastic noise (older bridges rated less consistently), 5–10% missingness on ADT/ADTT%/climate columns. The latent quality factor is the main defense against unrealistic near-100% model accuracy later — it's real variance no feature can explain.

**H. Leakage prevention**: ground-truth RUL and the latent factor live in a **separate file**, keyed by bridge_id + inspection_year, explicitly not merged into `bridges_synthetic.csv`. `years_since_rehab` only ever uses rehab years ≤ the inspection year.

**I. ML targets** (future phases, not computed here): deck/superstructure/substructure/overall condition are ready now; future condition at t+5/t+10 and predicted RUL will be derived at modeling time using proper temporal splits — not baked into the generator.

**J. Dashboard outputs** (future phases): health score, current + forecasted condition, 5/10/15-yr risk, RUL with interval, SHAP-based risk factors, and a persistent synthetic-data disclaimer banner.

## What this is NOT
- Not real Indian bridge inspection data
- Not real Indian bridge deterioration statistics
- Not validated against any real-world model accuracy
- Not deployment-ready or a certified structural safety assessment
- The generating relationships (decay rates, climate multipliers, etc.) are plausibility-tuned by the developer, not fitted to any real dataset
