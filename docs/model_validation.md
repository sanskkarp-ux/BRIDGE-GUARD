# Temporal Validation Strategy

Applies to the synthetic dataset (`docs/synthetic_data.md`). Random row splitting is never used — this is longitudinal panel data, and a random split would let a bridge's future rows leak into training alongside its past rows.

## Split (data-driven, not copied from the NBI plan)
Actual inspection-year range: **1995–2024**. Cumulative row-count-by-year was checked before picking cutoffs (not guessed):

| Split | Years | Rows | Bridges | % of rows |
|---|---|---|---|---|
| TRAIN | 1995–2019 | 23,852 | 4,910 | 70.6% |
| VAL | 2020–2021 | 4,154 | 4,154 | 12.3% |
| TEST | 2022–2024 | 5,800 | 4,966 | 17.2% |

Cutoffs (`TRAIN_END_YEAR=2019`, `VAL_END_YEAR=2021` in `feature_engineering.py`) were chosen from the cumulative-% table to land close to a standard 70/15/15 split on round years, with VAL sitting as a 2-year buffer between TRAIN and TEST.

## How the same bridge appearing in multiple splits is handled
Bridges are **not** partitioned into disjoint train/val/test sets — the same bridge_id legitimately has early rows in TRAIN and later rows in TEST, since that's what longitudinal deterioration data is. The leakage guard is not "keep bridges apart," it's:

1. **A row's features never contain information dated after that row's own year** (guaranteed by construction — see `docs/feature_engineering.md`).
2. **The model is fit only on rows whose `inspection_year` ≤ TRAIN_END_YEAR.** A bridge's later (VAL/TEST-period) row is never shown to the fitting process, even if that same bridge also has an earlier row inside TRAIN.
3. A TRAIN row's **target** (current/+5/+10) is allowed to be dated later than TRAIN_END_YEAR — that's normal supervised learning (a label is always "the future" relative to its features in a forecasting task) and is not leakage, since only a single scalar target value is read for that one row, never a full future feature vector re-entering the model as an input.

## Preprocessing leakage guard
`build_preprocessor()` (one-hot encoding + imputation) is `.fit()` **only on TRAIN rows**. VAL/TEST are `.transform()`-only. `OneHotEncoder(handle_unknown="ignore")` so a category that only appears in VAL/TEST doesn't require having been seen during fit.

## Tests (`tests/test_feature_engineering.py`)
Covers: no duplicate bridge-year rows, no negative bridge age, static fields consistent per bridge, lag features verified against actual chronological history (not just assumed), +5/+10 targets verified to reference strictly future years within their tolerance windows, train/val/test year ranges verified non-overlapping, and the preprocessor verified to fit once (train-only) and transform val/test without error.
