"""Feature engineering + temporal split for the synthetic bridge dataset.

CRITICAL RULE: every feature for row (bridge, Y) must be computable using
only information dated at or before Y. See docs/feature_engineering.md and
docs/model_validation.md for the full design and rationale.

Features and targets are built and saved SEPARATELY (features.csv / targets.csv)
on purpose, so a target can never be accidentally merged into a feature set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SYNTH_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "synthetic"
RAW_PATH = SYNTH_DIR / "bridges_synthetic.csv"

TRAIN_END_YEAR = 2019   # inclusive
VAL_END_YEAR = 2021     # inclusive (val = TRAIN_END_YEAR+1 .. VAL_END_YEAR)
# test = VAL_END_YEAR+1 .. end of data

CATEGORICAL_COLS = ["state", "material", "structure_type", "road_class", "exposure_condition"]
NUMERIC_COLS = [
    "bridge_age", "years_since_rehab_filled", "adt", "adtt_percent", "heavy_vehicle_traffic",
    "annual_rainfall_mm", "avg_temp_c", "monsoon_intensity", "flood_risk_score",
    "num_spans", "max_span_m", "total_length_m", "deck_width_m", "num_lanes",
    "latitude", "longitude",
    "prev_deck_condition", "prev_superstructure_condition", "prev_substructure_condition",
    "prev_overall_condition", "overall_condition_trend", "num_previous_inspections",
    "years_since_first_inspection", "historical_deterioration_rate",
]
BOOL_COLS = ["ever_rehabilitated"]


def load_synthetic(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(["bridge_id", "inspection_year"]).reset_index(drop=True)


def assign_split(inspection_year: pd.Series) -> pd.Series:
    return pd.cut(
        inspection_year,
        bins=[-np.inf, TRAIN_END_YEAR, VAL_END_YEAR, np.inf],
        labels=["train", "val", "test"],
    ).astype(str)


def _add_historical_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("bridge_id")

    for col in ["deck_condition", "superstructure_condition", "substructure_condition", "overall_condition"]:
        df[f"prev_{col}"] = g[col].shift(1)
    df["prev2_overall_condition"] = g["overall_condition"].shift(2)
    df["overall_condition_trend"] = df["prev_overall_condition"] - df["prev2_overall_condition"]

    df["num_previous_inspections"] = g.cumcount()
    df["first_inspection_year"] = g["inspection_year"].transform("min")
    df["years_since_first_inspection"] = df["inspection_year"] - df["first_inspection_year"]

    first_overall = df.groupby("bridge_id")["overall_condition"].transform("first")
    prev_year = g["inspection_year"].shift(1)
    year_gap = (prev_year - df["first_inspection_year"]).replace(0, np.nan)
    df["historical_deterioration_rate"] = (first_overall - df["prev_overall_condition"]) / year_gap

    return df


def _add_rehab_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ever_rehabilitated"] = df["last_rehab_year"] > 0
    df["years_since_rehab_filled"] = np.where(
        df["years_since_rehab"].notna(), df["years_since_rehab"], df["bridge_age_years"]
    )
    return df


def _add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["heavy_vehicle_traffic"] = df["adt"] * df["adtt_percent"] / 100
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bridge_age"] = df["inspection_year"] - df["year_built"]
    assert (df["bridge_age"] == df["bridge_age_years"]).all(), "bridge_age recompute mismatch"

    df = _add_rehab_features(df)
    df = _add_traffic_features(df)
    df = _add_historical_condition_features(df)
    df["split"] = assign_split(df["inspection_year"])

    keep = (
        ["bridge_id", "inspection_year", "split"]
        + CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS
    )
    return df[keep]


def _nearest_future_match(bridge_df: pd.DataFrame, horizon: int, tol: int) -> pd.DataFrame:
    """For each row in a single bridge's chronological records, find the row whose
    inspection_year is nearest to (this row's year + horizon), within +/- tol,
    and strictly after this row's own year."""
    years = bridge_df["inspection_year"].to_numpy()
    out_year, out_gap, out_idx = [], [], []
    for y in years:
        lo, hi = y + horizon - tol, y + horizon + tol
        candidates = bridge_df[(bridge_df["inspection_year"] > y) &
                                (bridge_df["inspection_year"] >= lo) &
                                (bridge_df["inspection_year"] <= hi)]
        if candidates.empty:
            out_year.append(np.nan); out_gap.append(np.nan); out_idx.append(None)
            continue
        nearest = candidates.iloc[(candidates["inspection_year"] - (y + horizon)).abs().argsort().iloc[0]]
        out_year.append(nearest["inspection_year"])
        out_gap.append(nearest["inspection_year"] - y)
        out_idx.append(nearest.name)
    result = pd.DataFrame({"matched_year": out_year, "actual_gap": out_gap})
    result["matched_idx"] = pd.array(out_idx, dtype=object)
    return result


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = assign_split(df["inspection_year"])
    cond_cols = ["deck_condition", "superstructure_condition", "substructure_condition", "overall_condition"]

    out_rows = []
    for bridge_id, bdf in df.groupby("bridge_id", sort=False):
        bdf = bdf.sort_values("inspection_year")
        m5 = _nearest_future_match(bdf, horizon=5, tol=1)
        m10 = _nearest_future_match(bdf, horizon=10, tol=2)
        m5.index = bdf.index
        m10.index = bdf.index

        for pos, idx in enumerate(bdf.index):
            row = {"bridge_id": bridge_id, "inspection_year": bdf.loc[idx, "inspection_year"],
                   "split": bdf.loc[idx, "split"]}
            for c in cond_cols:
                row[f"target_current_{c}"] = bdf.loc[idx, c]

            m5_idx = m5.loc[idx, "matched_idx"]
            row["has_plus5_target"] = m5_idx is not None
            row["plus5_actual_gap_years"] = m5.loc[idx, "actual_gap"]
            for c in cond_cols:
                row[f"target_plus5_{c}"] = df.loc[m5_idx, c] if m5_idx is not None else np.nan

            m10_idx = m10.loc[idx, "matched_idx"]
            row["has_plus10_target"] = m10_idx is not None
            row["plus10_actual_gap_years"] = m10.loc[idx, "actual_gap"]
            for c in cond_cols:
                row[f"target_plus10_{c}"] = df.loc[m10_idx, c] if m10_idx is not None else np.nan

            out_rows.append(row)

    return pd.DataFrame(out_rows)


def build_preprocessor() -> ColumnTransformer:
    """Fit ONLY on train rows. handle_unknown='ignore' so a category never seen
    in train (e.g. only appears in val/test) doesn't crash transform -- it's
    encoded as all-zeros instead of leaking a new category definition backward."""
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer([
        ("cat", categorical, CATEGORICAL_COLS),
        ("num", numeric, NUMERIC_COLS + BOOL_COLS),
    ])


def main():
    raw = load_synthetic()
    features = build_features(raw)
    targets = build_targets(raw)

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(SYNTH_DIR / "features.csv", index=False)
    targets.to_csv(SYNTH_DIR / "targets.csv", index=False)

    train = features[features["split"] == "train"]
    val = features[features["split"] == "val"]
    test = features[features["split"] == "test"]

    pre = build_preprocessor()
    pre.fit(train[CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS])
    pre.transform(val[CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS])
    pre.transform(test[CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS])

    print("=== FEATURE LIST ===")
    print(CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS)
    print()
    print("=== TARGET LIST ===")
    print([c for c in targets.columns if c.startswith("target_")])
    print()
    print(f"TRAIN years: <= {TRAIN_END_YEAR} | rows={len(train)} bridges={train['bridge_id'].nunique()}")
    print(f"VAL years: {TRAIN_END_YEAR+1}-{VAL_END_YEAR} | rows={len(val)} bridges={val['bridge_id'].nunique()}")
    print(f"TEST years: > {VAL_END_YEAR} | rows={len(test)} bridges={test['bridge_id'].nunique()}")
    print()
    print("usable +5 targets:", int(targets["has_plus5_target"].sum()), "/", len(targets))
    print("usable +10 targets:", int(targets["has_plus10_target"].sum()), "/", len(targets))
    print()
    print("=== MISSING VALUES (features) ===")
    print((features.isnull().mean() * 100).round(1)[lambda s: s > 0])
    print()
    print("preprocessor fit-on-train-only check: OK (fit called once, on train rows only)")


if __name__ == "__main__":
    main()
