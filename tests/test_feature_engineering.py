"""Leakage and correctness checks for the feature-engineering pipeline.
Run: pytest tests/test_feature_engineering.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.feature_engineering import (
    build_features, build_targets, build_preprocessor, load_synthetic,
    CATEGORICAL_COLS, NUMERIC_COLS, BOOL_COLS, TRAIN_END_YEAR, VAL_END_YEAR,
)


@pytest.fixture(scope="module")
def raw():
    return load_synthetic()


@pytest.fixture(scope="module")
def features(raw):
    return build_features(raw)


@pytest.fixture(scope="module")
def targets(raw):
    return build_targets(raw)


def test_no_duplicate_bridge_year_rows(features):
    assert features.duplicated(subset=["bridge_id", "inspection_year"]).sum() == 0


def test_no_negative_bridge_age(features):
    assert (features["bridge_age"] >= 0).all()


def test_static_features_consistent(raw):
    static_cols = ["state", "material", "structure_type", "year_built", "road_class"]
    viol = (raw.groupby("bridge_id")[static_cols].nunique() > 1).sum()
    assert (viol == 0).all()


def test_lag_features_genuinely_lagged(raw, features):
    """prev_deck_condition at (bridge, Y) must equal that bridge's chronologically
    previous row's deck_condition -- never Y's own value or a future value."""
    merged = features.merge(
        raw[["bridge_id", "inspection_year", "deck_condition"]],
        on=["bridge_id", "inspection_year"],
    )
    sample = merged.groupby("bridge_id").sample(n=1, random_state=1)
    for _, row in sample.iterrows():
        history = raw[(raw["bridge_id"] == row["bridge_id"]) &
                       (raw["inspection_year"] < row["inspection_year"])]
        if history.empty:
            assert pd.isna(row["prev_deck_condition"])
        else:
            expected = history.sort_values("inspection_year").iloc[-1]["deck_condition"]
            assert row["prev_deck_condition"] == expected


def test_plus5_targets_refer_to_strict_future(raw, targets):
    have = targets[targets["has_plus5_target"]]
    assert (have["plus5_actual_gap_years"] > 0).all()
    assert have["plus5_actual_gap_years"].between(4, 6).all()


def test_plus10_targets_refer_to_strict_future(targets):
    have = targets[targets["has_plus10_target"]]
    assert (have["plus10_actual_gap_years"] > 0).all()
    assert have["plus10_actual_gap_years"].between(8, 12).all()


def test_no_future_features_via_year_gap(raw):
    """Sanity re-check on the raw generator invariant this pipeline relies on:
    a bridge's rehab info is never dated after its own inspection year."""
    valid = raw["years_since_rehab"].dropna()
    assert (valid >= 0).all()


def test_train_val_test_years_no_overlap(features):
    train_years = features.loc[features["split"] == "train", "inspection_year"]
    val_years = features.loc[features["split"] == "val", "inspection_year"]
    test_years = features.loc[features["split"] == "test", "inspection_year"]
    assert train_years.max() <= TRAIN_END_YEAR
    assert val_years.min() > TRAIN_END_YEAR and val_years.max() <= VAL_END_YEAR
    assert test_years.min() > VAL_END_YEAR
    assert train_years.max() < val_years.min()
    assert val_years.max() < test_years.min()


def test_preprocessor_fit_on_train_only_does_not_crash_on_val_test(features):
    cols = CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS
    train = features[features["split"] == "train"]
    val = features[features["split"] == "val"]
    test = features[features["split"] == "test"]

    pre = build_preprocessor()
    pre.fit(train[cols])
    val_t = pre.transform(val[cols])
    test_t = pre.transform(test[cols])
    assert val_t.shape[0] == len(val)
    assert test_t.shape[0] == len(test)
