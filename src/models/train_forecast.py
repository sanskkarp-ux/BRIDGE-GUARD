"""Multi-horizon (+5yr, +10yr) overall-condition forecasting.

Synthetic India-inspired dataset; model performance is a prototype
demonstration and is not evidence of real-world Indian bridge prediction
accuracy.

Leakage rule: features use only information at/before year Y (same
features.csv as the current-condition baseline). The target comes from
Y+5 / Y+10 -- that's the label, not a feature, so it being "from the
future" is normal supervised learning, not leakage (see docs/model_validation.md).
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import (
    CATEGORICAL_COLS, NUMERIC_COLS, BOOL_COLS, TRAIN_END_YEAR, VAL_END_YEAR,
    build_preprocessor, SYNTH_DIR,
)

SEED = 42
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS

# Horizon-specific split cutoffs. The GLOBAL split (train<=2019/val 2020-21/test 2022-24,
# from feature_engineering.py) cannot be reused here: rows with a usable +5/+10 target are
# structurally capped well before 2024 (a +10yr target needs a same-bridge record up to 10
# years later, and the dataset ends 2024) -- confirmed empirically: has_plus5_target rows
# only go up to inspection_year=2020, has_plus10_target rows only up to 2016. Reusing the
# global split would leave val/test with ZERO usable rows for these horizons (this is what
# the first run of this script hit -- an actual bug, not a style choice, so a horizon-specific
# split is used instead, chosen the same way as the global one: from the real cumulative
# row-count-by-year distribution of rows that actually have a usable target).
PLUS5_TRAIN_END, PLUS5_VAL_END = 2014, 2017    # feasible range 1995-2020; ~60/24/17% split
PLUS10_TRAIN_END, PLUS10_VAL_END = 2012, 2014  # feasible range 1995-2016; ~64/22/14% split

try:
    import xgboost
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_data():
    features = pd.read_csv(SYNTH_DIR / "features.csv")
    targets = pd.read_csv(SYNTH_DIR / "targets.csv")
    return features.merge(targets, on=["bridge_id", "inspection_year", "split"])


def metrics(y_true, y_pred):
    pred_rounded = np.clip(np.round(y_pred), 0, 9)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "ExactAcc": float((pred_rounded == y_true).mean()),
        "Within1Acc": float((np.abs(pred_rounded - y_true) <= 1).mean()),
    }


def train_horizon(df, target_col, has_flag_col, horizon_label, train_end, val_end):
    d = df[df[has_flag_col]].copy()
    train = d[d["inspection_year"] <= train_end]
    val = d[(d["inspection_year"] > train_end) & (d["inspection_year"] <= val_end)]
    test = d[d["inspection_year"] > val_end]

    pre = build_preprocessor()
    X_train = pre.fit_transform(train[FEATURE_COLS])
    X_val = pre.transform(val[FEATURE_COLS])
    X_test = pre.transform(test[FEATURE_COLS])
    y_train, y_val, y_test = train[target_col].values, val[target_col].values, test[target_col].values

    models = {}
    models["MedianBaseline"] = ("baseline", np.median(y_train))
    models["Ridge"] = ("model", Ridge(alpha=1.0, random_state=SEED).fit(X_train, y_train))
    models["RandomForest"] = ("model", RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1).fit(X_train, y_train))
    models["GradientBoosting"] = ("model", GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=SEED).fit(X_train, y_train))
    if HAS_XGB:
        models["XGBoost"] = ("model", xgboost.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED,
            n_jobs=-1).fit(X_train, y_train))

    results = {}
    for name, (kind, obj) in models.items():
        preds = {}
        for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = np.full_like(y, fill_value=obj, dtype=float) if kind == "baseline" else obj.predict(X)
            preds[split_name] = metrics(y, pred)
        results[name] = preds

    print(f"\n=== {horizon_label} : rows train={len(train)} val={len(val)} test={len(test)} ===")
    for name, preds in results.items():
        for split_name in ["train", "val", "test"]:
            m = preds[split_name]
            print(f"{name:16s} {split_name:5s} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
                  f"ExactAcc={m['ExactAcc']:.3f} Within1={m['Within1Acc']:.3f}")

    best_name = min((n for n in results if n != "MedianBaseline"), key=lambda n: results[n]["val"]["MAE"])
    best_kind, best_obj = models[best_name]
    print(f"BEST ({horizon_label}) by val MAE: {best_name} -> test: {results[best_name]['test']}")

    test_pred_rounded = np.clip(np.round(best_obj.predict(X_test)), 0, 9).astype(int)
    cm = confusion_matrix(y_test, test_pred_rounded, labels=list(range(10)))
    print(f"Confusion matrix ({horizon_label}, test):\n{cm}")

    feature_names = pre.get_feature_names_out()
    importances = []
    if hasattr(best_obj, "feature_importances_"):
        importances = sorted(zip(feature_names, best_obj.feature_importances_), key=lambda x: -x[1])[:10]
    elif hasattr(best_obj, "coef_"):
        importances = sorted(zip(feature_names, np.abs(best_obj.coef_)), key=lambda x: -x[1])[:10]
    print(f"Top-10 importances ({horizon_label}):")
    for f, v in importances:
        print(f"  {f:40s} {float(v):.4f}")

    dump(best_obj, MODELS_DIR / f"best_model_{horizon_label}.joblib")
    dump(pre, MODELS_DIR / f"preprocessor_{horizon_label}.joblib")

    return {
        "horizon": horizon_label, "best_model": best_name,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "results": results,
        "top_importances": [(f, float(v)) for f, v in importances],
    }


def main():
    df = load_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary5 = train_horizon(df, "target_plus5_overall_condition", "has_plus5_target", "plus5",
                              PLUS5_TRAIN_END, PLUS5_VAL_END)
    summary10 = train_horizon(df, "target_plus10_overall_condition", "has_plus10_target", "plus10",
                               PLUS10_TRAIN_END, PLUS10_VAL_END)

    metadata = {
        "disclaimer": "Synthetic India-inspired dataset; model performance is a prototype "
                       "demonstration and is not evidence of real-world Indian bridge prediction accuracy.",
        "seed": SEED,
        "feature_list": FEATURE_COLS,
        "train_years": f"<= {TRAIN_END_YEAR}", "val_years": f"{TRAIN_END_YEAR+1}-{VAL_END_YEAR}",
        "test_years": f"> {VAL_END_YEAR}",
        "plus5": summary5, "plus10": summary10,
        "package_versions": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
            "xgboost": xgboost.__version__ if HAS_XGB else "not installed",
        },
    }
    with open(MODELS_DIR / "forecast_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    print(f"\nSaved metadata: {MODELS_DIR / 'forecast_metadata.json'}")


if __name__ == "__main__":
    main()
