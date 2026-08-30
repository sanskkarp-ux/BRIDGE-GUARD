"""Baseline modeling: predict target_current_overall_condition.

Synthetic India-inspired dataset; model performance is a prototype
demonstration and is not evidence of real-world Indian bridge prediction
accuracy.

Uses the leakage-safe features/preprocessor from src/features/feature_engineering.py
and the pre-built features.csv/targets.csv (already generated, no need to
rebuild the slow target-matching step).
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
TARGET = "target_current_overall_condition"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DOCS_PATH = Path(__file__).resolve().parents[2] / "docs" / "model_results.md"
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS

try:
    import xgboost
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_data():
    features = pd.read_csv(SYNTH_DIR / "features.csv")
    targets = pd.read_csv(SYNTH_DIR / "targets.csv")[["bridge_id", "inspection_year", TARGET]]
    df = features.merge(targets, on=["bridge_id", "inspection_year"])
    return df


def metrics(y_true, y_pred):
    pred_rounded = np.clip(np.round(y_pred), 0, 9)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "ExactAcc": (pred_rounded == y_true).mean(),
        "Within1Acc": (np.abs(pred_rounded - y_true) <= 1).mean(),
    }


def main():
    df = load_data()
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    pre = build_preprocessor()
    X_train = pre.fit_transform(train[FEATURE_COLS])
    X_val = pre.transform(val[FEATURE_COLS])
    X_test = pre.transform(test[FEATURE_COLS])
    y_train, y_val, y_test = train[TARGET].values, val[TARGET].values, test[TARGET].values

    models = {}
    train_median = np.median(y_train)
    models["MedianBaseline"] = ("baseline", train_median)
    models["Ridge"] = ("model", Ridge(alpha=1.0, random_state=SEED).fit(X_train, y_train))
    models["RandomForest"] = ("model", RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1).fit(X_train, y_train))
    models["GradientBoosting"] = ("model", GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=SEED).fit(X_train, y_train))
    if HAS_XGB:
        models["XGBoost"] = ("model", xgboost.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED,
            n_jobs=-1).fit(X_train, y_train))
    else:
        print("xgboost not installed - skipping (not installing per instructions)")

    results = {}
    for name, (kind, obj) in models.items():
        preds = {}
        for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = np.full_like(y, fill_value=obj, dtype=float) if kind == "baseline" else obj.predict(X)
            preds[split_name] = metrics(y, pred)
        results[name] = preds

    print("=== TRAIN / VAL / TEST METRICS PER MODEL ===")
    for name, preds in results.items():
        for split_name in ["train", "val", "test"]:
            m = preds[split_name]
            print(f"{name:16s} {split_name:5s} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
                  f"ExactAcc={m['ExactAcc']:.3f} Within1={m['Within1Acc']:.3f}")

    best_name = min(
        (n for n in results if n != "MedianBaseline"),
        key=lambda n: results[n]["val"]["MAE"],
    )
    print()
    print(f"BEST MODEL (by val MAE): {best_name}")
    print("Test metrics (untouched, chosen only after val selection):", results[best_name]["test"])

    best_kind, best_obj = models[best_name]
    test_pred_rounded = np.clip(np.round(best_obj.predict(X_test)), 0, 9).astype(int)
    cm = confusion_matrix(y_test, test_pred_rounded, labels=list(range(10)))
    print()
    print("Confusion matrix (test, rows=actual 0-9, cols=predicted 0-9):")
    print(cm)

    err = test_pred_rounded - y_test
    print()
    print("Ordinal error distribution (predicted - actual, test):")
    print(pd.Series(err).value_counts().sort_index())

    feature_names = pre.get_feature_names_out()
    importances = None
    if hasattr(best_obj, "feature_importances_"):
        importances = sorted(zip(feature_names, best_obj.feature_importances_),
                              key=lambda x: -x[1])[:10]
        print()
        print("Top-10 feature importances (best model):")
        for f, v in importances:
            print(f"  {f:40s} {v:.4f}")
    elif hasattr(best_obj, "coef_"):
        importances = sorted(zip(feature_names, np.abs(best_obj.coef_)), key=lambda x: -x[1])[:10]
        print()
        print("Top-10 |coefficients| (best model, Ridge):")
        for f, v in importances:
            print(f"  {f:40s} {v:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_obj, MODELS_DIR / "best_model.joblib")
    dump(pre, MODELS_DIR / "preprocessor.joblib")

    metadata = {
        "disclaimer": "Synthetic India-inspired dataset; model performance is a prototype "
                       "demonstration and is not evidence of real-world Indian bridge prediction accuracy.",
        "seed": SEED,
        "target": TARGET,
        "feature_list": FEATURE_COLS,
        "train_years": f"<= {TRAIN_END_YEAR}",
        "val_years": f"{TRAIN_END_YEAR+1}-{VAL_END_YEAR}",
        "test_years": f"> {VAL_END_YEAR}",
        "best_model": best_name,
        "metrics": {n: preds for n, preds in results.items()},
        "package_versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost": xgboost.__version__ if HAS_XGB else "not installed",
        },
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    print()
    print(f"Saved: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved: {MODELS_DIR / 'preprocessor.joblib'}")
    print(f"Saved: {MODELS_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
