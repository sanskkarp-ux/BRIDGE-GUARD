"""SHAP explainability for the current/+5yr/+10yr GradientBoosting models.

Synthetic India-inspired prototype -- explanations describe what the model
learned from synthetic data, not validated real-world engineering drivers.

The RUL/survival model (Weibull-AFT) is NOT explained with SHAP -- see
docs/explainability.md for why, and its own native coefficient-based
explanation instead (already computed in docs/rul_results.md).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import CATEGORICAL_COLS, NUMERIC_COLS, BOOL_COLS, SYNTH_DIR
from src.models.train_forecast import PLUS5_TRAIN_END, PLUS5_VAL_END, PLUS10_TRAIN_END, PLUS10_VAL_END

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS

# raw feature name -> human-readable label (used in individual-bridge text explanations)
FRIENDLY_NAMES = {
    "num__bridge_age": "bridge age",
    "num__prev_overall_condition": "previous overall condition",
    "num__prev_deck_condition": "previous deck condition",
    "num__prev_superstructure_condition": "previous superstructure condition",
    "num__prev_substructure_condition": "previous substructure condition",
    "num__overall_condition_trend": "recent condition trend",
    "num__years_since_rehab_filled": "years since rehabilitation",
    "num__adt": "average daily traffic",
    "num__adtt_percent": "heavy vehicle traffic %",
    "num__heavy_vehicle_traffic": "heavy vehicle traffic volume",
    "num__annual_rainfall_mm": "annual rainfall",
    "num__avg_temp_c": "average temperature",
    "num__monsoon_intensity": "monsoon intensity",
    "num__flood_risk_score": "flood/scour exposure",
    "bool__ever_rehabilitated": "ever rehabilitated",
    "cat__material_Timber": "material: Timber",
    "cat__material_Steel": "material: Steel",
    "cat__material_RCC": "material: RCC",
    "cat__material_PSC": "material: PSC",
    "cat__material_Masonry": "material: Masonry",
}


def _friendly(name: str) -> str:
    return FRIENDLY_NAMES.get(name, name.replace("cat__", "material/type: ").replace("num__", "").replace("_", " "))


def load_task(task: str):
    """task in {'current', 'plus5', 'plus10'}"""
    suffix = "" if task == "current" else f"_{task}"
    model = load(MODELS_DIR / f"best_model{suffix}.joblib")
    pre = load(MODELS_DIR / f"preprocessor{suffix}.joblib")
    return model, pre


def get_test_rows(task: str) -> pd.DataFrame:
    features = pd.read_csv(SYNTH_DIR / "features.csv")
    if task == "current":
        return features[features["split"] == "test"]
    targets = pd.read_csv(SYNTH_DIR / "targets.csv")
    flag = "has_plus5_target" if task == "plus5" else "has_plus10_target"
    val_end = PLUS5_VAL_END if task == "plus5" else PLUS10_VAL_END
    d = features.merge(targets[["bridge_id", "inspection_year", flag]], on=["bridge_id", "inspection_year"])
    return d[d[flag] & (d["inspection_year"] > val_end)]


def compute_shap(task: str, max_rows: int = 2000):
    model, pre = load_task(task)
    rows = get_test_rows(task)
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=42)
    X = pre.transform(rows[FEATURE_COLS])
    feature_names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values, X, feature_names, rows


def global_importance(shap_values, feature_names, top_n=10):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_n]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def validate(explainer, shap_values, X, model):
    """Additivity check: sum(shap) + expected_value ~= model prediction."""
    preds = model.predict(X)
    reconstructed = shap_values.sum(axis=1) + explainer.expected_value
    ok = np.allclose(preds, reconstructed, atol=1e-4)
    max_err = np.max(np.abs(preds - reconstructed))
    return ok, max_err


def explain_bridge(task: str, feature_row: dict) -> dict:
    """Individual-bridge explanation. feature_row must contain all FEATURE_COLS."""
    model, pre = load_task(task)
    row_df = pd.DataFrame([feature_row])[FEATURE_COLS]
    X = pre.transform(row_df)
    feature_names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)[0]
    pred = float(model.predict(X)[0])

    order = np.argsort(-np.abs(sv))
    contributors = [(feature_names[i], float(sv[i])) for i in order if abs(sv[i]) > 1e-6]
    positive = [c for c in contributors if c[1] > 0][:5]
    negative = [c for c in contributors if c[1] < 0][:5]

    def describe(clist, direction):
        lines = []
        for name, val in clist:
            lines.append(f"  - {_friendly(name)} (SHAP {val:+.3f})")
        return lines or [f"  (none significant)"]

    text = [f"Prediction ({task}): {pred:.2f}", "",
            "Factors pushing condition UP (better):"] + describe(positive, "up") + \
           ["", "Factors pushing condition DOWN (worse):"] + describe(negative, "down")

    return {
        "task": task, "prediction": pred,
        "top_positive": positive, "top_negative": negative,
        "human_readable": "\n".join(text),
    }


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_importance = {}

    for task in ["current", "plus5", "plus10"]:
        explainer, shap_values, X, feature_names, rows = compute_shap(task)
        model, _ = load_task(task)
        ok, max_err = validate(explainer, shap_values, X, model)
        top10 = global_importance(shap_values, feature_names)

        print(f"\n=== {task} : SHAP on {len(rows)} test rows ===")
        print(f"additivity check (sum(shap)+base == prediction): {'OK' if ok else 'FAILED'} (max err={max_err:.6f})")
        print("Top 10 features by mean(|SHAP|):")
        for name, val in top10:
            print(f"  {_friendly(name):40s} {val:.4f}")
        all_importance[task] = top10

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [_friendly(n) for n, _ in reversed(top10)]
        vals = [v for _, v in reversed(top10)]
        plt.figure(figsize=(8, 5))
        plt.barh(names, vals, color="#4c72b0")
        plt.xlabel("mean(|SHAP value|)")
        plt.title(f"Top-10 SHAP feature importance -- {task} (synthetic data)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"shap_importance_{task}.png", dpi=120)
        plt.close()
        print(f"Saved plot: {REPORTS_DIR / f'shap_importance_{task}.png'}")

    print("\n=== Example individual bridge explanation (current, first test row) ===")
    rows = get_test_rows("current")
    example = rows.iloc[0][FEATURE_COLS].to_dict()
    result = explain_bridge("current", example)
    print(result["human_readable"])
