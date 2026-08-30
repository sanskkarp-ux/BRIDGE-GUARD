"""Prediction utility: bridge info at year Y -> predicted condition at Y+5 / Y+10.

Synthetic India-inspired prototype -- not a real-world prediction tool.
Loads the models trained by train_forecast.py; run that first if models/
doesn't yet contain best_model_plus5.joblib / best_model_plus10.joblib.
"""

import sys
from pathlib import Path

import pandas as pd
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import CATEGORICAL_COLS, NUMERIC_COLS, BOOL_COLS

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS


def predict_future_condition(bridge_features: dict) -> dict:
    """bridge_features must contain every column in FEATURE_COLS (see
    docs/feature_engineering.md) for a single bridge at year Y. Returns
    predicted overall_condition at Y+5 and Y+10, rounded/clipped to 0-9."""
    missing = set(FEATURE_COLS) - set(bridge_features)
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    row = pd.DataFrame([bridge_features])[FEATURE_COLS]
    out = {}
    for horizon in ["plus5", "plus10"]:
        model_path = MODELS_DIR / f"best_model_{horizon}.joblib"
        pre_path = MODELS_DIR / f"preprocessor_{horizon}.joblib"
        if not model_path.exists():
            out[horizon] = None
            continue
        model = load(model_path)
        pre = load(pre_path)
        X = pre.transform(row)
        pred = model.predict(X)[0]
        out[horizon] = int(max(0, min(9, round(pred))))
    return out


if __name__ == "__main__":
    example = {
        "state": "Kerala", "material": "RCC", "structure_type": "Slab",
        "road_class": "StateHighway", "exposure_condition": "Severe",
        "bridge_age": 25, "years_since_rehab_filled": 25, "adt": 5000, "adtt_percent": 12,
        "heavy_vehicle_traffic": 600, "annual_rainfall_mm": 2900, "avg_temp_c": 27,
        "monsoon_intensity": 0.6, "flood_risk_score": 0.55, "num_spans": 3, "max_span_m": 20,
        "total_length_m": 60, "deck_width_m": 9, "num_lanes": 2, "latitude": 10.5, "longitude": 76.3,
        "prev_deck_condition": 6, "prev_superstructure_condition": 6, "prev_substructure_condition": 6,
        "prev_overall_condition": 6, "overall_condition_trend": -1, "num_previous_inspections": 4,
        "years_since_first_inspection": 12, "historical_deterioration_rate": 0.25,
        "ever_rehabilitated": False,
    }
    print("SYNTHETIC PROTOTYPE - not a real prediction. Example output:")
    print(predict_future_condition(example))
