"""Central prediction orchestration -- the ONLY place that calls the models.
API routes must not duplicate this logic; they call run_full_prediction().

Synthetic India-inspired prototype -- see disclaimer in every response.
"""

import sys
from pathlib import Path

import pandas as pd
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import CATEGORICAL_COLS, NUMERIC_COLS, BOOL_COLS
from src.models.forecast import predict_future_condition
from src.models.survival import get_population_fit_stats, encode_covariates, median_and_range
from src.explainability.shap_explainer import explain_bridge, _friendly
from src.scoring.health_score import compute_health_score, DISCLAIMER

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS

_cache = {}


def _get(name):
    if name not in _cache:
        _cache[name] = load(MODELS_DIR / name)
    return _cache[name]


def _predict_current_condition(feature_row: dict) -> float:
    model = _get("best_model.joblib")
    pre = _get("preprocessor.joblib")
    X = pre.transform(pd.DataFrame([feature_row])[FEATURE_COLS])
    return float(model.predict(X)[0])


def _rul_estimate(feature_row: dict) -> dict:
    """RUL via the Weibull-AFT model. Uses the request's bridge_age as a stand-in
    for bridge_age_at_start (survival.py's model expects first-inspection baseline
    covariates; a full API would separately ask for the bridge's original
    inspection year -- documented simplification, see docs/api.md)."""
    static_row = {
        "bridge_age_at_start": feature_row["bridge_age"],
        "material": feature_row["material"], "structure_type": feature_row["structure_type"],
        "road_class": feature_row["road_class"], "exposure_condition": feature_row["exposure_condition"],
        "ever_rehabilitated": feature_row["ever_rehabilitated"],
        "adt": feature_row["adt"], "adtt_percent": feature_row["adtt_percent"],
        "annual_rainfall_mm": feature_row["annual_rainfall_mm"], "avg_temp_c": feature_row["avg_temp_c"],
        "monsoon_intensity": feature_row["monsoon_intensity"], "flood_risk_score": feature_row["flood_risk_score"],
        "num_spans": feature_row["num_spans"], "max_span_m": feature_row["max_span_m"],
        "total_length_m": feature_row["total_length_m"], "deck_width_m": feature_row["deck_width_m"],
        "num_lanes": feature_row["num_lanes"],
    }
    fit_stats = get_population_fit_stats()
    row_df = pd.DataFrame([static_row])
    row_df["duration"] = 0.5
    row_df["event"] = 0
    X, _ = encode_covariates(row_df, fit_stats)

    aft = _get("weibull_aft_survival_model.joblib")
    for c in aft.params_.index.get_level_values(1).unique():
        if c not in X.columns and c != "Intercept":
            X[c] = 0
    covariates = X.drop(columns=["duration", "event"])

    median, lo, hi = median_and_range(aft, covariates)
    sf = aft.predict_survival_function(covariates).iloc[:, 0]
    idx = (pd.Series(sf.index.values) - 10).abs().to_numpy().argmin()
    s10 = float(sf.iloc[idx])

    if median is not None:
        reliability = "estimable"
        rul_text = f"approximately {lo:.0f}-{hi:.0f} years" if (lo is not None and hi is not None) else f"median {median:.0f} years"
    else:
        reliability = "not_reached_within_observed_horizon"
        rul_text = "not reliably estimable within the observed horizon"

    return {
        "median_years": median, "range_low": lo, "range_high": hi,
        "reliability_flag": reliability, "text": rul_text,
        "survival_10yr_probability": s10,
    }


def run_full_prediction(feature_row: dict) -> dict:
    """feature_row must contain every column in FEATURE_COLS. Single entry
    point -- API routes must call only this function, never the model files
    directly."""
    current = _predict_current_condition(feature_row)
    horizons = predict_future_condition(feature_row)
    rul = _rul_estimate(feature_row)
    shap_result = explain_bridge("current", feature_row)

    top_risk_factors = [f"{_friendly(name)} (SHAP {val:+.3f})" for name, val in
                         (shap_result["top_negative"][:5])]

    health = compute_health_score(
        current_condition=current,
        predicted_condition_5yr=horizons.get("plus5"),
        predicted_condition_10yr=horizons.get("plus10"),
        survival_10yr_probability=rul["survival_10yr_probability"],
        main_risk_factors=top_risk_factors,
    )

    return {
        "health_score": health["health_score"],
        "category": health["category"],
        "confidence": health["confidence"],
        "current_condition": round(current, 2),
        "5_year_prediction": horizons.get("plus5"),
        "10_year_prediction": horizons.get("plus10"),
        "rul_estimate": rul["text"],
        "rul_reliability_flag": rul["reliability_flag"],
        "survival_10yr_probability": round(rul["survival_10yr_probability"], 3),
        "top_risk_factors": top_risk_factors,
        "shap_explanation": shap_result["human_readable"],
        "component_scores": health["component_scores"],
        "prototype_disclaimer": DISCLAIMER,
    }
