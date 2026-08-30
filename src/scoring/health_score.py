"""BridgeGuard Prototype Health Score.

Synthetic India-inspired prototype. NOT a certified structural safety
rating, NOT an official Indian bridge condition rating, NOT a replacement
for physical inspection, NOT evidence of real-world Indian bridge
performance. See docs/health_score.md for full methodology.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.survival import (
    build_survival_dataset, encode_covariates, STATIC_NUMERIC, STATIC_CATEGORICAL, STATIC_BOOL,
)
from src.features.feature_engineering import SYNTH_DIR

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

DISCLAIMER = (
    "BridgeGuard Prototype Health Score -- composite ML-derived risk indicator on "
    "synthetic India-inspired data. NOT a certified structural safety rating, NOT an "
    "official Indian bridge condition rating, NOT a replacement for physical inspection, "
    "NOT evidence of real-world Indian bridge performance."
)

# Weights tied to this project's own documented reliability findings, not picked for
# a nice-looking score:
#  - current_condition gets the plurality weight (40%) because its model has the
#    lowest error of anything in the project (test MAE 0.405, docs/model_results.md).
#  - near/long-term components use each horizon's ABSOLUTE predicted level (not a
#    decline magnitude -- see the fix note below), but get REDUCED weight (20%, 15%)
#    rather than an equal 40% each. This is the deliberate anti-double-counting choice:
#    current/+5/+10 are strongly correlated (docs/explainability.md SHAP analysis --
#    prior condition's importance is high for current but collapses by +10yr as
#    age/material dominate instead), so giving all three full, equal weight would
#    let the same underlying age/material signal vote 3x. Tapering the weight by
#    horizon keeps their COMBINED influence (40+20+15=75%) below what three
#    independent full-weight signals would get (120%), while still letting genuinely
#    new information at each horizon (which SHAP confirms exists, just shrinking)
#    count for something. Near-term > long-term additionally because the +10yr
#    model has higher error (test MAE 0.464 vs 0.447, docs/forecasting_results.md).
#  - survival_risk (25%): a genuinely different model/data construction (baseline-
#    only covariates), so it isn't pure double-counting -- but its concordance
#    (~0.96) was explicitly flagged in docs/rul_results.md as likely inflated by
#    generator determinism, not evidence of superior reliability -- so it does NOT
#    outweigh current_condition despite the higher concordance number.
WEIGHTS = {
    "current_condition": 0.40,
    "near_term_trend": 0.20,
    "long_term_trend": 0.15,
    "survival_risk": 0.25,
}

# IRC:SP:35 (India's own bridge-inspection guideline, docs/india_data_strategy.md research)
# already uses a 5-category Excellent/Good/Fair/Poor/Critical qualitative scale -- reusing
# those category NAMES (not the underlying data) is a deliberate choice over an arbitrary
# NBI-style relabeling, since a label isn't a data-accuracy claim the way a score value is.
CATEGORIES = [(81, 100, "Excellent"), (61, 80, "Good"), (41, 60, "Fair"),
              (21, 40, "Poor"), (0, 20, "Critical")]


def _category(score: int) -> str:
    for lo, hi, name in CATEGORIES:
        if lo <= score <= hi:
            return name
    return "Unknown"


def compute_health_score(current_condition, predicted_condition_5yr=None,
                          predicted_condition_10yr=None, survival_10yr_probability=None,
                          main_risk_factors=None) -> dict:
    """Deterministic, reproducible: same inputs always give the same output.
    Missing components (None) are handled by re-normalizing weights over
    whichever components ARE available -- not by silently treating a
    missing value as 0 or 100."""
    components = {"current_condition": (current_condition / 9) * 100}

    # Uses each horizon's ABSOLUTE predicted level (not decline magnitude).
    # An earlier decline-only design let a bridge that's already bad and stays
    # bad score well on "trend" (small further decline from a low base) --
    # masking absolute badness. Caught by this module's own validation tests
    # (see docs/health_score.md) and fixed here, not by changing any model.
    if predicted_condition_5yr is not None:
        components["near_term_trend"] = (predicted_condition_5yr / 9) * 100
    if predicted_condition_10yr is not None:
        components["long_term_trend"] = (predicted_condition_10yr / 9) * 100
    if survival_10yr_probability is not None:
        components["survival_risk"] = survival_10yr_probability * 100

    active = {k: WEIGHTS[k] for k in components}
    total_w = sum(active.values())
    norm_weights = {k: w / total_w for k, w in active.items()}
    raw_score = sum(components[k] * norm_weights[k] for k in components)
    score = int(round(max(0, min(100, raw_score))))

    n_missing = len(WEIGHTS) - len(components)
    confidence = "high" if n_missing == 0 else ("medium" if n_missing == 1 else "low")

    return {
        "health_score": score,
        "category": _category(score),
        "current_condition": current_condition,
        "predicted_condition_5yr": predicted_condition_5yr,
        "predicted_condition_10yr": predicted_condition_10yr,
        "survival_10yr_probability": survival_10yr_probability,
        "component_scores": {k: round(v, 1) for k, v in components.items()},
        "weights_used": {k: round(v, 3) for k, v in norm_weights.items()},
        "confidence": confidence,
        "main_risk_factors": main_risk_factors or [],
        "disclaimer": DISCLAIMER,
    }


def get_survival_10yr_probability(bridge_row: dict) -> float:
    """bridge_row must have the STATIC_* covariates from survival.py (first-
    inspection attributes). Reuses the already-fitted Weibull-AFT model
    (no retraining) -- normalization stats are recomputed deterministically
    from the full population (see docs/health_score.md for why this is a
    documented, negligible approximation vs. the exact original train-only
    stats, not a retrain)."""
    raw = pd.read_csv(SYNTH_DIR / "bridges_synthetic.csv")
    surv = build_survival_dataset(raw)
    _, fit_stats = encode_covariates(surv)

    row_df = pd.DataFrame([bridge_row])
    row_df["duration"] = 0.5  # placeholder, unused by predict_survival_function; required by encode_covariates
    row_df["event"] = 0
    X, _ = encode_covariates(row_df, fit_stats)
    aft = load(MODELS_DIR / "weibull_aft_survival_model.joblib")
    for c in aft.params_.index.get_level_values(1).unique():
        if c not in X.columns and c not in ("Intercept",):
            X[c] = 0
    sf = aft.predict_survival_function(X.drop(columns=["duration", "event"]))
    idx = (pd.Series(sf.index.values) - 10).abs().to_numpy().argmin()
    return float(sf.iloc[idx, 0])


if __name__ == "__main__":
    examples = {
        "1_healthy_new": dict(current_condition=8, predicted_condition_5yr=8, predicted_condition_10yr=7),
        "2_old_high_risk": dict(current_condition=3, predicted_condition_5yr=2, predicted_condition_10yr=1),
        "3_recently_rehabbed": dict(current_condition=7, predicted_condition_5yr=7, predicted_condition_10yr=6),
        "4_poor_condition": dict(current_condition=2, predicted_condition_5yr=1, predicted_condition_10yr=1),
        "5_uncertain_rul": dict(current_condition=6, predicted_condition_5yr=5, predicted_condition_10yr=None),
    }
    survival_probs = {
        "1_healthy_new": 0.95, "2_old_high_risk": 0.35, "3_recently_rehabbed": 0.85,
        "4_poor_condition": 0.20, "5_uncertain_rul": None,
    }

    results = {}
    for name, ex in examples.items():
        r = compute_health_score(**ex, survival_10yr_probability=survival_probs[name])
        results[name] = r
        print(f"\n{name}: score={r['health_score']} category={r['category']} confidence={r['confidence']}")
        print(f"  components={r['component_scores']} weights={r['weights_used']}")

    print("\n=== VALIDATION ===")
    scores = {k: v["health_score"] for k, v in results.items()}
    print("all scores in [0,100]:", all(0 <= s <= 100 for s in scores.values()))
    print("healthy > old-high-risk:", scores["1_healthy_new"] > scores["2_old_high_risk"])
    print("healthy > poor-condition:", scores["1_healthy_new"] > scores["4_poor_condition"])
    print("recently-rehabbed > old-high-risk:", scores["3_recently_rehabbed"] > scores["2_old_high_risk"])
    print("uncertain-RUL example has reduced confidence:", results["5_uncertain_rul"]["confidence"] != "high")
    print("worse predicted 10yr condition (bridge 2) scores lower than stable one (bridge 1):",
          scores["2_old_high_risk"] < scores["1_healthy_new"])
    print("regression check -- bridges already below our own poor-threshold (condition<=4) "
          "must NOT land in Fair/Good/Excellent:",
          results["2_old_high_risk"]["category"] in ("Poor", "Critical") and
          results["4_poor_condition"]["category"] in ("Poor", "Critical"))
