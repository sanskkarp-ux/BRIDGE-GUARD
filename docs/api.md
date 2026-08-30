# BridgeGuard Prototype API

> Synthetic India-inspired prototype. Not a real safety assessment tool. Every response includes `prototype_disclaimer`.

## Endpoint
`POST /predict`

## Architecture
```
Request (Pydantic-validated) -> src/api/main.py (thin route)
                              -> src/pipeline/predict.run_full_prediction()  [single orchestration entry point]
                                   -> current-condition model
                                   -> +5yr / +10yr forecast models (src/models/forecast.py)
                                   -> Weibull-AFT survival model (src/models/survival.py)
                                   -> SHAP explanation (src/explainability/shap_explainer.py)
                                   -> Health Score (src/scoring/health_score.py, formula unchanged)
                              -> JSON response
```
No model is called from `src/api/main.py` directly — every prediction goes through `run_full_prediction()`, so logic isn't duplicated between routes and any future caller (CLI, batch job, notebook).

## Run it
```
uvicorn src.api.main:app --reload
```
Interactive docs at `/docs` (FastAPI auto-generated).

## Request fields
Required: `state`, `material`, `structure_type`, `road_class`, `exposure_condition` (all validated against known categories), `bridge_age`, `ever_rehabilitated`, `adt`, `adtt_percent`, `annual_rainfall_mm`, `avg_temp_c`, `monsoon_intensity`, `flood_risk_score`, `num_spans`, `max_span_m`, `total_length_m`, `deck_width_m`, `num_lanes`, `latitude`, `longitude`.
Optional (null if unknown — a first-time/never-inspected bridge legitimately has none of these): `years_since_rehab`, `prev_deck_condition`, `prev_superstructure_condition`, `prev_substructure_condition`, `prev_overall_condition`, `overall_condition_trend`, `num_previous_inspections`, `years_since_first_inspection`, `historical_deterioration_rate`.
`heavy_vehicle_traffic` is **not** a request field — derived server-side as `adt × adtt_percent / 100`, so callers don't duplicate that arithmetic.

Invalid values are rejected with HTTP 422 (Pydantic `Field` bounds + category `Literal` enums) — e.g. unknown material, negative age, adtt_percent>100, out-of-India-bounding-box lat/long.

## Response fields
`health_score`, `category`, `confidence`, `current_condition`, `5_year_prediction`, `10_year_prediction`, `rul_estimate` (text — never a fabricated decimal), `rul_reliability_flag` (`"estimable"` or `"not_reached_within_observed_horizon"`), `survival_10yr_probability`, `top_risk_factors`, `shap_explanation`, `component_scores`, `prototype_disclaimer`.

## Example
Request (abbreviated — see `tests/test_api.py::_base_payload` for the full set):
```json
{"state":"Kerala","material":"RCC","structure_type":"Slab","road_class":"StateHighway",
 "exposure_condition":"Severe","bridge_age":25,"years_since_rehab":25,"ever_rehabilitated":false,
 "adt":5000,"adtt_percent":12,"annual_rainfall_mm":2900,"avg_temp_c":27,"monsoon_intensity":0.6,
 "flood_risk_score":0.55,"num_spans":3,"max_span_m":20,"total_length_m":60,"deck_width_m":9,
 "num_lanes":2,"latitude":10.5,"longitude":76.3,"prev_deck_condition":6,
 "prev_superstructure_condition":6,"prev_substructure_condition":6,"prev_overall_condition":6,
 "overall_condition_trend":-1,"num_previous_inspections":4,"years_since_first_inspection":12,
 "historical_deterioration_rate":0.25}
```
Response:
```json
{
  "health_score": 64, "category": "Good", "confidence": "high",
  "current_condition": 5.4, "5_year_prediction": 5, "10_year_prediction": 4,
  "rul_estimate": "median 27 years", "rul_reliability_flag": "estimable",
  "survival_10yr_probability": 0.871,
  "top_risk_factors": ["bridge age (SHAP -0.428)", "previous substructure condition (SHAP -0.177)",
                        "years since rehabilitation (SHAP -0.135)", "annual rainfall (SHAP -0.101)",
                        "previous deck condition (SHAP -0.047)"],
  "component_scores": {"current_condition": 60.0, "near_term_trend": 55.6,
                        "long_term_trend": 44.4, "survival_risk": 87.1},
  "prototype_disclaimer": "BridgeGuard Prototype Health Score -- ... NOT a certified structural safety rating ..."
}
```

## Known simplification
The RUL/survival component treats the request's `bridge_age` as a stand-in for the survival model's expected "bridge_age_at_start" (first-inspection baseline) covariate — a full production API would separately capture a bridge's original inspection year. Documented, not hidden.

## Not built (per scope)
No frontend/dashboard, no authentication, no database, no Docker, no cloud deployment.
