"""API tests. Run: pytest tests/test_api.py -q"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.main import app
from src.pipeline.predict import run_full_prediction

client = TestClient(app)


def _base_payload(**overrides):
    payload = {
        "state": "Kerala", "material": "RCC", "structure_type": "Slab",
        "road_class": "StateHighway", "exposure_condition": "Severe",
        "bridge_age": 25, "years_since_rehab": 25, "ever_rehabilitated": False,
        "adt": 5000, "adtt_percent": 12,
        "annual_rainfall_mm": 2900, "avg_temp_c": 27, "monsoon_intensity": 0.6,
        "flood_risk_score": 0.55, "num_spans": 3, "max_span_m": 20,
        "total_length_m": 60, "deck_width_m": 9, "num_lanes": 2,
        "latitude": 10.5, "longitude": 76.3,
        "prev_deck_condition": 6, "prev_superstructure_condition": 6,
        "prev_substructure_condition": 6, "prev_overall_condition": 6,
        "overall_condition_trend": -1, "num_previous_inspections": 4,
        "years_since_first_inspection": 12, "historical_deterioration_rate": 0.25,
    }
    payload.update(overrides)
    return payload


def test_valid_request_returns_200():
    r = client.post("/predict", json=_base_payload())
    assert r.status_code == 200


def test_invalid_material_rejected():
    r = client.post("/predict", json=_base_payload(material="Diamond"))
    assert r.status_code == 422


def test_invalid_negative_age_rejected():
    r = client.post("/predict", json=_base_payload(bridge_age=-5))
    assert r.status_code == 422


def test_invalid_out_of_range_adtt_rejected():
    r = client.post("/predict", json=_base_payload(adtt_percent=150))
    assert r.status_code == 422


def test_missing_required_field_rejected():
    payload = _base_payload()
    del payload["material"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_healthy_bridge_scores_well():
    payload = _base_payload(bridge_age=5, material="RCC", prev_overall_condition=9,
                             prev_deck_condition=9, prev_superstructure_condition=9,
                             prev_substructure_condition=9, ever_rehabilitated=False,
                             years_since_rehab=None)
    r = client.post("/predict", json=payload)
    body = r.json()
    assert body["health_score"] >= 60
    assert body["category"] in ("Good", "Excellent")


def test_high_risk_bridge_scores_poorly():
    payload = _base_payload(bridge_age=80, material="Timber", prev_overall_condition=2,
                             prev_deck_condition=2, prev_superstructure_condition=2,
                             prev_substructure_condition=2, ever_rehabilitated=False,
                             years_since_rehab=None)
    r = client.post("/predict", json=payload)
    body = r.json()
    assert body["health_score"] <= 40
    assert body["category"] in ("Poor", "Critical")


def test_uncertain_history_bridge_still_returns_valid_response():
    payload = _base_payload(
        prev_deck_condition=None, prev_superstructure_condition=None,
        prev_substructure_condition=None, prev_overall_condition=None,
        overall_condition_trend=None, num_previous_inspections=None,
        years_since_first_inspection=None, historical_deterioration_rate=None,
    )
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["health_score"] <= 100
    assert body["rul_reliability_flag"] in ("estimable", "not_reached_within_observed_horizon")


def test_health_score_always_in_range():
    for age in [0, 5, 20, 40, 60, 100]:
        r = client.post("/predict", json=_base_payload(bridge_age=age))
        body = r.json()
        assert 0 <= body["health_score"] <= 100


def test_prototype_disclaimer_present():
    r = client.post("/predict", json=_base_payload())
    body = r.json()
    disclaimer = body["prototype_disclaimer"]
    assert "Synthetic" in disclaimer or "synthetic" in disclaimer or "Prototype" in disclaimer
    assert "NOT a certified structural safety rating" in disclaimer


def test_api_output_matches_pipeline_directly():
    payload = _base_payload()
    r = client.post("/predict", json=payload)
    api_result = r.json()

    from src.api.main import BridgeInput
    feature_row = BridgeInput(**payload).to_feature_row()
    direct_result = run_full_prediction(feature_row)

    assert api_result["health_score"] == direct_result["health_score"]
    assert api_result["category"] == direct_result["category"]
    assert api_result["current_condition"] == direct_result["current_condition"]
    assert api_result["5_year_prediction"] == direct_result["5_year_prediction"]
    assert api_result["10_year_prediction"] == direct_result["10_year_prediction"]
