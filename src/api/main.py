"""BridgeGuard prototype API. Thin route layer only -- all prediction logic
lives in src/pipeline/predict.py (single orchestration entry point).

Synthetic India-inspired prototype. Every response carries the disclaimer.
"""

import sys
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pipeline.predict import run_full_prediction, FEATURE_COLS

app = FastAPI(title="BridgeGuard Prototype API",
              description="Synthetic India-inspired prototype -- not a real safety assessment tool.")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "src" / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

VALID_MATERIALS = ["RCC", "PSC", "Steel", "Masonry", "Timber"]
VALID_STRUCTURE_TYPES = ["Slab", "Girder", "Box", "Truss", "Arch"]
VALID_ROAD_CLASSES = ["NationalHighway", "StateHighway", "DistrictRoad"]
VALID_EXPOSURE = ["Mild", "Moderate", "Severe"]
VALID_STATES = ["Kerala", "Rajasthan", "Uttarakhand", "WestBengal", "Maharashtra", "Gujarat",
                "TamilNadu", "Karnataka", "UttarPradesh", "Bihar", "Assam", "Punjab", "Odisha",
                "Delhi", "MadhyaPradesh"]


class BridgeInput(BaseModel):
    state: Literal[tuple(VALID_STATES)]
    material: Literal[tuple(VALID_MATERIALS)]
    structure_type: Literal[tuple(VALID_STRUCTURE_TYPES)]
    road_class: Literal[tuple(VALID_ROAD_CLASSES)]
    exposure_condition: Literal[tuple(VALID_EXPOSURE)]

    bridge_age: float = Field(ge=0, le=150)
    years_since_rehab: Optional[float] = Field(default=None, ge=0, le=150)
    ever_rehabilitated: bool

    adt: float = Field(ge=0, le=200000)
    adtt_percent: float = Field(ge=0, le=100)

    annual_rainfall_mm: float = Field(ge=0, le=6000)
    avg_temp_c: float = Field(ge=-10, le=55)
    monsoon_intensity: float = Field(ge=0, le=1)
    flood_risk_score: float = Field(ge=0, le=1)

    num_spans: int = Field(ge=1, le=50)
    max_span_m: float = Field(gt=0, le=500)
    total_length_m: float = Field(gt=0, le=5000)
    deck_width_m: float = Field(gt=0, le=60)
    num_lanes: int = Field(ge=1, le=12)
    latitude: float = Field(ge=6, le=38)
    longitude: float = Field(ge=68, le=98)

    prev_deck_condition: Optional[float] = Field(default=None, ge=0, le=9)
    prev_superstructure_condition: Optional[float] = Field(default=None, ge=0, le=9)
    prev_substructure_condition: Optional[float] = Field(default=None, ge=0, le=9)
    prev_overall_condition: Optional[float] = Field(default=None, ge=0, le=9)
    overall_condition_trend: Optional[float] = Field(default=None, ge=-9, le=9)
    num_previous_inspections: Optional[int] = Field(default=None, ge=0)
    years_since_first_inspection: Optional[float] = Field(default=None, ge=0)
    historical_deterioration_rate: Optional[float] = Field(default=None)

    def to_feature_row(self) -> dict:
        row = self.model_dump()
        row["years_since_rehab_filled"] = (
            row.pop("years_since_rehab") if row.get("years_since_rehab") is not None
            else row["bridge_age"]
        )
        row.pop("years_since_rehab", None)
        row["heavy_vehicle_traffic"] = row["adt"] * row["adtt_percent"] / 100
        missing = set(FEATURE_COLS) - set(row)
        if missing:
            raise ValueError(f"internal schema mismatch, missing: {missing}")
        return row


@app.post("/predict")
def predict(bridge: BridgeInput):
    try:
        feature_row = bridge.to_feature_row()
        result = run_full_prediction(feature_row)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api")
def api_info():
    return {"message": "BridgeGuard Prototype API -- synthetic India-inspired prototype. POST /predict.",
            "docs": "/docs"}
