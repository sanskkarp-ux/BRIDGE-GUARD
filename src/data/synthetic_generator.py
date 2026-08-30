"""Synthetic India-inspired bridge dataset generator.

DISCLAIMER: Produces PLAUSIBLE, NOT REAL, data. No real Indian bridge
inspection records, statistics, or deterioration relationships are used
or implied. For BridgeGuard AI prototype/showcase purposes only.

Design rationale: see docs/synthetic_data.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_BRIDGES = 5000
CURRENT_YEAR = 2024
DISCLAIMER = "Synthetic India-inspired data - for prototype/showcase purposes. Not real inspection data."

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "synthetic"

# state: (rainfall_mean_mm, rainfall_sd, temp_mean_c, temp_sd, flood_risk_base, coastal, lat, lon)
STATES = {
    "Kerala":       (2900, 400, 27, 1.5, 0.55, True,  10.5, 76.3),
    "Rajasthan":    (450,  200, 27, 3.0, 0.10, False, 26.9, 73.0),
    "Uttarakhand":  (1500, 350, 15, 4.0, 0.30, False, 30.1, 78.8),
    "WestBengal":   (1800, 350, 26, 2.0, 0.65, True,  22.9, 88.0),
    "Maharashtra":  (1200, 400, 26, 2.5, 0.30, True,  19.5, 75.7),
    "Gujarat":      (900,  350, 27, 2.5, 0.35, True,  22.5, 71.5),
    "TamilNadu":    (1000, 300, 28, 1.5, 0.40, True,  11.1, 78.6),
    "Karnataka":    (1100, 350, 25, 2.0, 0.30, True,  15.0, 76.0),
    "UttarPradesh": (1000, 300, 26, 3.5, 0.45, False, 27.0, 80.5),
    "Bihar":        (1200, 300, 26, 3.0, 0.60, False, 25.6, 85.5),
    "Assam":        (2800, 500, 24, 2.0, 0.70, False, 26.2, 92.5),
    "Punjab":       (700,  250, 24, 4.0, 0.20, False, 31.0, 75.5),
    "Odisha":       (1500, 350, 27, 1.5, 0.60, True,  20.5, 84.7),
    "Delhi":        (800,  200, 25, 4.5, 0.15, False, 28.6, 77.2),
    "MadhyaPradesh":(1100, 300, 26, 3.0, 0.25, False, 23.5, 78.5),
}
STATE_NAMES = list(STATES.keys())

MATERIALS = ["RCC", "PSC", "Steel", "Masonry", "Timber"]
DECAY_RATE = {"RCC": 0.012, "PSC": 0.010, "Steel": 0.015, "Masonry": 0.008, "Timber": 0.035}
STRUCTURE_BY_MATERIAL = {
    "RCC": ["Slab", "Girder", "Box"],
    "PSC": ["Girder", "Box"],
    "Steel": ["Girder", "Truss"],
    "Masonry": ["Arch"],
    "Timber": ["Slab", "Truss"],
}
ROAD_CLASSES = ["NationalHighway", "StateHighway", "DistrictRoad"]


def _material_probs(year_built: np.ndarray) -> np.ndarray:
    """Older bridges skew masonry/timber; newer skew RCC/PSC/Steel."""
    modern = (year_built >= 1990).astype(float)
    probs = np.where(
        modern[:, None] == 1,
        np.array([0.50, 0.25, 0.18, 0.05, 0.02]),
        np.array([0.30, 0.05, 0.10, 0.35, 0.20]),
    )
    return probs


def generate_bridges(rng: np.random.Generator, n: int) -> pd.DataFrame:
    states = rng.choice(STATE_NAMES, size=n)
    year_built = np.round(rng.triangular(1960, 2005, 2022, size=n)).astype(int)
    year_built = np.clip(year_built, 1960, CURRENT_YEAR - 2)

    material_probs = _material_probs(year_built)
    material = np.array([rng.choice(MATERIALS, p=p) for p in material_probs])
    structure_type = np.array([rng.choice(STRUCTURE_BY_MATERIAL[m]) for m in material])

    road_class = rng.choice(ROAD_CLASSES, size=n, p=[0.2, 0.3, 0.5])
    adt_base = np.select(
        [road_class == "NationalHighway", road_class == "StateHighway"],
        [rng.lognormal(9.5, 0.6, n), rng.lognormal(8.2, 0.6, n)],
        default=rng.lognormal(6.5, 0.7, n),
    )
    adt = np.round(np.clip(adt_base, 50, 80000)).astype(int)
    adtt_percent = np.round(np.clip(rng.beta(2, 5, n) * 40, 2, 45), 1)

    num_spans = 1 + rng.poisson(2, n)
    max_span_m = np.round(np.clip(rng.gamma(3, 8, n), 5, 120), 1)
    total_length_m = np.round(num_spans * max_span_m * rng.uniform(0.7, 1.0, n), 1)
    deck_width_m = np.round(np.clip(rng.normal(9, 3, n), 4, 25), 1)
    num_lanes = np.clip(np.round(deck_width_m / 3.5).astype(int), 1, 6)

    rainfall_mean = np.array([STATES[s][0] for s in states])
    rainfall_sd = np.array([STATES[s][1] for s in states])
    temp_mean = np.array([STATES[s][2] for s in states])
    temp_sd = np.array([STATES[s][3] for s in states])
    flood_base = np.array([STATES[s][4] for s in states])
    coastal = np.array([STATES[s][5] for s in states])
    lat_c = np.array([STATES[s][6] for s in states])
    lon_c = np.array([STATES[s][7] for s in states])

    annual_rainfall_mm = np.round(np.clip(rng.normal(rainfall_mean, rainfall_sd), 100, None), 0)
    avg_temp_c = np.round(rng.normal(temp_mean, temp_sd), 1)
    monsoon_intensity = np.round(np.clip(rng.normal(flood_base + 0.2, 0.15), 0, 1), 2)
    flood_risk_score = np.round(np.clip(rng.normal(flood_base, 0.15), 0, 1), 2)
    latitude = np.round(lat_c + rng.normal(0, 1.2, n), 4)
    longitude = np.round(lon_c + rng.normal(0, 1.2, n), 4)

    exposure = np.where(
        coastal & (flood_risk_score > 0.4), "Severe",
        np.where((flood_risk_score > 0.4) | coastal, "Moderate", "Mild"),
    )

    ever_rehab = rng.random(n) < 0.40
    rehab_offset = rng.integers(8, np.maximum((CURRENT_YEAR - year_built), 9), size=n)
    last_rehab_year = np.where(ever_rehab, np.minimum(year_built + rehab_offset, CURRENT_YEAR - 1), 0)

    latent_quality_factor = rng.normal(0, 1, n)  # hidden — dropped before saving modeling data

    district = np.array([f"District_{s}_{i % 12}" for i, s in enumerate(states)])
    bridge_id = np.array([f"BR-{s[:3].upper()}-{i:05d}" for i, s in enumerate(states)])

    return pd.DataFrame({
        "bridge_id": bridge_id, "state": states, "district": district,
        "latitude": latitude, "longitude": longitude, "road_class": road_class,
        "year_built": year_built, "material": material, "structure_type": structure_type,
        "num_spans": num_spans, "max_span_m": max_span_m, "total_length_m": total_length_m,
        "deck_width_m": deck_width_m, "num_lanes": num_lanes,
        "adt": adt, "adtt_percent": adtt_percent,
        "annual_rainfall_mm": annual_rainfall_mm, "avg_temp_c": avg_temp_c,
        "monsoon_intensity": monsoon_intensity, "flood_risk_score": flood_risk_score,
        "exposure_condition": exposure, "last_rehab_year": last_rehab_year,
        "_latent_quality_factor": latent_quality_factor,
    })


def _stress_k(row) -> float:
    """Per-year deterioration rate constant (held ~constant over a bridge's life,
    a documented simplification for the synthetic ground-truth RUL calculation)."""
    decay = DECAY_RATE[row["material"]]
    traffic_factor = 1 + 0.30 * (row["adtt_percent"] / 30) + 0.10 * np.log1p(row["adt"]) / np.log1p(50000)
    climate_factor = 1 + 0.30 * (row["annual_rainfall_mm"] / 3000) + 0.20 * row["flood_risk_score"]
    if row["material"] in ("Steel", "Masonry") and row["exposure_condition"] == "Severe":
        climate_factor *= 1.15
    latent_effect = np.exp(-0.15 * row["_latent_quality_factor"])
    return decay * traffic_factor * climate_factor * latent_effect


def generate_inspections(rng: np.random.Generator, bridges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, gt_records = [], []

    for _, b in bridges.iterrows():
        k = _stress_k(b)
        last_year = CURRENT_YEAR - int(rng.integers(0, 3))
        step = int(rng.choice([2, 3]))
        n_target = int(rng.integers(5, 11))
        years = sorted({last_year - i * step for i in range(n_target) if last_year - i * step > b["year_built"]})
        if len(years) < 3:
            years = sorted({y for y in range(b["year_built"] + 2, last_year + 1, step)})
        if not years:
            continue

        for y in years:
            age = y - b["year_built"]
            rehab_year = b["last_rehab_year"] if b["last_rehab_year"] and b["last_rehab_year"] <= y else 0
            age_since_rehab = (y - rehab_year) if rehab_year else age

            deck_age_eff = age_since_rehab if rehab_year else age
            sub_age_eff = (0.5 * age + 0.5 * age_since_rehab) if rehab_year else age

            base_condition = 9 * np.exp(-age * k)
            deck_condition = 9 * np.exp(-deck_age_eff * k) - 0.3
            super_condition = base_condition
            sub_condition = 9 * np.exp(-sub_age_eff * k) + 0.2

            noise_sd = 0.35 + 0.10 * (age / 60)  # heteroscedastic: older bridges rated less consistently
            def _finalize(x):
                return int(np.clip(round(x + rng.normal(0, noise_sd)), 0, 9))

            deck_c = _finalize(deck_condition)
            super_c = _finalize(super_condition)
            sub_c = _finalize(sub_condition)
            overall_c = min(deck_c, super_c, sub_c)

            records.append({
                "bridge_id": b["bridge_id"], "inspection_year": y, "bridge_age_years": age,
                "years_since_rehab": age_since_rehab if rehab_year else None,
                "adt": b["adt"], "adtt_percent": b["adtt_percent"],
                "annual_rainfall_mm": b["annual_rainfall_mm"], "avg_temp_c": b["avg_temp_c"],
                "monsoon_intensity": b["monsoon_intensity"], "flood_risk_score": b["flood_risk_score"],
                "deck_condition": deck_c, "superstructure_condition": super_c,
                "substructure_condition": sub_c, "overall_condition": overall_c,
                "data_source": DISCLAIMER,
            })

            # ground truth: analytic RUL from the *unrounded* latent trajectory, holding k constant.
            # 9*exp(-age*k) = 4  =>  age* = ln(9/4)/k
            age_at_threshold = np.log(9 / 4) / k
            true_rul = age_at_threshold - age
            censored = true_rul > 100
            gt_records.append({
                "bridge_id": b["bridge_id"], "inspection_year": y,
                "latent_quality_factor": b["_latent_quality_factor"],
                "true_stress_rate_k": k,
                "true_rul_years": None if censored else round(max(true_rul, 0), 1),
                "rul_censored": bool(censored),
            })

    return pd.DataFrame(records), pd.DataFrame(gt_records)


def apply_missingness(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, rate in [("adt", 0.08), ("adtt_percent", 0.10), ("annual_rainfall_mm", 0.05),
                       ("avg_temp_c", 0.05), ("monsoon_intensity", 0.07)]:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bridges = generate_bridges(rng, N_BRIDGES)
    inspections, ground_truth = generate_inspections(rng, bridges)
    inspections = apply_missingness(rng, inspections)

    bridge_static = bridges.drop(columns=["_latent_quality_factor"])
    modeling_df = inspections.merge(bridge_static, on="bridge_id", how="left")

    modeling_path = OUT_DIR / "bridges_synthetic.csv"
    gt_path = OUT_DIR / "bridges_synthetic_ground_truth.csv"
    modeling_df.to_csv(modeling_path, index=False)
    ground_truth.to_csv(gt_path, index=False)

    sample_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "synthetic_sample.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    modeling_df.head(50).to_csv(sample_path, index=False)

    print(DISCLAIMER)
    print(f"bridges: {len(bridges)}, bridge-year records: {len(modeling_df)}")
    print(f"saved: {modeling_path}")
    print(f"saved: {gt_path}")
    print(f"saved sample: {sample_path}")
    print()
    print("overall_condition distribution:")
    print(modeling_df["overall_condition"].value_counts().sort_index())
    print()
    print("mean overall_condition by bridge_age decile:")
    modeling_df["age_decile"] = pd.qcut(modeling_df["bridge_age_years"], 10, duplicates="drop")
    print(modeling_df.groupby("age_decile", observed=True)["overall_condition"].mean())
    print()
    censored_pct = ground_truth["rul_censored"].mean() * 100
    print(f"RUL censored (ground truth, eval-only): {censored_pct:.1f}%")


if __name__ == "__main__":
    main()
