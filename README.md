# BridgeGuard AI

An end-to-end ML prototype that predicts bridge condition, forecasts deterioration, estimates remaining useful life, and explains its predictions — built on a synthetic India-inspired dataset.

> **Prototype notice**: Synthetic India-inspired data — demonstration only. Not real Indian bridge inspection data. Not a certified structural safety system. Does not replace professional bridge inspection.

New here? Start with **[GETTING_STARTED.md](GETTING_STARTED.md)** — a full beginner-friendly walkthrough of the whole project.

## Key Features

- Current bridge-condition prediction (Gradient Boosting, test MAE 0.405 on synthetic data)
- +5-year and +10-year deterioration forecasting
- Remaining Useful Life estimation via Kaplan-Meier / Cox / Weibull-AFT survival analysis, with honest uncertainty (never fabricates a precise number when one isn't estimable)
- SHAP-based explainability for every prediction
- A transparent, documented 0–100 "BridgeGuard Prototype Health Score"
- A FastAPI backend (`POST /predict`) and a vanilla HTML/CSS/JS dashboard, no frontend framework required

## Architecture

```
Frontend → FastAPI (POST /predict) → Prediction Pipeline
  → Current-condition model → +5yr / +10yr forecast models
  → Weibull-AFT survival model → SHAP explanation → Health Score → JSON response
```

See `docs/api.md` and `docs/frontend.md` for details, and `GETTING_STARTED.md` for the full plain-English explanation.

## Quick Start

```
cd "D:\BRIDGE GUARD"
run_bridgeguard.bat
```
Then open **http://127.0.0.1:8000/**.

(Manual alternative: `.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000`)

Run tests: `.venv\Scripts\python.exe -m pytest`

Full install/run/troubleshooting instructions: **[GETTING_STARTED.md](GETTING_STARTED.md)**.

## Technology Stack

Python, pandas, numpy, scikit-learn, lifelines (survival analysis), SHAP, FastAPI, Pydantic, uvicorn — vanilla HTML/CSS/JS frontend, no build step.

## Project Status

Prototype / portfolio-demonstration stage. All phases complete: synthetic data generation → feature engineering → baseline models → deterioration forecasting → RUL/survival analysis → explainability → health score → API → frontend dashboard. 20/20 automated tests passing. See `docs/` for the full phase-by-phase technical record, including documented limitations.

## Disclaimer

This project uses a synthetic, India-inspired dataset for prototype and showcase purposes. It is **not** real Indian bridge inspection data, is **not** a certified structural safety rating or official Indian bridge condition system, and does **not** replace professional bridge inspection.
