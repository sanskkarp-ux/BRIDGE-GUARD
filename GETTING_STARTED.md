# Getting Started with BridgeGuard AI

A beginner-friendly guide. You don't need to know anything about this project yet — just basic Python.

## QUICK START

1. Clone/download the repository
2. Open a terminal in the project folder (`D:\BRIDGE GUARD`)
3. Run `run_bridgeguard.bat`
4. Open `http://127.0.0.1:8000/` in your browser
5. Enter bridge information
6. Click **ANALYZE BRIDGE**

Everything below explains what just happened, in detail.

---

## 1. What Is BridgeGuard?

BridgeGuard AI predicts the future health of a highway bridge. You enter information about a bridge (age, material, traffic, rainfall, etc.), and the system:

- Predicts its current condition (on a 0–9 scale)
- Forecasts what condition it will likely be in 5 and 10 years from now
- Estimates its Remaining Useful Life (RUL) — roughly how many years until it reaches "poor" condition
- Explains *why* it made that prediction
- Combines everything into one easy "Health Score" (0–100) with a category like "Good" or "Critical"

All of this appears on a single web dashboard.

## 2. Important Disclaimer

**This project uses a synthetic India-inspired dataset for prototype/showcase purposes.**

- This is **NOT** real Indian bridge inspection data.
- This is **NOT** a certified structural safety system.
- It does **NOT** replace professional bridge inspection.
- Every score and prediction is a demonstration of the ML pipeline, not a real-world engineering assessment.

**Why synthetic data?** During research (see `docs/india_data_strategy.md`), no publicly downloadable, bridge-by-bridge, multi-year Indian bridge condition dataset could be found. Rather than wait indefinitely for official data access, the project generates a *plausible, India-inspired* dataset (real Indian states, realistic climate/traffic ranges, realistic materials) to build and demonstrate the full ML pipeline end-to-end. The relationships in the data (e.g., how fast a bridge deteriorates) are chosen by the developer to be *believable*, not learned from real bridges.

## 3. Project Architecture

```
User
 ↓
Frontend (a web page in your browser)
 ↓
FastAPI (the web server that receives your request)
 ↓
Prediction Pipeline (one function that calls everything below, in order)
 ↓
ML Models (predicts current bridge condition)
 ↓
5-Year / 10-Year Forecast (predicts future condition)
 ↓
RUL / Survival Analysis (estimates years until "poor" condition)
 ↓
SHAP Explainability (explains which factors drove the prediction)
 ↓
Health Score (combines everything into one 0–100 number)
 ↓
Dashboard (displays it all back to you)
```

- **Frontend**: the web page you interact with (HTML/CSS/JavaScript). No installation needed — your browser just renders it.
- **FastAPI**: a Python web framework. It listens for your request and hands it to the pipeline.
- **Prediction Pipeline**: one central Python function that calls every model in the right order, so the logic exists in exactly one place.
- **ML Models**: trained on the synthetic dataset to predict a bridge's condition rating.
- **Forecast Models**: two more trained models, specifically for "5 years from now" and "10 years from now."
- **Survival Analysis**: a different kind of model that estimates *time until an event* (here: reaching poor condition) rather than a condition rating.
- **SHAP**: a technique that explains individual predictions — which inputs pushed the score up or down.
- **Health Score**: a formula that combines the above into one summary number.

## 4. Folder Structure

| Folder / File | Purpose |
|---|---|
| `src/data/` | Generates the synthetic bridge dataset (and earlier research code for the abandoned real-US-data approach). |
| `src/features/` | Turns raw bridge-year records into model-ready features, without ever leaking future information into a row. |
| `src/models/` | Trains and stores the current-condition, +5yr, +10yr, and RUL/survival models. |
| `src/scoring/` | Computes the BridgeGuard Prototype Health Score from model outputs. |
| `src/pipeline/` | The single orchestration layer — calls every model in order, once, so no other file duplicates this logic. |
| `src/api/` | The FastAPI web server (`POST /predict`) and static-file hosting for the dashboard. |
| `src/frontend/` | The dashboard itself (HTML/CSS/JS). |
| `src/explainability/` | SHAP explanation code. |
| `tests/` | Automated tests — run these to confirm nothing is broken. |
| `docs/` | Detailed technical documentation for every phase of the project (see below). |
| `models/` | Saved, already-trained model files (so you don't have to retrain anything to use the app). |
| `requirements.txt` | The exact Python packages this project needs. |
| `run_bridgeguard.bat` | One-click Windows script to start the whole app. |

## 5. Data

A **bridge-year record** is one row of data describing one bridge in one particular year — e.g., "Bridge #1042, as inspected in 2015." Because the same bridge is inspected multiple times across its life, the dataset is **longitudinal**: it tracks the same bridges over time, which is essential for learning how bridges deteriorate.

**Synthetic** means the data is computer-generated to look and behave like real data, using believable rules (e.g., older bridges tend to be in worse condition), but it does not come from any real inspection.

The generator produces **5,000 bridges** (a small number failed to get any valid inspection year and were dropped, leaving **4,994**), with **~33,806 bridge-year records** total, spanning inspection years **1995–2024**. A fixed random seed (42) makes the dataset reproducible.

**Important variables**: bridge age, material (RCC/PSC/Steel/Masonry/Timber), structure type, traffic (ADT, heavy-vehicle %), rainfall, temperature, flood/scour exposure, rehabilitation history, and the condition ratings themselves.

**Condition scale**: 0–9, where 9 = excellent and 0 = worst (this matches the convention used by the U.S. National Bridge Inventory, which the project researched early on).

**Poor-condition threshold**: a bridge is considered "poor" once its overall condition drops to **4 or below**. This exact threshold is used consistently everywhere in the project — feature engineering, RUL, and the Health Score.

## 6. Feature Engineering

A "feature" is any input the model is allowed to use. The project computes features like:

- **Bridge age** = inspection year − year built
- **Rehabilitation history** = years since the bridge was last repaired (or, if never repaired, its full age)
- **Traffic** = daily vehicle count and what percentage are heavy vehicles
- **Environmental variables** = rainfall, temperature, monsoon intensity, flood/scour exposure
- **Previous condition** = the bridge's condition rating at its last inspection
- **Historical deterioration features** = how much the bridge's condition has changed recently, and its average rate of decline over its whole history

### Data Leakage (explained simply)

**Data leakage** means accidentally letting the model "see the future" — a feature that wouldn't actually be known yet at prediction time.

> **Example**: If we are predicting a bridge's condition in 2020, we must not give the model information from 2023. That would be like giving a student the exam answers before the test — the model would look accurate during testing but fail in real use, where the future genuinely isn't known yet.

Every feature in this project is checked to make sure it only uses information dated at or before the row's own year. This was actively tested (see `docs/model_validation.md`).

## 7. ML Models (Current Condition)

The project predicts a bridge's **current condition** (0–9) using its structural, traffic, environmental, and history features. Several models were trained and compared:

- **Median baseline** — just guesses the typical value, as a sanity floor
- **Ridge regression** — a simple linear model
- **Random Forest** — an ensemble of decision trees
- **Gradient Boosting** — another tree-based ensemble, built one tree at a time to fix previous mistakes

**Gradient Boosting won** (chosen using validation data, not test data). On the held-out **synthetic** test set: **MAE = 0.405**.

**MAE (Mean Absolute Error)** in simple terms: if MAE = 0.405, it means the model's predicted rating is, on average, only about 0.4 points off from the real rating (on a 0–9 scale). If the real condition is 6, the model typically predicts somewhere around 5.6–6.4.

**Why temporal splitting, not random splitting?** The data is longitudinal — the same bridge shows up many times across different years. If we randomly split rows into train/test, a bridge's *later* record could end up in training while its *earlier* record ends up in testing — the model would effectively be tested on the past using knowledge of the future. Instead, the project splits strictly by year: earlier years train the model, later years test it, exactly matching how it would really be used (predict the future from the past).

*(All results above are on the synthetic dataset — see `docs/model_results.md` for the full comparison table.)*

## 8. 5-Year and 10-Year Forecasting

- **+5-year prediction**: what condition the model expects the bridge to be in, 5 years from the input year.
- **+10-year prediction**: same idea, but 10 years out.

Long-term prediction is harder because more can change — deterioration compounds, and the bridge's *current* condition becomes a weaker clue the further out you look (confirmed in the project's own SHAP analysis — see `docs/explainability.md`).

Separate temporal splits were needed for these because a 10-year-ahead target needs a bridge record 10 years later to exist in the data — and the dataset only runs through 2024, so rows from recent years simply can't have a valid +10-year answer yet. Reusing the current-condition model's split would have left almost no valid test rows for the +10-year model — a real bug caught and fixed during development (see `docs/forecasting_results.md`).

**Documented synthetic-data test results**: +5-year MAE = 0.447; +10-year MAE = 0.464 (both slightly higher/harder than the current-condition model's 0.405, as expected).

## 9. RUL / Survival Analysis

**RUL (Remaining Useful Life)** answers: "roughly how many more years until this bridge reaches poor condition?"

- **Event**: the moment a bridge's condition first reaches the poor threshold (≤4).
- **Censored bridge**: a bridge that, as of the last time we observed it, had *not yet* reached poor condition. We don't know exactly when (or if) it will — only that it hadn't happened yet.
- **Survival probability**: the chance a bridge is *still above* the poor threshold at a given number of years in the future.
- **Cox model**: a classic statistical model for this kind of "time until an event" data. It was tried first.
- **Weibull-AFT model**: a different survival model that assumes deterioration follows a specific mathematical curve.

**Why Weibull-AFT for individual RUL?** When the project tested the Cox model's core assumption (that risk factors have a *constant* relative effect over time), it failed for the most important variables (age, material, rehab status). The Weibull-AFT model doesn't require that assumption and fit the data's actual shape better, so it's used for individual bridge RUL estimates. This is documented, not hidden — see `docs/rul_results.md`.

The project's own documented finding: population median survival time ≈ **27 years**, with **26.8% of bridges reaching the event** and **73.2% censored** in the synthetic dataset.

## 10. SHAP (Explainability)

**What is SHAP?** A technique that answers: *"Why did the model make this prediction?"* — for one specific bridge, it shows which input features pushed the predicted condition up (better) and which pushed it down (worse), and by how much.

- **Global explanation**: which features matter most *on average*, across many bridges.
- **Individual explanation**: for *one specific* bridge's prediction, exactly which of its features mattered most.

**Synthetic-data limitation**: the project's synthetic generator intentionally makes **Timber** bridges deteriorate much faster than other materials. SHAP correctly picks this up as a very strong signal — but that strength is a property of *how the fake data was generated*, not a real-world engineering finding about timber bridges. This is explicitly documented in `docs/explainability.md` so it's never mistaken for real domain knowledge.

## 11. Health Score

The **BridgeGuard Prototype Health Score** (0–100) combines everything into one number:

```
health_score = 0.40 × (current condition, scaled 0-100)
             + 0.20 × (+5-year predicted condition, scaled 0-100)
             + 0.15 × (+10-year predicted condition, scaled 0-100)
             + 0.25 × (10-year survival probability × 100)
```

The weights aren't arbitrary — they're explained in `docs/health_score.md`: current condition gets the most weight because its model is the most accurate; the +5/+10-year weights are reduced (not equal to current) because SHAP showed they share a lot of the same underlying signal as current condition (avoiding "triple counting"); survival gets meaningful weight but not the most, because its very high accuracy was found to be inflated by the synthetic generator, not real predictive power.

**Categories**:

| Score | Category |
|---|---|
| 81–100 | Excellent |
| 61–80 | Good |
| 41–60 | Fair |
| 21–40 | Poor |
| 0–20 | Critical |

These category *names* deliberately match India's own IRC:SP:35 bridge-inspection guideline terminology — but **these are prototype categories from a synthetic-data demo, not official Indian bridge ratings.**

## 12. API

The dashboard talks to one endpoint: **`POST /predict`**.

**You send**: a JSON object describing one bridge — state, material, structure type, age, traffic, rainfall, temperature, flood exposure, rehabilitation info, and (optionally, if known) its previous inspection history.

**You get back**: the health score, category, confidence, current condition, 5-year and 10-year predictions, RUL estimate, survival probability, top risk factors, a SHAP explanation, and the prototype disclaimer.

Simplified example:
```json
// Request (abbreviated)
{"state":"Kerala","material":"RCC","bridge_age":25,"adt":5000,"adtt_percent":12, ...}

// Response (abbreviated)
{"health_score":64,"category":"Good","current_condition":5.4,
 "5_year_prediction":5,"10_year_prediction":4,
 "rul_estimate":"median 27 years","survival_10yr_probability":0.871,
 "prototype_disclaimer":"BridgeGuard Prototype Health Score -- ..."}
```
Full field list in `docs/api.md`.

## 13. Frontend

The dashboard has:
- **Input section** — a form for every field the API needs, grouped into Bridge Identity, Structural Profile, Traffic & Load, Environment, and Maintenance.
- **Health Score** — a large circular score with its category and confidence level.
- **Forecast** — current, +5-year, and +10-year condition shown as numbers and a chart.
- **RUL** — the estimated remaining useful life, with a range where the model can support one, or an honest "not reliably estimable" message when it can't.
- **Survival probability** — the 10-year survival percentage.
- **Risk factors** — a bar chart of what's driving the score up or down.
- **SHAP explanation** — a plain-language breakdown of contributing factors.
- **Disclaimer** — always visible, reminding you this is a synthetic-data prototype.

## 14. How to Install

Assuming you've already downloaded/cloned the repository to `D:\BRIDGE GUARD`:

```
cd "D:\BRIDGE GUARD"
```

This project uses its **own local Python environment** (`.venv`) rather than whatever Python happens to be on your system PATH — this avoids a real problem this project hit (see section 17). If `.venv` doesn't exist yet:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, just install/update dependencies:
```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 15. How to Run

**Easiest method**:
```
run_bridgeguard.bat
```
Double-click it, or run it from any terminal. It handles the `.venv` setup automatically if needed.

**Manual method**:
```
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Then open:
```
http://127.0.0.1:8000/
```

## 16. How to Test

```
.venv\Scripts\python.exe -m pytest
```

As documented in this project's own test runs, **all 20 existing tests pass** (feature-engineering leakage checks + API tests).

## 17. Common Problems

**"uvicorn is not recognized"**
You're likely running a `python`/`uvicorn` command that isn't inside this project's `.venv`. Always use the full path: `.venv\Scripts\python.exe -m uvicorn ...`.

**Wrong Python installation**
This project's own development machine had *three* separate Python installs, and the system PATH pointed to one that never had these packages installed. Using `.venv\Scripts\python.exe` directly sidesteps this entirely — it's a self-contained environment tied to this project folder, not your system PATH.

**Missing packages**
Run `.venv\Scripts\python.exe -m pip install -r requirements.txt` again.

**Port 8000 already in use**
Another BridgeGuard instance (or something else) is already using that port. Either stop it, or run uvicorn with `--port 8001` and open `http://127.0.0.1:8001/` instead.

**API not responding**
Confirm the server is actually running (you should see `Uvicorn running on http://127.0.0.1:8000` in the terminal). If the terminal window was closed, the server stopped — just start it again.

## 18. How to Use the Dashboard

1. Start BridgeGuard (`run_bridgeguard.bat`)
2. Open your browser to `http://127.0.0.1:8000/`
3. Enter bridge information in the form
4. Click **ANALYZE BRIDGE**
5. Read the Health Score and its category
6. Check the 5-year / 10-year condition forecast
7. Check the RUL estimate (or its "not reliably estimable" message)
8. Check the risk factors bar chart
9. Read the "Why This Prediction?" explanation

## 19. For Developers

To later swap in real Indian bridge data instead of the synthetic generator:

1. Build a new data-loading module (parallel to `src/data/synthetic_generator.py`) that outputs the same column schema as `data/processed/synthetic/bridges_synthetic.csv` (see `docs/synthetic_data.md` for the exact schema).
2. Feed it through the existing, unchanged `src/features/feature_engineering.py` pipeline — it doesn't care where the rows came from, only that the schema matches.
3. Retrain the models in `src/models/` (`train_baseline.py`, `train_forecast.py`, `survival.py`) against the new data.
4. Everything downstream — SHAP, Health Score, API, frontend — needs no changes at all, since they only consume model outputs, not raw data directly.

`docs/india_data_strategy.md` documents the real-data research already done (Uttarakhand PWD InfraMgt is the strongest lead found) as a starting point.

## 20. Project Limitations

Stated plainly, as documented throughout `docs/`:

- The dataset is **entirely synthetic** — not real bridges.
- The deterioration relationships (how fast bridges decay) are rules chosen by the developer, not learned from real-world data.
- **Timber's dominant effect on predictions is generator-driven** — an artifact of a rule in the synthetic generator, not a validated real-world engineering finding.
- The survival model's very high accuracy (concordance ~0.96) is **unusually high because of how clean the synthetic generation process is** — real bridge data would very likely show much lower, more realistic accuracy.
- Traffic and climate variables currently contribute **very little signal** to any model — this is a known limitation, not something to trust as-is.
- The Health Score's weights are **documented prototype judgment calls**, not fitted or optimized against any ground truth.
- **This is not a certified safety system** and does not replace professional inspection.

---

## Where to Learn More

Every phase of this project has its own detailed technical document in `docs/`: `synthetic_data.md`, `feature_engineering.md`, `model_validation.md`, `model_results.md`, `forecasting_results.md`, `rul_results.md`, `explainability.md`, `health_score.md`, `api.md`, `frontend.md`, and the earlier real-data research in `data_sources.md`, `research_review.md`, and `india_data_strategy.md`.
