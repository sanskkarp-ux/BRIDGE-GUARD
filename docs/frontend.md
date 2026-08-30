# BridgeGuard Frontend (Showcase Dashboard)

> Synthetic India-inspired prototype. No dashboard value is hardcoded — every number rendered comes from the live `POST /predict` response.

## Technology
**Vanilla HTML/CSS/JS**, no framework, no build step, no new frontend package manager. Served directly by the existing FastAPI app (`src/api/main.py` mounts `src/frontend/` as static files and returns `index.html` at `GET /`), so there's one process to run and no CORS to configure. Chosen specifically to satisfy "no unnecessary packages" — a React/Node toolchain would have been a much larger, unjustified addition for a single-page prototype dashboard.

## Files
- `src/frontend/index.html` — layout: header with persistent "SYNTHETIC INDIA-INSPIRED PROTOTYPE" badge, input form (every `/predict` field, grouped into Location & Structure / Age & Geometry / Traffic / Environment / Rehabilitation / optional Inspection History), results panel (hidden until first successful prediction), disclaimer footer.
- `src/frontend/style.css` — clean engineering-product styling: card layout, restrained color palette, responsive two-column → single-column at ≤900px.
- `src/frontend/app.js` — builds the request payload from the form, calls `POST /predict`, renders every response field, draws a small hand-rolled SVG line chart for the condition forecast (no charting library), handles loading/error/invalid-input states.

## API changes made (minimal, `/predict` untouched)
`src/api/main.py`: added a `StaticFiles` mount at `/static` and changed `GET /` to serve `index.html` instead of a JSON info message (moved that message to `GET /api` instead, so nothing that existed before was removed, just relocated). **`POST /predict`'s request/response contract was not touched.**

## Known limitation
The dropdown option lists (state/material/structure_type/road_class/exposure_condition) in `app.js` mirror `src/api/main.py`'s `VALID_*` constants by hand — there's no schema-introspection endpoint, so if the backend's valid categories ever change, the frontend list needs a matching manual update.

## Python environment (fixed after a real startup failure)
This machine has **three separate Python installations** (`C:\Python314`, `C:\Users\DELL\AppData\Local\Programs\Python\Python311`, and a WindowsApps 3.12) with `C:\Python314` appearing first on PATH. Running plain `python` or `uvicorn` from PowerShell resolves to whichever one PATH finds first -- not necessarily the one any packages were installed into, which is exactly what broke `python -m uvicorn src.api.main:app --reload` when first tried. **Fixed with a project-local `.venv`** (Python 3.12.10, created via `python3 -m venv .venv` from the same interpreter this project's packages were verified against) so startup no longer depends on PATH resolution at all -- everything below invokes `.venv\Scripts\python.exe` by its full path.

## Install dependencies
```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
`requirements.txt` lists the actual runtime set this project imports (fastapi, uvicorn, httpx, pydantic, numpy, pandas, scikit-learn, joblib, requests, lifelines, shap, pytest) -- verified working versions, not a dump of everything on the machine.

## How to run
**Easiest**: double-click `run_bridgeguard.bat` (or run it from any shell) -- it `cd`s into the project directory, creates `.venv` and installs requirements automatically if missing, then starts the server using `.venv\Scripts\python.exe` directly. No manual venv activation needed.

Equivalent manual command from `D:\BRIDGE GUARD`:
```
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
(`--reload` was intentionally left out of the default command -- it needs the extra `watchfiles` package, which wasn't installed to keep the dependency set minimal; add `--reload` yourself after `pip install watchfiles` if you want it during active development.)

**Dashboard URL**: `http://127.0.0.1:8000/`

## Verification performed
Started the server locally and ran the exact request `app.js` constructs (different structural/traffic/climate inputs than any example in the docs, to confirm results are genuinely computed, not copy-pasted) directly against the running server:
- `GET /` → 200, `GET /static/style.css` → 200 (page and assets actually serve)
- `POST /predict` with a Punjab/Steel/42-year-old bridge → 200, every field the dashboard reads (`health_score`, `category`, `confidence`, `current_condition`, `5_year_prediction`, `10_year_prediction`, `rul_estimate`, `rul_reliability_flag`, `survival_10yr_probability`, `top_risk_factors`, `shap_explanation`, `component_scores`, `prototype_disclaimer`) present and populated
- Full test suite (`pytest tests/`) re-run after the static-file wiring — all pre-existing tests still pass

**Caught and fixed one real bug this way**: the RUL range was printing backwards ("approximately 12-6 years") — `src/pipeline/predict.py` had `{hi}-{lo}` instead of `{lo}-{hi}` in the display string. Fixed (display formatting only, not the underlying survival calculation), re-verified: now prints "approximately 6-12 years" (ascending). This is exactly the kind of integration bug that only surfaces by actually exercising the endpoint, not by reading the code.

**Browser-based click-through verification was not available** in this environment (the Chrome extension used for browser automation wasn't connected — this session runs as a background job with no interactive browser attached). Verification was instead done by replaying the identical HTTP request the frontend JS constructs directly against the live server and inspecting the response — this validates the real integration contract (every field the UI reads is present and correctly computed) even without a visual screenshot. Documented here rather than silently claimed as a full browser test.

### Re-verified after the .venv fix
Rebuilt the environment from scratch inside `D:\BRIDGE GUARD\.venv` (see above), reinstalled `requirements.txt` into it, confirmed `import src.api.main / src.pipeline.predict / src.scoring.health_score / src.models.survival` all succeed under that interpreter, started the server with `.venv\Scripts\python.exe -m uvicorn ...`, and re-ran every check above against it: `GET /` → 200, `GET /static/style.css` → 200, `POST /predict` (Maharashtra/PSC/18yr bridge, a third distinct example) → 200 with all required fields present, and `pytest tests/` → 20/20 passed using `.venv\Scripts\python.exe` specifically (not whatever Python PATH happens to resolve to).

## Not built (per scope)
No React/Vue/build tooling, no authentication, no database, no Docker, no cloud deployment.
