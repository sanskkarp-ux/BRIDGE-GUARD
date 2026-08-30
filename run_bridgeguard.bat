@echo off
REM BridgeGuard AI -- Windows startup script.
REM Uses the project's own .venv (not whatever "python" PATH happens to resolve
REM to -- see docs/frontend.md for why that matters on this machine).

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Creating it and installing requirements...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
)

echo Starting BridgeGuard AI at http://127.0.0.1:8000/
".venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
