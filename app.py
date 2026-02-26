import os
import json
import traceback
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from main import run_agent

app = FastAPI(title="Investing Agent", version="1.0")

# IMPORTANT: write output to /tmp on Railway (writable)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/agent_output.json")
TASKS_API_KEY = os.getenv("TASKS_API_KEY", "").strip()

def require_api_key(x_api_key: str | None) -> None:
    if not TASKS_API_KEY:
        raise HTTPException(status_code=500, detail="TASKS_API_KEY is not set on the server.")
    if not x_api_key or x_api_key.strip() != TASKS_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

@app.get("/")
def health():
    return {"status": "ok", "service": "investing-agent"}

@app.post("/run")
def run(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    try:
        # tell main.py where to save output
        os.environ["OUTPUT_PATH"] = OUTPUT_PATH
        return run_agent()
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "run_agent crashed", "trace": traceback.format_exc()[:4000]},
        )

@app.get("/tasks/daily")
def tasks_daily(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    try:
        os.environ["OUTPUT_PATH"] = OUTPUT_PATH
        return run_agent()
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "run_agent crashed", "trace": traceback.format_exc()[:4000]},
        )

@app.get("/last")
def last():
    if not os.path.exists(OUTPUT_PATH):
        raise HTTPException(status_code=404, detail=f"{OUTPUT_PATH} not found yet. Call POST /run first.")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
