import os
import json
from fastapi import FastAPI, HTTPException, Header
from main import run_agent

app = FastAPI(title="Investing Agent", version="1.0")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "agent_output.json")
TASKS_API_KEY = os.getenv("TASKS_API_KEY", "").strip()


def require_api_key(x_api_key: str | None) -> None:
    # If you forget to set TASKS_API_KEY, we fail closed (secure by default)
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
    return run_agent()


@app.get("/last")
def last():
    # last output is safe to keep public; if you want it protected, say so.
    if not os.path.exists(OUTPUT_PATH):
        raise HTTPException(status_code=404, detail=f"{OUTPUT_PATH} not found yet. Call POST /run first.")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/tasks/daily")
def tasks_daily(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    return run_agent()
