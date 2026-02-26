import os
import json
import traceback
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from main import run_agent

app = FastAPI(title="Investing Agent", version="1.0")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/agent_output.json")
TASKS_API_KEY = os.getenv("TASKS_API_KEY", "").strip()

def require_api_key(x_api_key: str | None) -> None:
    if not TASKS_API_KEY:
        raise HTTPException(status_code=500, detail="TASKS_API_KEY is not set on the server.")
    if not x_api_key or x_api_key.strip() != TASKS_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ✅ Catch ANY unhandled error and return JSON trace + log it to Railway logs
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    trace = traceback.format_exc()
    print("UNHANDLED ERROR:", trace)  # shows in Railway logs
    return JSONResponse(status_code=500, content={"error": str(exc), "trace": trace[:4000]})

@app.get("/")
def health():
    return {"status": "ok", "service": "investing-agent"}

@app.post("/run")
def run(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    os.environ["OUTPUT_PATH"] = OUTPUT_PATH
    return run_agent()

@app.get("/tasks/daily")
def tasks_daily(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    os.environ["OUTPUT_PATH"] = OUTPUT_PATH
    return run_agent()

@app.get("/last")
def last():
    if not os.path.exists(OUTPUT_PATH):
        raise HTTPException(status_code=404, detail=f"{OUTPUT_PATH} not found yet. Call POST /run first.")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
