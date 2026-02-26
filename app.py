import os
import json
from fastapi import FastAPI, HTTPException
from main import run_agent

app = FastAPI(title="Investing Agent", version="1.0")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "agent_output.json")

@app.get("/")
def health():
    return {"status": "ok", "service": "investing-agent"}

@app.post("/run")
def run():
    return run_agent()

@app.get("/last")
def last():
    if not os.path.exists(OUTPUT_PATH):
        raise HTTPException(status_code=404, detail=f"{OUTPUT_PATH} not found yet. Call POST /run first.")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ✅ This is the route you wanted
@app.get("/tasks/daily")
def tasks_daily():
    return run_agent()
