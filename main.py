from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os, json
from datetime import datetime, timezone

app = FastAPI()

DATA_DIR = "data"
BRIEF_PATH = os.path.join(DATA_DIR, "daily_brief.json")
PORTFOLIO_PATH = os.path.join(DATA_DIR, "portfolio.json")

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, payload):
    ensure_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

@app.get("/", response_class=HTMLResponse)
def dashboard():
    brief = load_json(BRIEF_PATH, {
        "date": now_utc(),
        "material_events": [],
        "technical_exceptions": [],
        "action_required": ["First run not executed yet. Open /tasks/daily once."]
    })
    portfolio = load_json(PORTFOLIO_PATH, {"date": now_utc(), "positions": []})

    def li(items):
        return "".join(f"<li>{x}</li>" for x in items) if items else "<li>None</li>"

    rows = ""
    for p in portfolio.get("positions", []):
        rows += (
            "<tr>"
            f"<td>{p.get('ticker','')}</td>"
            f"<td>{p.get('weight_pct','')}</td>"
            f"<td>{p.get('pnl_pct','')}</td>"
            f"<td>{p.get('thesis_score','')}</td>"
            f"<td>{p.get('tech_status','')}</td>"
            "</tr>"
        )

    return f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Investing Agent</title>
      <style>
        body {{ font-family: -apple-system, Arial; margin: 24px; }}
        .card {{ border:1px solid #ddd; border-radius:12px; padding:16px; margin-bottom:16px; }}
        table {{ width:100%; border-collapse: collapse; }}
        th, td {{ border-bottom:1px solid #eee; padding:10px; text-align:left; }}
        .muted {{ color:#666; font-size:14px; }}
        code {{ background:#f6f6f6; padding:2px 6px; border-radius:6px; }}
      </style>
    </head>
    <body>
      <h1>My AI Investing Agent</h1>
      <div class="muted">Last update: {brief.get('date', now_utc())}</div>

      <div class="card"><h2>Material Events</h2><ul>{li(brief.get("material_events", []))}</ul></div>
      <div class="card"><h2>Technical Exceptions</h2><ul>{li(brief.get("technical_exceptions", []))}</ul></div>
      <div class="card">
        <h2>Action Required</h2><ul>{li(brief.get("action_required", []))}</ul>
        <div class="muted">Run <code>/tasks/daily</code> once to generate today’s brief.</div>
      </div>

      <div class="card">
        <h2>Portfolio (placeholder)</h2>
        <table>
          <thead><tr><th>Ticker</th><th>Weight %</th><th>P&L %</th><th>Thesis</th><th>Tech</th></tr></thead>
          <tbody>{rows if rows else "<tr><td colspan='5'>No positions saved yet.</td></tr>"}</tbody>
        </table>
      </div>

      <div class="muted">API: <code>/api/daily-brief</code> • <code>/api/portfolio</code></div>
    </body>
    </html>
    """

@app.get("/api/daily-brief")
def api_daily_brief():
    return load_json(BRIEF_PATH, {"date": now_utc(), "material_events": [], "technical_exceptions": [], "action_required": []})

@app.get("/api/portfolio")
def api_portfolio():
    return load_json(PORTFOLIO_PATH, {"date": now_utc(), "positions": []})

@app.get("/tasks/daily")
def run_daily_task():
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    etoro_ok = bool(os.getenv("ETORO_API_KEY"))

    portfolio = {
        "date": now_utc(),
        "positions": [
            {"ticker": "EQT", "weight_pct": "6.1", "pnl_pct": "-9.0", "thesis_score": "3/5", "tech_status": "Watch (placeholder)"},
            {"ticker": "CCJ", "weight_pct": "8.2", "pnl_pct": "+14.0", "thesis_score": "4/5", "tech_status": "Strong (placeholder)"},
        ],
    }
    save_json(PORTFOLIO_PATH, portfolio)

    brief = {
        "date": now_utc(),
        "material_events": [
            f"System check: OpenAI key loaded={openai_ok}, eToro key loaded={etoro_ok}.",
            "Placeholder run complete. Next: real eToro pull + AI summary."
        ],
        "technical_exceptions": ["Placeholder: RSI/MAs/volume exceptions will be computed next."],
        "action_required": ["Next: connect eToro portfolio endpoint mapping."]
    }
    save_json(BRIEF_PATH, brief)

    return {"status": "ok", "message": "Daily task executed. Open / to view dashboard."}
