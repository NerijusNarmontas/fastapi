from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import json
from datetime import datetime, timezone
import uuid

import requests  # make sure requests is in requirements.txt


app = FastAPI()

DATA_DIR = "data"
BRIEF_PATH = os.path.join(DATA_DIR, "daily_brief.json")
PORTFOLIO_PATH = os.path.join(DATA_DIR, "portfolio.json")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, payload):
    ensure_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    brief = load_json(
        BRIEF_PATH,
        {
            "date": now_utc(),
            "material_events": [],
            "technical_exceptions": [],
            "action_required": ["Run /tasks/daily once to generate today’s brief."],
        },
    )

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

        <div class="card">
          <h2>Material Events</h2>
          <ul>{li(brief.get("material_events", []))}</ul>
        </div>

        <div class="card">
          <h2>Technical Exceptions</h2>
          <ul>{li(brief.get("technical_exceptions", []))}</ul>
        </div>

        <div class="card">
          <h2>Action Required</h2>
          <ul>{li(brief.get("action_required", []))}</ul>
          <div class="muted">Run <code>/tasks/daily</code> once to refresh data.</div>
        </div>

        <div class="card">
          <h2>Portfolio</h2>
          <table>
            <thead>
              <tr><th>Ticker</th><th>Weight %</th><th>P&L %</th><th>Thesis</th><th>Tech</th></tr>
            </thead>
            <tbody>
              {rows if rows else "<tr><td colspan='5'>No positions saved yet.</td></tr>"}
            </tbody>
          </table>
          <div class="muted">Note: ticker may show instrumentId until we map instrumentId → symbol.</div>
        </div>

        <div class="muted">API: <code>/api/daily-brief</code> • <code>/api/portfolio</code></div>
      </body>
    </html>
    """


@app.get("/api/daily-brief")
def api_daily_brief():
    return load_json(
        BRIEF_PATH,
        {"date": now_utc(), "material_events": [], "technical_exceptions": [], "action_required": []},
    )


@app.get("/api/portfolio")
def api_portfolio():
    return load_json(PORTFOLIO_PATH, {"date": now_utc(), "positions": []})


@app.get("/tasks/daily")
def run_daily_task():
    # Key presence checks (does not reveal secrets)
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))

    etoro_api_key = os.getenv("ETORO_API_KEY", "")
    etoro_user_key = os.getenv("ETORO_USER_KEY", "")
    etoro_ok = bool(etoro_api_key and etoro_user_key)

    portfolio = {"date": now_utc(), "positions": []}
    brief = {"date": now_utc(), "material_events": [], "technical_exceptions": [], "action_required": []}

    brief["material_events"].append(
        f"System check: OpenAI key loaded={openai_ok}, eToro keys loaded={etoro_ok}."
    )

    if not etoro_ok:
        brief["action_required"].append(
            "Add both Railway variables: ETORO_API_KEY (Public Key) and ETORO_USER_KEY (Generated Key)."
        )
        save_json(PORTFOLIO_PATH, portfolio)
        save_json(BRIEF_PATH, brief)
        return {"status": "ok", "message": "Daily task executed (missing eToro keys)."}

    # Real eToro endpoint (from your docs screenshot)
    url = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
    headers = {
        "x-api-key": etoro_api_key,
        "x-user-key": etoro_user_key,
        "x-request-id": str(uuid.uuid4()),
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)

        if r.status_code != 200:
            brief["material_events"].append(f"eToro API error: HTTP {r.status_code}")
            # show only a small snippet to avoid dumping sensitive info
            brief["material_events"].append(r.text[:300])
            brief["action_required"].append(
                "Check eToro key permissions: Generated Key must have Read (Real)."
            )
        else:
            data = r.json()
            client = data.get("clientPortfolio") or {}
            positions = client.get("positions") or []

            brief["material_events"].append(
                f"Pulled eToro portfolio successfully. Positions: {len(positions)}"
            )

            # Minimal mapping for dashboard table
            mapped = []
            for p in positions:
                mapped.append(
                    {
                        # We'll map this to ticker symbol later
                        "ticker": str(p.get("instrumentId") or p.get("cid") or "NA"),
                        "weight_pct": "",
                        "pnl_pct": "",
                        "thesis_score": "",
                        "tech_status": "",
                    }
                )
            portfolio["positions"] = mapped

    except Exception as e:
        brief["material_events"].append(f"eToro request failed: {type(e).__name__}: {e}")
        brief["action_required"].append("Open Railway Logs to see details.")

    # Placeholder until we add real technicals + OpenAI summary
    brief["technical_exceptions"].append(
        "Next: compute RSI/MACD/MAs/Volume/ADV per ticker + generate AI brief."
    )

    save_json(PORTFOLIO_PATH, portfolio)
    save_json(BRIEF_PATH, brief)
    return {"status": "ok", "message": "Daily task executed. Open / to view dashboard."}
