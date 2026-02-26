import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="My AI Investing Agent")

# ----------------------------
# Config (ENV VARS on Railway)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()       # eToro x-api-key
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()     # eToro x-user-key
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()           # optional (keep empty if you don't want auth yet)

# Files (Railway containers can restart; /tmp is fine for lightweight cache)
STATE_PATH = "/tmp/investing_agent_state.json"

# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        # If disk write fails, we still keep it running
        pass


def require_admin(request: Request) -> None:
    # Optional: if ADMIN_TOKEN set, require header
    if ADMIN_TOKEN:
        token = request.headers.get("x-admin-token", "")
        if token != ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Missing/invalid x-admin-token")


def normalize_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def pick_ticker(p: Dict[str, Any]) -> str:
    """
    eToro returns keys like:
      CID, instrumentID, unrealizedPnL, initialAmountInDollars, amount, etc.
    Your debug shows CID is present - use it first.
    """
    ticker = (
        p.get("symbol")
        or p.get("ticker")
        or p.get("CID")                # ✅ from your /debug/position-keys output
        or p.get("cid")
        or p.get("instrumentID")       # ✅ from your output
        or p.get("instrumentId")
        or p.get("InstrumentId")
        or (p.get("instrument") or {}).get("symbol")
        or (p.get("instrument") or {}).get("ticker")
        or (p.get("position") or {}).get("CID")
        or (p.get("position") or {}).get("instrumentID")
        or "NA"
    )
    return str(ticker)


def pick_unrealized_pnl(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(
        p.get("unrealizedPnL")
        or p.get("unrealized_pnl")
        or p.get("unrealizedPnl")
    )


def pick_initial_usd(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(
        p.get("initialAmountInDollars")
        or p.get("initial_amount_usd")
        or p.get("initialAmount")
    )


def pick_amount_usd(p: Dict[str, Any]) -> Optional[float]:
    # Some payloads have "amount" as notional-ish
    return normalize_number(p.get("amount"))


# ----------------------------
# eToro fetch
# ----------------------------
async def etoro_get_real_pnl() -> Dict[str, Any]:
    """
    Based on the eToro API portal page you showed:
      GET https://public-api.etoro.com/api/v1/trading/info/real/pnl
    Required headers:
      x-api-key, x-user-key, x-request-id
    """
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        raise HTTPException(status_code=400, detail="Missing ETORO_API_KEY or ETORO_USER_KEY")

    url = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
    headers = {
        "x-api-key": ETORO_API_KEY,
        "x-user-key": ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)

    if r.status_code >= 400:
        # return richer error so you see what happened on dashboard
        try:
            payload = r.json()
        except Exception:
            payload = {"text": r.text}
        raise HTTPException(
            status_code=r.status_code,
            detail={"etoro_error": True, "status": r.status_code, "payload": payload},
        )

    return r.json()


def extract_positions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    The example response in the portal shows:
      { "clientPortfolio": { "positions": [ ... ] } }
    We'll try multiple fallbacks just in case.
    """
    if not isinstance(payload, dict):
        return []

    cp = payload.get("clientPortfolio") or payload.get("ClientPortfolio") or {}
    if isinstance(cp, dict) and isinstance(cp.get("positions"), list):
        return cp["positions"]

    # fallback if positions at root
    if isinstance(payload.get("positions"), list):
        return payload["positions"]

    return []


def build_portfolio_rows(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build rows for the dashboard + /api/portfolio
    - ticker: CID first
    - weight_pct: based on initialAmountInDollars where possible
    - pnl_pct: unrealizedPnL / initialAmountInDollars * 100
    """
    # Compute totals for weights
    initials: List[float] = []
    for p in positions:
        v = pick_initial_usd(p)
        if v is not None and v > 0:
            initials.append(v)

    total_initial = sum(initials) if initials else 0.0

    rows: List[Dict[str, Any]] = []
    for p in positions:
        ticker = pick_ticker(p)

        unreal = pick_unrealized_pnl(p)
        initial = pick_initial_usd(p)

        pnl_pct = None
        if unreal is not None and initial is not None and initial != 0:
            pnl_pct = (unreal / initial) * 100.0

        weight_pct = None
        if total_initial > 0 and initial is not None and initial > 0:
            weight_pct = (initial / total_initial) * 100.0

        rows.append(
            {
                "ticker": ticker,
                "weight_pct": (f"{weight_pct:.2f}" if weight_pct is not None else ""),
                "pnl_pct": (f"{pnl_pct:.2f}" if pnl_pct is not None else ""),
                "thesis_score": "",   # placeholder (we’ll add your thesis rules later)
                "tech_status": "",    # placeholder (RSI/MAs later)
            }
        )

    # Sort: biggest weight first (if available), else keep original
    def sort_key(r: Dict[str, Any]):
        try:
            return -float(r["weight_pct"])
        except Exception:
            return 0.0

    rows.sort(key=sort_key)
    return rows


# ----------------------------
# OpenAI summary (optional)
# ----------------------------
async def generate_openai_brief(portfolio_rows: List[Dict[str, Any]]) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI key not set. (Skipping AI brief.)"

    # Keep prompt compact and safe.
    top = portfolio_rows[:25]
    lines = []
    for r in top:
        lines.append(f"{r['ticker']}: weight={r['weight_pct']}% pnl={r['pnl_pct']}%")
    portfolio_text = "\n".join(lines) if lines else "(no positions)"

    prompt = (
        "You are an investing assistant. "
        "Given today's portfolio snapshot (tickers, weights, pnl%), write a short daily brief:\n"
        "- 5 bullet material events to check today (macro/sector angles)\n"
        "- 5 bullet risk items\n"
        "- 5 bullet actions (trim/add/watch)\n"
        "Do not give financial advice; use cautious language.\n\n"
        f"Portfolio:\n{portfolio_text}\n"
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "gpt-5-mini",
        "input": prompt,
        "max_output_tokens": 500,
    }

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=body)

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        return f"OpenAI error {r.status_code}: {err}"

    data = r.json()

    # Responses API: try common extraction patterns
    # Some payloads return output_text under convenience field; fallback to raw JSON dump.
    output_text = None
    if isinstance(data, dict):
        output_text = data.get("output_text")
        if not output_text and isinstance(data.get("output"), list):
            # Try to stitch content blocks
            chunks = []
            for item in data["output"]:
                for c in item.get("content", []) if isinstance(item, dict) else []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        chunks.append(c.get("text", ""))
            if chunks:
                output_text = "\n".join([x for x in chunks if x])

    return output_text or "AI brief generated (but could not parse output_text)."


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    state = load_state()
    last_update = state.get("date") or utc_now_iso()

    material_events = state.get("material_events") or []
    technical_exceptions = state.get("technical_exceptions") or []
    action_required = state.get("action_required") or [
        "Run /tasks/daily once to generate today's brief."
    ]

    portfolio = state.get("positions") or []
    ai_brief = state.get("ai_brief") or ""

    # basic HTML, no templates
    def bullets(items: List[str]) -> str:
        if not items:
            return "<ul><li>None</li></ul>"
        lis = "".join([f"<li>{x}</li>" for x in items])
        return f"<ul>{lis}</ul>"

    rows_html = ""
    if portfolio:
        for r in portfolio[:150]:
            rows_html += (
                "<tr>"
                f"<td>{r.get('ticker','')}</td>"
                f"<td>{r.get('weight_pct','')}</td>"
                f"<td>{r.get('pnl_pct','')}</td>"
                f"<td>{r.get('thesis_score','')}</td>"
                f"<td>{r.get('tech_status','')}</td>"
                "</tr>"
            )
    else:
        rows_html = (
            "<tr><td colspan='5'>No positions saved yet.</td></tr>"
        )

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>My AI Investing Agent</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; }}
        h1 {{ margin-bottom: 6px; }}
        .muted {{ color: #666; font-size: 14px; }}
        .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 18px; margin: 16px 0; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ text-align: left; border-bottom: 1px solid #eee; padding: 8px; }}
        code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }}
        .brief {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #eee; padding: 12px; border-radius: 10px; }}
      </style>
    </head>
    <body>
      <h1>My AI Investing Agent</h1>
      <div class="muted">Last update: {last_update}</div>

      <div class="card">
        <h2>Material Events</h2>
        {bullets(material_events)}
      </div>

      <div class="card">
        <h2>Technical Exceptions</h2>
        {bullets(technical_exceptions)}
      </div>

      <div class="card">
        <h2>Action Required</h2>
        {bullets(action_required)}
        <div class="muted">Run <code>/tasks/daily</code> once to refresh data.</div>
      </div>

      <div class="card">
        <h2>AI Brief</h2>
        <div class="brief">{ai_brief or "No brief yet. Run /tasks/daily."}</div>
      </div>

      <div class="card">
        <h2>Portfolio</h2>
        <div class="muted">Note: ticker uses eToro <code>CID</code>. If you still see NA, open <code>/debug/position-keys</code>.</div>
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Weight %</th>
              <th>P&L %</th>
              <th>Thesis</th>
              <th>Tech</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        <div class="muted" style="margin-top:10px;">
          API: <code>/api/daily-brief</code> • <code>/api/portfolio</code>
        </div>
      </div>

    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/tasks/daily")
async def run_daily():
    state = load_state()
    material_events: List[str] = []
    technical_exceptions: List[str] = []
    action_required: List[str] = []

    # System check
    material_events.append(
        f"System check: OpenAI key loaded={'True' if bool(OPENAI_API_KEY) else 'False'}, "
        f"eToro keys loaded={'True' if (bool(ETORO_API_KEY) and bool(ETORO_USER_KEY)) else 'False'}."
    )

    # Fetch eToro portfolio
    try:
        payload = await etoro_get_real_pnl()
        positions = extract_positions(payload)
        material_events.append(f"Pulled eToro portfolio successfully. Positions: {len(positions)}")
    except HTTPException as e:
        # show exact detail in dashboard
        material_events.append(f"eToro API error: HTTP {e.status_code}")
        material_events.append(str(e.detail))
        positions = []

    # Build rows
    portfolio_rows = build_portfolio_rows(positions) if positions else []

    # If still NA, prompt user to check keys
    if portfolio_rows and all(r["ticker"] == "NA" for r in portfolio_rows[:20]):
        action_required.append("Ticker still NA: open /debug/position-keys and confirm CID exists (it should).")

    # AI Brief
    ai_brief = ""
    if portfolio_rows:
        technical_exceptions.append("Next: compute RSI/MACD/MAs/Volume/ADV per ticker + generate AI brief.")
        ai_brief = await generate_openai_brief(portfolio_rows)
    else:
        ai_brief = "No positions found yet (or eToro call failed)."

    if not action_required:
        action_required.append("None")

    state.update(
        {
            "date": utc_now_iso(),
            "material_events": material_events,
            "technical_exceptions": technical_exceptions,
            "action_required": action_required,
            "positions": portfolio_rows,
            "ai_brief": ai_brief,
        }
    )
    save_state(state)
    return {"status": "ok", "message": "Daily task executed. Open / to view dashboard."}


@app.get("/api/portfolio")
async def api_portfolio():
    state = load_state()
    return JSONResponse(
        {
            "date": state.get("date") or utc_now_iso(),
            "positions": state.get("positions") or [],
        }
    )


@app.get("/api/daily-brief")
async def api_daily_brief():
    state = load_state()
    return JSONResponse(
        {
            "date": state.get("date") or utc_now_iso(),
            "material_events": state.get("material_events") or [],
            "technical_exceptions": state.get("technical_exceptions") or [],
            "action_required": state.get("action_required") or [],
            "ai_brief": state.get("ai_brief") or "",
        }
    )


@app.get("/debug/position-keys")
async def debug_position_keys():
    """
    Returns the keys eToro sends in the first position (keys only, not values).
    Safe for debugging mapping problems.
    """
    payload = await etoro_get_real_pnl()
    positions = extract_positions(payload)
    first = positions[0] if positions else {}
    keys = sorted(list(first.keys())) if isinstance(first, dict) else []
    return {"note": "This shows only keys, not values.", "raw_first_position_keys": {"keys": keys, "total_keys": len(keys)}}
