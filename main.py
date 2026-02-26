import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="My AI Investing Agent")

# ----------------------------
# Config (Railway ENV VARS)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

STATE_PATH = "/tmp/investing_agent_state.json"

ETORO_REAL_PNL_URL = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
ETORO_SEARCH_URL = "https://public-api.etoro.com/api/v1/market-data/search"


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
        pass


def require_admin(request: Request) -> None:
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


def pick_unrealized_pnl(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(p.get("unrealizedPnL") or p.get("unrealizedPnl") or p.get("unrealized_pnl"))


def pick_initial_usd(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(p.get("initialAmountInDollars") or p.get("initialAmount") or p.get("initial_amount_usd"))


def pick_instrument_id(p: Dict[str, Any]) -> Optional[int]:
    iid = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
    try:
        return int(iid) if iid is not None else None
    except Exception:
        return None


def extract_positions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    cp = payload.get("clientPortfolio") or payload.get("ClientPortfolio") or {}
    if isinstance(cp, dict) and isinstance(cp.get("positions"), list):
        return cp["positions"]
    if isinstance(payload.get("positions"), list):
        return payload["positions"]
    return []


def aggregate_positions_by_instrument(raw_positions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    buckets = defaultdict(list)
    for p in raw_positions:
        iid = pick_instrument_id(p)
        if iid is None:
            continue
        buckets[iid].append(p)

    aggregated: List[Dict[str, Any]] = []
    for iid, lots in buckets.items():
        total_initial_usd = sum(float(pick_initial_usd(x) or 0) for x in lots)
        total_unreal_pnl = sum(float(pick_unrealized_pnl(x) or 0) for x in lots)

        aggregated.append({
            "instrumentID": iid,
            "lots": len(lots),
            "initialAmountInDollars": total_initial_usd,
            "unrealizedPnL": total_unreal_pnl,
        })

    aggregated.sort(key=lambda x: x.get("initialAmountInDollars", 0) or 0, reverse=True)

    stats = {
        "lots_count": len(raw_positions),
        "unique_instruments_count": len(aggregated),
    }
    return aggregated, stats


# ----------------------------
# eToro HTTP
# ----------------------------
def etoro_headers() -> Dict[str, str]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        return {}
    return {
        "x-api-key": ETORO_API_KEY,
        "x-user-key": ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
        "user-agent": "fastapi-investing-agent/1.0",
        "accept": "application/json",
    }


async def etoro_get_real_pnl() -> Dict[str, Any]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        raise HTTPException(status_code=400, detail="Missing ETORO_API_KEY or ETORO_USER_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(ETORO_REAL_PNL_URL, headers=etoro_headers())
        if r.status_code >= 400:
            try:
                payload = r.json()
            except Exception:
                payload = {"text": r.text}
            raise HTTPException(status_code=r.status_code, detail=payload)
        return r.json()


async def etoro_search_one(field: str, value: int) -> Optional[Dict[str, Any]]:
    """
    Uses the Search endpoint as a lookup:
      /market-data/search?<field>=<value>&fields=...&pageSize=1&pageNumber=1

    Docs show Search endpoint and that it can be filtered by internal fields (guide uses internalSymbolFull). :contentReference[oaicite:1]{index=1}
    """
    params = {
        field: str(value),
        # fields is required by the API reference. :contentReference[oaicite:2]{index=2}
        "fields": "instrumentId,internalInstrumentId,internalSymbolFull,displayname,symbol,exchangeID,instrumentTypeID",
        "pageSize": "1",
        "pageNumber": "1",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(ETORO_SEARCH_URL, headers=etoro_headers(), params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            return None
        it = items[0]
        return it if isinstance(it, dict) else None


async def build_symbol_map_from_search(instrument_ids: List[int]) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Returns:
      map: { instrumentID(from pnl) -> ticker }
    Strategy:
      1) Try internalInstrumentId=<pnl_instrumentID> (most likely)
      2) If no result, try instrumentId=<pnl_instrumentID> (fallback)
    """
    debug = {"total": len(instrument_ids), "mapped": 0, "failed": 0, "samples": []}
    out: Dict[int, str] = {}

    sem = asyncio.Semaphore(10)

    async def one(iid: int):
        async with sem:
            # Try internalInstrumentId first
            it = await etoro_search_one("internalInstrumentId", iid)
            used = "internalInstrumentId"
            if not it:
                it = await etoro_search_one("instrumentId", iid)
                used = "instrumentId"

            if not it:
                debug["failed"] += 1
                if len(debug["samples"]) < 10:
                    debug["samples"].append({"instrumentID": iid, "status": "no_match"})
                return

            ticker = (it.get("internalSymbolFull") or it.get("symbol") or "").strip()
            if not ticker:
                debug["failed"] += 1
                if len(debug["samples"]) < 10:
                    debug["samples"].append({"instrumentID": iid, "status": "matched_no_ticker", "used": used, "item": it})
                return

            out[iid] = ticker
            debug["mapped"] += 1
            if len(debug["samples"]) < 10:
                debug["samples"].append({"instrumentID": iid, "ticker": ticker, "used": used})

    await asyncio.gather(*(one(i) for i in sorted(set(instrument_ids))))
    return out, debug


def build_portfolio_rows(agg: List[Dict[str, Any]], symbol_map: Dict[int, str]) -> List[Dict[str, Any]]:
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0

    rows: List[Dict[str, Any]] = []
    for a in agg:
        iid = int(a["instrumentID"])
        ticker = symbol_map.get(iid) or str(iid)

        initial = normalize_number(a.get("initialAmountInDollars"))
        unreal = normalize_number(a.get("unrealizedPnL"))

        weight_pct = (initial / total_initial * 100.0) if (total_initial > 0 and initial and initial > 0) else None
        pnl_pct = (unreal / initial * 100.0) if (initial and initial != 0 and unreal is not None) else None

        rows.append({
            "ticker": ticker,
            "instrumentID": str(iid),
            "lots": str(a.get("lots", "")),
            "weight_pct": f"{weight_pct:.2f}" if weight_pct is not None else "",
            "pnl_pct": f"{pnl_pct:.2f}" if pnl_pct is not None else "",
            "thesis_score": "",
            "tech_status": "",
        })

    return rows


# ----------------------------
# OpenAI brief (optional)
# ----------------------------
async def generate_openai_brief(portfolio_rows: List[Dict[str, Any]]) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI key not set. (Skipping AI brief.)"

    top = portfolio_rows[:25]
    lines = [f"{r['ticker']}: weight={r['weight_pct']}% pnl={r['pnl_pct']}%" for r in top]
    portfolio_text = "\n".join(lines) if lines else "(no positions)"

    prompt = (
        "You are an investing assistant. READ-ONLY.\n"
        "Do NOT give buy/sell instructions. Do NOT predict prices.\n"
        "Write a short daily brief: material events, risks, watchlist.\n\n"
        f"Portfolio:\n{portfolio_text}\n"
    )

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "gpt-5-mini", "input": prompt, "max_output_tokens": 500}

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=body)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"text": r.text}
            return f"OpenAI error {r.status_code}: {err}"
        data = r.json()
        return data.get("output_text") or "AI brief generated (output_text missing)."


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    state = load_state()

    last_update = state.get("date") or utc_now_iso()
    material_events = state.get("material_events") or []
    technical_exceptions = state.get("technical_exceptions") or []
    action_required = state.get("action_required") or ["Run /tasks/daily once to refresh data."]

    portfolio = state.get("positions") or []
    stats = state.get("stats") or {}
    ai_brief = state.get("ai_brief") or ""

    def bullets(items: List[str]) -> str:
        lis = "".join([f"<li>{x}</li>" for x in items]) if items else "<li>None</li>"
        return f"<ul>{lis}</ul>"

    rows_html = ""
    for r in portfolio[:200]:
        rows_html += (
            "<tr>"
            f"<td>{r.get('ticker','')}</td>"
            f"<td>{r.get('lots','')}</td>"
            f"<td>{r.get('weight_pct','')}</td>"
            f"<td>{r.get('pnl_pct','')}</td>"
            f"<td>{r.get('thesis_score','')}</td>"
            f"<td>{r.get('tech_status','')}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html = "<tr><td colspan='6'>No positions saved yet.</td></tr>"

    lots_count = stats.get("lots_count", "")
    uniq_count = stats.get("unique_instruments_count", "")
    mapped = state.get("symbol_map_stats", {}).get("mapped", "")

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>My AI Investing Agent</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; margin: 24px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
          th {{ background: #f5f5f5; text-align: left; }}
          code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 4px; }}
        </style>
      </head>
      <body>
        <h1>My AI Investing Agent</h1>
        <p><b>Last update:</b> {last_update}</p>
        <p><b>eToro:</b> Lots = {lots_count} | Unique instruments = {uniq_count} | Mapped tickers = {mapped}</p>

        <h2>Material Events</h2>
        {bullets(material_events)}

        <h2>Technical Exceptions</h2>
        {bullets(technical_exceptions)}

        <h2>Action Required</h2>
        {bullets(action_required)}

        <p>Run <code>/tasks/daily</code> once to refresh data.</p>

        <h2>AI Brief</h2>
        <pre style="white-space: pre-wrap; background:#fafafa; border:1px solid #eee; padding:12px;">{ai_brief or "No brief yet. Run /tasks/daily."}</pre>

        <h2>Portfolio (unique instruments)</h2>
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Lots</th>
              <th>Weight %</th>
              <th>P&amp;L %</th>
              <th>Thesis</th>
              <th>Tech</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>

        <p>API: <code>/api/portfolio</code> • <code>/api/daily-brief</code> • Debug: <code>/debug/symbol-map-last</code></p>
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

    material_events.append(
        f"System check: OpenAI={'True' if bool(OPENAI_API_KEY) else 'False'}, "
        f"eToro keys={'True' if (bool(ETORO_API_KEY) and bool(ETORO_USER_KEY)) else 'False'}."
    )

    # Fetch portfolio
    try:
        payload = await etoro_get_real_pnl()
        raw_positions = extract_positions(payload)
        agg_positions, stats = aggregate_positions_by_instrument(raw_positions)
        material_events.append(
            f"Pulled eToro successfully. Lots: {stats['lots_count']} | Unique instruments: {stats['unique_instruments_count']}"
        )
    except HTTPException as e:
        state.update({
            "date": utc_now_iso(),
            "material_events": material_events + [f"eToro API error: {e.status_code}", str(e.detail)],
            "technical_exceptions": technical_exceptions,
            "action_required": ["Fix eToro keys or API access."],
            "positions": [],
            "stats": {"lots_count": 0, "unique_instruments_count": 0},
            "ai_brief": "",
        })
        save_state(state)
        return {"status": "error", "detail": e.detail}

    instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]

    # Build symbol map via Search endpoint (the correct mapper)
    symbol_map, symbol_map_stats = await build_symbol_map_from_search(instrument_ids)
    material_events.append(f"Symbol mapping via search: {symbol_map_stats['mapped']}/{symbol_map_stats['total']} mapped.")

    portfolio_rows = build_portfolio_rows(agg_positions, symbol_map)

    technical_exceptions.append("Next: RSI/MACD/MAs/Volume/ADV + liquidity flags.")
    ai_brief = await generate_openai_brief(portfolio_rows) if portfolio_rows else "No positions."

    state.update({
        "date": utc_now_iso(),
        "material_events": material_events,
        "technical_exceptions": technical_exceptions,
        "action_required": ["None"],
        "positions": portfolio_rows,
        "stats": stats,
        "ai_brief": ai_brief,
        "symbol_map": {str(k): v for k, v in symbol_map.items()},
        "symbol_map_stats": symbol_map_stats,
        "symbol_map_last_debug": symbol_map_stats,
    })
    save_state(state)

    mapped = symbol_map_stats.get("mapped", 0)
    return {"status": "ok", "lots": stats["lots_count"], "unique_instruments": stats["unique_instruments_count"], "mapped_symbols": mapped}


@app.get("/api/portfolio")
async def api_portfolio():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "stats": state.get("stats") or {},
        "positions": state.get("positions") or [],
    })


@app.get("/api/daily-brief")
async def api_daily_brief():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "material_events": state.get("material_events") or [],
        "technical_exceptions": state.get("technical_exceptions") or [],
        "action_required": state.get("action_required") or [],
        "ai_brief": state.get("ai_brief") or "",
        "stats": state.get("stats") or {},
    })


@app.get("/debug/symbol-map-last")
async def debug_symbol_map_last():
    state = load_state()
    return JSONResponse(state.get("symbol_map_last_debug") or {"note": "Run /tasks/daily first."})
