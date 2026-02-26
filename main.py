import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="My AI Investing Agent")

# ----------------------------
# Config (ENV VARS on Railway)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()   # eToro x-api-key
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip() # eToro x-user-key

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()       # optional (keep empty if you don't want auth yet)

# Files (Railway containers can restart; /tmp is fine for lightweight cache)
STATE_PATH = "/tmp/investing_agent_state.json"

ETORO_REAL_PNL_URL = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
ETORO_INSTRUMENTS_META_URL = "https://public-api.etoro.com/api/v1/market-data/instruments"


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
    return normalize_number(
        p.get("unrealizedPnL") or p.get("unrealized_pnl") or p.get("unrealizedPnl")
    )


def pick_initial_usd(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(
        p.get("initialAmountInDollars") or p.get("initial_amount_usd") or p.get("initialAmount")
    )


def pick_instrument_id(p: Dict[str, Any]) -> Optional[int]:
    iid = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
    try:
        return int(iid) if iid is not None else None
    except Exception:
        return None


def chunk_list(items: List[int], size: int = 100) -> List[List[int]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def extract_positions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    eToro API commonly returns:
    {
      "clientPortfolio": {
        "positions": [ ...lots... ]
      }
    }
    We'll try multiple fallbacks.
    """
    if not isinstance(payload, dict):
        return []
    cp = payload.get("clientPortfolio") or payload.get("ClientPortfolio") or {}
    if isinstance(cp, dict) and isinstance(cp.get("positions"), list):
        return cp["positions"]
    if isinstance(payload.get("positions"), list):
        return payload["positions"]
    return []


def aggregate_positions_by_instrument(raw_positions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    eToro returns LOTS (many rows per same instrument). You want UNIQUE instruments (~65).
    This groups lots by instrumentID and sums notional and PnL.
    """
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
# eToro calls
# ----------------------------
def etoro_headers() -> Dict[str, str]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        return {}
    return {
        "x-api-key": ETORO_API_KEY,
        "x-user-key": ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
    }


async def etoro_get_real_pnl() -> Dict[str, Any]:
    """
    GET https://public-api.etoro.com/api/v1/trading/info/real/pnl
    Required headers: x-api-key, x-user-key, x-request-id
    """
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        raise HTTPException(status_code=400, detail="Missing ETORO_API_KEY or ETORO_USER_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(ETORO_REAL_PNL_URL, headers=etoro_headers())
        if r.status_code >= 400:
            try:
                payload = r.json()
            except Exception:
                payload = {"text": r.text}
            raise HTTPException(
                status_code=r.status_code,
                detail={"etoro_error": True, "status": r.status_code, "payload": payload},
            )
        return r.json()


async def etoro_get_instruments_metadata(instrument_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    GET https://public-api.etoro.com/api/v1/market-data/instruments?instrumentIds=1,2,3

    Returns mapping:
      instrumentId -> { symbolFull, instrumentDisplayName, exchangeId, instrumentTypeId, ... }

    We store symbolFull as the ticker to display (e.g., EQT, CCJ, ...).
    """
    if not instrument_ids:
        return {}

    if not ETORO_API_KEY or not ETORO_USER_KEY:
        raise HTTPException(status_code=400, detail="Missing ETORO_API_KEY or ETORO_USER_KEY")

    out: Dict[int, Dict[str, Any]] = {}

    batches = chunk_list(sorted(set(int(x) for x in instrument_ids)), size=100)
    async with httpx.AsyncClient(timeout=30) as client:
        for batch in batches:
            params = {"instrumentIds": ",".join(str(i) for i in batch)}
            r = await client.get(ETORO_INSTRUMENTS_META_URL, headers=etoro_headers(), params=params)
            if r.status_code >= 400:
                # Do not break whole run if one batch fails; just skip it
                continue
            data = r.json() if isinstance(r.json(), dict) else {}
            items = data.get("instrumentDisplayDatas") or data.get("instrumentDisplayData") or []
            if not isinstance(items, list):
                continue

            for it in items:
                try:
                    iid = int(it.get("instrumentId") or it.get("instrumentID"))
                except Exception:
                    continue
                out[iid] = it

    return out


def build_portfolio_rows_from_aggregates(
    agg: List[Dict[str, Any]],
    instrument_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build rows for dashboard + /api/portfolio from aggregated positions.
    ticker = symbolFull if available else instrumentID
    """
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0

    rows: List[Dict[str, Any]] = []
    for a in agg:
        instrument_id = a.get("instrumentID")
        iid = int(instrument_id) if instrument_id is not None else None

        meta = instrument_map.get(iid, {}) if iid is not None else {}
        symbol = (meta.get("symbolFull") or meta.get("SymbolFull") or "").strip()
        display_name = (meta.get("instrumentDisplayName") or meta.get("displayName") or "").strip()

        initial = normalize_number(a.get("initialAmountInDollars"))
        unreal = normalize_number(a.get("unrealizedPnL"))

        weight_pct = None
        if total_initial > 0 and initial is not None and initial > 0:
            weight_pct = (initial / total_initial) * 100.0

        pnl_pct = None
        if unreal is not None and initial is not None and initial != 0:
            pnl_pct = (unreal / initial) * 100.0

        rows.append({
            "ticker": symbol if symbol else (str(iid) if iid is not None else "NA"),
            "name": display_name,
            "instrumentID": str(iid) if iid is not None else "",
            "lots": str(a.get("lots", "")),
            "weight_pct": (f"{weight_pct:.2f}" if weight_pct is not None else ""),
            "pnl_pct": (f"{pnl_pct:.2f}" if pnl_pct is not None else ""),
            "thesis_score": "",
            "tech_status": "",
        })

    return rows


# ----------------------------
# OpenAI summary (optional)
# ----------------------------
async def generate_openai_brief(portfolio_rows: List[Dict[str, Any]]) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI key not set. (Skipping AI brief.)"

    top = portfolio_rows[:25]
    lines = []
    for r in top:
        lines.append(f"{r['ticker']}: weight={r['weight_pct']}% pnl={r['pnl_pct']}%")
    portfolio_text = "\n".join(lines) if lines else "(no positions)"

    prompt = (
        "You are an investing assistant. READ-ONLY.\n"
        "Given today's portfolio snapshot (tickers, weights, pnl%), write a short daily brief:\n"
        "- 5 bullet material events to check today (macro/sector angles)\n"
        "- 5 bullet risk items\n"
        "- 5 bullet watchlist items\n"
        "Do NOT give buy/sell instructions. Do NOT give price predictions.\n\n"
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
        output_text = None
        if isinstance(data, dict):
            output_text = data.get("output_text")

            if not output_text and isinstance(data.get("output"), list):
                chunks = []
                for item in data["output"]:
                    if not isinstance(item, dict):
                        continue
                    for c in item.get("content", []) or []:
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
    action_required = state.get("action_required") or ["Run /tasks/daily once to generate today's brief."]

    portfolio = state.get("positions") or []
    stats = state.get("stats") or {}
    ai_brief = state.get("ai_brief") or ""

    def bullets(items: List[str]) -> str:
        if not items:
            return "<ul><li>None</li></ul>"
        lis = "".join([f"<li>{x}</li>" for x in items])
        return f"<ul>{lis}</ul>"

    rows_html = ""
    if portfolio:
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
    else:
        rows_html = "<tr><td colspan='6'>No positions saved yet.</td></tr>"

    lots_count = stats.get("lots_count", "")
    uniq_count = stats.get("unique_instruments_count", "")

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
        <p><b>eToro:</b> Lots = {lots_count} | Unique instruments = {uniq_count}</p>

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
        <p><i>Now showing mapped tickers when available (fallback to instrumentID if eToro metadata is missing).</i></p>

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

        <p>API: <code>/api/daily-brief</code> • <code>/api/portfolio</code> • Debug: <code>/debug/instrument-map</code></p>
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
        f"System check: OpenAI key loaded={'True' if bool(OPENAI_API_KEY) else 'False'}, "
        f"eToro keys loaded={'True' if (bool(ETORO_API_KEY) and bool(ETORO_USER_KEY)) else 'False'}."
    )

    raw_positions: List[Dict[str, Any]] = []
    agg_positions: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {"lots_count": 0, "unique_instruments_count": 0}

    # Fetch eToro portfolio
    try:
        payload = await etoro_get_real_pnl()
        raw_positions = extract_positions(payload)
        agg_positions, stats = aggregate_positions_by_instrument(raw_positions)

        material_events.append(
            f"Pulled eToro successfully. Lots: {stats['lots_count']} | Unique instruments: {stats['unique_instruments_count']}"
        )
    except HTTPException as e:
        material_events.append(f"eToro API error: HTTP {e.status_code}")
        material_events.append(str(e.detail))
        raw_positions = []
        agg_positions = []
        stats = {"lots_count": 0, "unique_instruments_count": 0}

    # Symbol mapping step
    instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]
    prev_map = state.get("instrument_map") or {}
    # convert prev_map keys back to int for internal use
    instrument_map: Dict[int, Dict[str, Any]] = {}
    try:
        for k, v in prev_map.items():
            instrument_map[int(k)] = v
    except Exception:
        instrument_map = {}

    try:
        fetched_meta = await etoro_get_instruments_metadata(instrument_ids)
        # merge/overwrite cached map
        instrument_map.update(fetched_meta)
        material_events.append(f"Instrument metadata fetched: {len(fetched_meta)} mapped.")
    except Exception as e:
        material_events.append(f"Instrument metadata fetch failed (will use cache/fallback): {str(e)}")

    # Build rows (unique instruments, with mapped tickers)
    portfolio_rows = build_portfolio_rows_from_aggregates(agg_positions, instrument_map) if agg_positions else []

    # Mapping stats
    mapped = 0
    for r in portfolio_rows:
        # mapped if ticker is not numeric
        t = (r.get("ticker") or "").strip()
        if t and not t.isdigit():
            mapped += 1
    technical_exceptions.append(
        f"Symbol mapping: {mapped}/{len(portfolio_rows)} tickers mapped (rest fallback to instrumentID)."
    )
    technical_exceptions.append("Next steps: RSI/MACD/MAs/Volume/ADV per ticker + liquidity flags.")

    # AI Brief
    if portfolio_rows:
        ai_brief = await generate_openai_brief(portfolio_rows)
    else:
        ai_brief = "No positions found yet (or eToro call failed)."

    if not action_required:
        action_required.append("None")

    # Save mapping as JSON-safe keys (strings)
    instrument_map_jsonsafe = {str(k): v for k, v in instrument_map.items()}

    state.update(
        {
            "date": utc_now_iso(),
            "material_events": material_events,
            "technical_exceptions": technical_exceptions,
            "action_required": action_required,
            "positions": portfolio_rows,       # aggregated view for dashboard
            "stats": stats,                    # lots vs unique counts
            "ai_brief": ai_brief,
            "raw_positions_count": len(raw_positions),
            "instrument_map": instrument_map_jsonsafe,
        }
    )
    save_state(state)

    return {
        "status": "ok",
        "message": "Daily task executed. Open / to view dashboard.",
        "lots": stats.get("lots_count"),
        "unique_instruments": stats.get("unique_instruments_count"),
        "mapped_symbols": mapped,
    }


@app.get("/api/portfolio")
async def api_portfolio():
    state = load_state()
    return JSONResponse(
        {
            "date": state.get("date") or utc_now_iso(),
            "stats": state.get("stats") or {},
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
            "stats": state.get("stats") or {},
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
    return {
        "note": "This shows only keys, not values.",
        "raw_positions_count": len(positions),
        "raw_first_position_keys": {"keys": keys, "total_keys": len(keys)},
    }


@app.get("/debug/instrument-map")
async def debug_instrument_map():
    """
    Shows current cached instrumentID -> symbolFull mapping summary.
    """
    state = load_state()
    m = state.get("instrument_map") or {}
    # show only a small sample
    sample = []
    for k in list(m.keys())[:50]:
        it = m.get(k) or {}
        sample.append({
            "instrumentID": k,
            "symbolFull": it.get("symbolFull") or "",
            "name": it.get("instrumentDisplayName") or it.get("displayName") or "",
        })
    return {
        "cached_count": len(m),
        "sample_first_50": sample,
    }
