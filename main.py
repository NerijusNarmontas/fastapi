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

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()  # optional
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()                  # optional

STATE_PATH = "/tmp/investing_agent_state.json"

ETORO_REAL_PNL_URL = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
ETORO_SEARCH_URL = "https://public-api.etoro.com/api/v1/market-data/search"
# Candles:
# https://public-api.etoro.com/api/v1/market-data/instruments/{instrumentId}/history/candles/asc/OneDay/{candlesCount}
ETORO_CANDLES_URL_TMPL = "https://public-api.etoro.com/api/v1/market-data/instruments/{instrumentId}/history/candles/asc/OneDay/{count}"


# ----------------------------
# State helpers
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


# ----------------------------
# Generic helpers
# ----------------------------
def normalize_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def pick_instrument_id(p: Dict[str, Any]) -> Optional[int]:
    iid = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
    try:
        return int(iid) if iid is not None else None
    except Exception:
        return None


def pick_unrealized_pnl(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(p.get("unrealizedPnL") or p.get("unrealizedPnl") or p.get("unrealized_pnl"))


def pick_initial_usd(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(p.get("initialAmountInDollars") or p.get("initialAmount") or p.get("initial_amount_usd"))


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
    stats = {"lots_count": len(raw_positions), "unique_instruments_count": len(aggregated)}
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


async def etoro_search(params: Dict[str, str]) -> Tuple[int, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(ETORO_SEARCH_URL, headers=etoro_headers(), params=params)
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        return r.status_code, data


def _extract_ticker_from_search_item(item: Dict[str, Any]) -> str:
    for k in ("internalSymbolFull", "symbolFull", "symbol"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


async def map_instrument_ids_to_tickers_search(instrument_ids: List[int]) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Reverse mapping via /market-data/search, no fields param.
    """
    ids = sorted(set(int(x) for x in instrument_ids if x is not None))
    out: Dict[int, str] = {}
    debug = {"requested": len(ids), "mapped": 0, "failed": 0, "samples": []}

    sem = asyncio.Semaphore(10)

    async def one(iid: int):
        async with sem:
            status, data = await etoro_search({"instrumentId": str(iid), "pageSize": "5", "pageNumber": "1"})
            items = data.get("items") if isinstance(data, dict) else None
            if isinstance(items, list) and items:
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    t = _extract_ticker_from_search_item(it)
                    if t:
                        out[iid] = t
                        debug["mapped"] += 1
                        if len(debug["samples"]) < 10:
                            debug["samples"].append({"instrumentID": iid, "ticker": t, "via": "instrumentId", "status": status})
                        return

            # fallback
            status2, data2 = await etoro_search({"internalInstrumentId": str(iid), "pageSize": "5", "pageNumber": "1"})
            items2 = data2.get("items") if isinstance(data2, dict) else None
            if isinstance(items2, list) and items2:
                for it in items2:
                    if not isinstance(it, dict):
                        continue
                    t = _extract_ticker_from_search_item(it)
                    if t:
                        out[iid] = t
                        debug["mapped"] += 1
                        if len(debug["samples"]) < 10:
                            debug["samples"].append({"instrumentID": iid, "ticker": t, "via": "internalInstrumentId", "status": status2})
                        return

            debug["failed"] += 1
            if len(debug["samples"]) < 10:
                debug["samples"].append({"instrumentID": iid, "status": "no_match_or_no_ticker"})

    await asyncio.gather(*(one(i) for i in ids))
    return out, debug


async def etoro_get_daily_candles(instrument_id: int, count: int = 250) -> List[Dict[str, Any]]:
    url = ETORO_CANDLES_URL_TMPL.format(instrumentId=instrument_id, count=count)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=etoro_headers())
        if r.status_code >= 400:
            return []
        try:
            data = r.json()
        except Exception:
            return []

    # Typical response: {"candles":[{"candles":[{open,high,low,close,volume,fromDate}, ...]}]}
    if not isinstance(data, dict):
        return []
    groups = data.get("candles")
    if not isinstance(groups, list) or not groups:
        return []
    inner = groups[0].get("candles")
    if not isinstance(inner, list):
        return []
    return [c for c in inner if isinstance(c, dict)]


# ----------------------------
# Technical indicators (lightweight)
# ----------------------------
def sma(values: List[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def ema_series(values: List[float], n: int) -> List[float]:
    if not values:
        return []
    k = 2 / (n + 1)
    out = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) <= n:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-n, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100 - (100 / (1 + rs))


def macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < slow + signal:
        return None, None, None
    fast_ = ema_series(closes, fast)
    slow_ = ema_series(closes, slow)
    macd_line = [f - s for f, s in zip(fast_, slow_)]
    sig = ema_series(macd_line, signal)
    return macd_line[-1], sig[-1], (macd_line[-1] - sig[-1])


def tech_tags(t: Dict[str, Any]) -> str:
    tags = []
    r = t.get("rsi14")
    if isinstance(r, (int, float)):
        if r >= 70:
            tags.append("RSI_OB")
        elif r <= 30:
            tags.append("RSI_OS")

    above200 = t.get("above_sma200")
    if above200 is True:
        tags.append(">SMA200")
    elif above200 is False:
        tags.append("<SMA200")

    mh = t.get("macd_hist")
    if isinstance(mh, (int, float)):
        tags.append("MACD+" if mh >= 0 else "MACD-")

    illq = t.get("illiquid")
    if illq:
        tags.append("ILLQ")

    return " | ".join(tags)


# ----------------------------
# Discord (optional)
# ----------------------------
async def discord_notify(text: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(DISCORD_WEBHOOK_URL, json={"content": text})
    except Exception:
        pass


# ----------------------------
# Daily brief (deterministic + optional OpenAI)
# ----------------------------
def deterministic_brief(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No portfolio rows."

    # Sort helpers
    def fnum(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    # PnL%
    winners = sorted(rows, key=lambda r: fnum(r.get("pnl_pct")), reverse=True)[:5]
    losers = sorted(rows, key=lambda r: fnum(r.get("pnl_pct")))[:5]

    # RSI extremes
    rsis = [(r["ticker"], r.get("tech", {}).get("rsi14")) for r in rows]
    rsis_clean = [(t, float(v)) for (t, v) in rsis if isinstance(v, (int, float))]
    overbought = sorted(rsis_clean, key=lambda x: x[1], reverse=True)[:5]
    oversold = sorted(rsis_clean, key=lambda x: x[1])[:5]

    below200 = [r["ticker"] for r in rows if r.get("tech", {}).get("above_sma200") is False][:10]
    illq = [r["ticker"] for r in rows if r.get("tech", {}).get("illiquid")][:10]

    lines = []
    lines.append("DAILY BRIEF (read-only)")
    lines.append("")
    lines.append("Top winners (PnL%): " + ", ".join([f"{r['ticker']} {r.get('pnl_pct','')}" for r in winners]))
    lines.append("Top losers  (PnL%): " + ", ".join([f"{r['ticker']} {r.get('pnl_pct','')}" for r in losers]))
    lines.append("")
    if overbought:
        lines.append("RSI overbought: " + ", ".join([f"{t} {v:.1f}" for t, v in overbought]))
    if oversold:
        lines.append("RSI oversold:  " + ", ".join([f"{t} {v:.1f}" for t, v in oversold]))
    lines.append("")
    if below200:
        lines.append("Below SMA200 (watch trend): " + ", ".join(below200))
    if illq:
        lines.append("Liquidity flags (low $ADV20): " + ", ".join(illq))
    lines.append("")
    lines.append("Notes: No buy/sell instructions. Use as a checklist for research.")
    return "\n".join(lines)


async def generate_openai_brief_safe(portfolio_rows: List[Dict[str, Any]]) -> str:
    """
    If OpenAI quota isn't active, this will return an error string.
    We'll fall back to deterministic brief.
    """
    if not OPENAI_API_KEY:
        return ""

    top = portfolio_rows[:30]
    compact = []
    for r in top:
        compact.append({
            "ticker": r.get("ticker"),
            "weight_pct": r.get("weight_pct"),
            "pnl_pct": r.get("pnl_pct"),
            "tech": r.get("tech"),
        })

    prompt = (
        "You are an investing assistant. READ-ONLY.\n"
        "No buy/sell instructions. No price prediction.\n"
        "Summarize: key portfolio signals today, technical extremes (RSI/MACD/SMA200), liquidity flags.\n"
        "Output: short bullets + one paragraph conclusion.\n"
        f"Data:\n{json.dumps(compact, ensure_ascii=False)}\n"
    )

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "gpt-5-mini", "input": prompt, "max_output_tokens": 500}

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=body)
            if r.status_code >= 400:
                return ""
            data = r.json()
            return data.get("output_text") or ""
    except Exception:
        return ""


# ----------------------------
# Build portfolio rows with tech
# ----------------------------
def build_portfolio_rows(
    agg: List[Dict[str, Any]],
    ticker_map: Dict[int, str],
    tech_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0
    rows: List[Dict[str, Any]] = []

    for a in agg:
        iid = int(a["instrumentID"])
        ticker = ticker_map.get(iid) or str(iid)

        initial = normalize_number(a.get("initialAmountInDollars"))
        unreal = normalize_number(a.get("unrealizedPnL"))

        weight_pct = (initial / total_initial * 100.0) if (total_initial > 0 and initial and initial > 0) else None
        pnl_pct = (unreal / initial * 100.0) if (initial and initial != 0 and unreal is not None) else None

        tech = tech_map.get(iid, {})
        rows.append({
            "ticker": ticker,
            "instrumentID": str(iid),
            "lots": str(a.get("lots", "")),
            "weight_pct": f"{weight_pct:.2f}" if weight_pct is not None else "",
            "pnl_pct": f"{pnl_pct:.2f}" if pnl_pct is not None else "",
            "thesis_score": "",
            "tech_status": tech_tags(tech) if tech else "",
            "tech": tech,
        })

    return rows


async def compute_technicals_for_ids(instrument_ids: List[int]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch candles and compute indicators with a concurrency cap.
    """
    ids = sorted(set(int(x) for x in instrument_ids if x is not None))
    tech_map: Dict[int, Dict[str, Any]] = {}
    debug = {"requested": len(ids), "computed": 0, "failed": 0, "samples": []}

    sem = asyncio.Semaphore(7)  # safe parallelism

    async def one(iid: int):
        async with sem:
            candles = await etoro_get_daily_candles(iid, 250)
            if not candles or len(candles) < 60:
                debug["failed"] += 1
                if len(debug["samples"]) < 10:
                    debug["samples"].append({"instrumentID": iid, "status": "no_candles"})
                return

            closes = [normalize_number(c.get("close")) for c in candles]
            vols = [normalize_number(c.get("volume")) for c in candles]
            closes = [c for c in closes if isinstance(c, (int, float))]
            vols = [v if isinstance(v, (int, float)) else 0.0 for v in vols]

            if len(closes) < 60:
                debug["failed"] += 1
                if len(debug["samples"]) < 10:
                    debug["samples"].append({"instrumentID": iid, "status": "not_enough_closes"})
                return

            last_close = closes[-1]
            last_vol = vols[-1] if vols else 0.0

            rsi14 = rsi(closes, 14)
            macd_line, macd_sig, macd_hist = macd(closes)
            sma20 = sma(closes, 20)
            sma50 = sma(closes, 50)
            sma200 = sma(closes, 200)

            above200 = None
            if sma200 is not None:
                above200 = True if last_close >= sma200 else False

            adv20 = None
            dollar_adv20 = None
            if len(vols) >= 20 and len(closes) >= 20:
                v20 = vols[-20:]
                c20 = closes[-20:]
                adv20 = sum(v20) / 20.0
                dollar_adv20 = sum(v * c for v, c in zip(v20, c20)) / 20.0

            # Liquidity threshold (tune later): $ADV20 < 1,000,000 => illiquid
            illiquid = False
            if isinstance(dollar_adv20, (int, float)) and dollar_adv20 > 0:
                illiquid = dollar_adv20 < 1_000_000

            tech_map[iid] = {
                "close": last_close,
                "volume": last_vol,
                "rsi14": rsi14,
                "macd": macd_line,
                "macd_signal": macd_sig,
                "macd_hist": macd_hist,
                "sma20": sma20,
                "sma50": sma50,
                "sma200": sma200,
                "above_sma200": above200,
                "adv20": adv20,
                "dollar_adv20": dollar_adv20,
                "illiquid": illiquid,
            }

            debug["computed"] += 1
            if len(debug["samples"]) < 5:
                debug["samples"].append({"instrumentID": iid, "rsi14": rsi14, "above_sma200": above200, "illiquid": illiquid})

    await asyncio.gather(*(one(i) for i in ids))
    return tech_map, debug


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
    mapping = state.get("mapping") or {}
    tech_debug = state.get("tech_debug") or {}

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
        <p><b>eToro:</b> Lots = {stats.get("lots_count","")} | Unique instruments = {stats.get("unique_instruments_count","")}</p>
        <p><b>Mapping cache:</b> {mapping.get("cached","")}/{mapping.get("total","")} cached</p>
        <p><b>Technicals:</b> computed {tech_debug.get("computed","")}/{tech_debug.get("requested","")} (failed {tech_debug.get("failed","")})</p>

        <h2>Material Events</h2>
        {bullets(material_events)}

        <h2>Technical Exceptions</h2>
        {bullets(technical_exceptions)}

        <h2>Action Required</h2>
        {bullets(action_required)}

        <p>Run <code>/tasks/daily</code> to refresh data.</p>

        <h2>Daily Brief</h2>
        <pre style="white-space: pre-wrap; background:#fafafa; border:1px solid #eee; padding:12px;">{ai_brief or "No brief yet. Run /tasks/daily."}</pre>

        <h2>Portfolio</h2>
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

        <p>API: <code>/api/portfolio</code> • <code>/api/daily-brief</code> • Debug: <code>/debug/mapping-last</code> • <code>/debug/tech-last</code></p>
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
        f"Discord={'True' if bool(DISCORD_WEBHOOK_URL) else 'False'}, "
        f"eToro keys={'True' if (bool(ETORO_API_KEY) and bool(ETORO_USER_KEY)) else 'False'}."
    )

    # Pull PnL
    payload = await etoro_get_real_pnl()
    raw_positions = extract_positions(payload)
    agg_positions, stats = aggregate_positions_by_instrument(raw_positions)
    material_events.append(
        f"Pulled eToro successfully. Lots: {stats['lots_count']} | Unique instruments: {stats['unique_instruments_count']}"
    )

    instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]

    # ----------------------------
    # (1) Ticker mapping cache
    # ----------------------------
    ticker_cache_raw = state.get("ticker_cache") or {}
    ticker_cache: Dict[int, str] = {}
    for k, v in ticker_cache_raw.items():
        try:
            ticker_cache[int(k)] = str(v)
        except Exception:
            continue

    missing = [iid for iid in instrument_ids if iid not in ticker_cache]
    if missing:
        new_map, map_debug = await map_instrument_ids_to_tickers_search(missing)
        ticker_cache.update(new_map)
        material_events.append(f"Resolved tickers (search): {map_debug['mapped']}/{map_debug['requested']} new")
    else:
        material_events.append("Resolved tickers: 0 new (cache hit)")

    # Save cache back as JSON-safe keys
    state["ticker_cache"] = {str(k): v for k, v in ticker_cache.items()}
    state["mapping_last_debug"] = {"missing": len(missing), "cached_total": len(ticker_cache), "total_today": len(instrument_ids)}
    state["mapping"] = {"cached": len([iid for iid in instrument_ids if iid in ticker_cache]), "total": len(instrument_ids)}

    # ----------------------------
    # (2) Technicals via eToro candles
    # ----------------------------
    tech_map, tech_dbg = await compute_technicals_for_ids(instrument_ids)
    state["tech_debug"] = tech_dbg
    if tech_dbg.get("failed", 0) > 0:
        technical_exceptions.append(f"Candles missing for {tech_dbg.get('failed')} instruments (see /debug/tech-last).")

    # ----------------------------
    # Build final portfolio rows
    # ----------------------------
    portfolio_rows = build_portfolio_rows(agg_positions, ticker_cache, tech_map)

    # ----------------------------
    # (3) Daily brief (deterministic + optional OpenAI)
    # ----------------------------
    det = deterministic_brief(portfolio_rows)
    ai = await generate_openai_brief_safe(portfolio_rows)
    ai_brief = ai if ai else det

    # ----------------------------
    # (4) Discord notify (optional)
    # ----------------------------
    mapped_count = state["mapping"]["cached"]
    tech_done = tech_dbg.get("computed", 0)
    msg = f"✅ Daily done | tickers {mapped_count}/{len(instrument_ids)} | tech {tech_done}/{len(instrument_ids)}"
    await discord_notify(msg)

    # Save state for dashboard/API
    state.update({
        "date": utc_now_iso(),
        "material_events": material_events,
        "technical_exceptions": technical_exceptions,
        "action_required": ["None"],
        "positions": portfolio_rows,
        "stats": stats,
        "ai_brief": ai_brief,
    })
    save_state(state)

    return {
        "status": "ok",
        "lots": stats["lots_count"],
        "unique_instruments": stats["unique_instruments_count"],
        "mapped_symbols": mapped_count,
        "technicals_computed": tech_done,
    }


@app.get("/api/portfolio")
async def api_portfolio():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "stats": state.get("stats") or {},
        "mapping": state.get("mapping") or {},
        "tech_debug": state.get("tech_debug") or {},
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
        "mapping": state.get("mapping") or {},
        "tech_debug": state.get("tech_debug") or {},
    })


@app.get("/debug/mapping-last")
async def debug_mapping_last():
    state = load_state()
    return JSONResponse(state.get("mapping_last_debug") or {"note": "Run /tasks/daily first."})


@app.get("/debug/tech-last")
async def debug_tech_last():
    state = load_state()
    return JSONResponse(state.get("tech_debug") or {"note": "Run /tasks/daily first."})
