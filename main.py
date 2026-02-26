import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse

# ============================================================
# GOAL (your requirements)
# - NO crypto (stocks/ETFs only)
# - Technicals computed reliably
# - Always works even if eToro "ticker reverse mapping" changes:
#   -> tries multiple mapping methods + shows debug
# ============================================================

app = FastAPI(title="My AI Investing Agent")

# ----------------------------
# ENV VARS (Railway)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()  # optional
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()                  # optional

# Universe controls
EXCLUDE_CRYPTO = os.getenv("EXCLUDE_CRYPTO", "true").strip().lower() in ("1", "true", "yes", "y")
CRYPTO_DENY = {
    x.strip().upper()
    for x in os.getenv(
        "CRYPTO_DENY",
        "BTC,ETH,SOL,ADA,XRP,DOT,AVAX,LINK,OP,RUNE,WLD,JTO,ARB,ATOM,NEAR,APT,SUI",
    ).split(",")
    if x.strip()
}

# Persist state between requests (note: /tmp survives runtime, but may reset on redeploy)
STATE_PATH = "/tmp/investing_agent_state.json"

# ----------------------------
# eToro endpoints (official)
# ----------------------------
ETORO_REAL_PNL_URL = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
ETORO_SEARCH_URL = "https://public-api.etoro.com/api/v1/market-data/search"

# NOTE:
# Reverse mapping instrumentId -> ticker is not clearly documented as supported via search.
# So we attempt multiple potential endpoints and keep debug for you.
ETORO_INSTRUMENTS_BULK_CANDIDATES = [
    # These are "best guesses" (some environments expose one of these).
    # If none work, we still function: you’ll see numeric IDs (and techs will be skipped for those).
    ("GET", "https://public-api.etoro.com/api/v1/market-data/instruments", "instrumentIds"),
    ("GET", "https://public-api.etoro.com/api/v1/market-data/instruments/details", "instrumentIds"),
    ("GET", "https://public-api.etoro.com/api/v1/market-data/instruments/info", "instrumentIds"),
    ("POST", "https://public-api.etoro.com/api/v1/market-data/instruments", "json_body"),
    ("POST", "https://public-api.etoro.com/api/v1/market-data/instruments/details", "json_body"),
]

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

def pick_instrument_id(p: Dict[str, Any]) -> Optional[int]:
    iid = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
    try:
        return int(iid) if iid is not None else None
    except Exception:
        return None

def pick_unrealized_pnl(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(p.get("unrealizedPnL") or p.get("unrealizedPnl") or p.get("unrealized_pnl") or p.get("pnL") or p.get("pnl"))

def pick_initial_usd(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(p.get("initialAmountInDollars") or p.get("initialAmount") or p.get("initial_amount_usd") or p.get("amount"))

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
    # Some APIs use these keys; we try a handful.
    for k in ("internalSymbolFull", "symbolFull", "symbol", "ticker", "displaySymbol", "instrumentSymbol"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def is_crypto_ticker(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    if not t:
        return False
    if t in CRYPTO_DENY:
        return True
    # common crypto pair shapes
    if t.endswith("-USD") or t.endswith("USD") or t.endswith("USDT"):
        return True
    return False

def looks_like_numeric_id(ticker: str) -> bool:
    t = (ticker or "").strip()
    return t.isdigit()

# ----------------------------
# Reverse mapping: instrumentId -> ticker (robust attempts)
# ----------------------------
async def map_instrument_ids_to_tickers(instrument_ids: List[int], state: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Returns mapping for ids (including existing cache), and debug.
    Strategy:
      1) Load cached mapping from state
      2) Try "bulk instrument details" endpoints (best-case)
      3) Try /market-data/search using multiple param patterns (may or may not work)
      4) Leave unmapped as numeric string (so UI still renders)
    """
    ids = sorted(set(int(x) for x in instrument_ids if x is not None))

    # cache in state
    cache_raw = state.get("ticker_cache") or {}
    cache: Dict[int, str] = {}
    for k, v in cache_raw.items():
        try:
            cache[int(k)] = str(v)
        except Exception:
            continue

    missing = [iid for iid in ids if iid not in cache or not cache[iid] or looks_like_numeric_id(cache[iid])]

    dbg: Dict[str, Any] = {
        "requested": len(ids),
        "cached_before": len(cache),
        "missing_before": len(missing),
        "bulk_attempts": [],
        "search_attempts": [],
        "mapped_new": 0,
        "still_missing": 0,
    }

    # ----------------------------
    # (A) Try bulk instrument info endpoints (best if one exists in your environment)
    # ----------------------------
    if missing:
        chunk_size = 50
        for method, url, mode in ETORO_INSTRUMENTS_BULK_CANDIDATES:
            try:
                newly: Dict[int, str] = {}
                for i in range(0, len(missing), chunk_size):
                    chunk = missing[i:i+chunk_size]
                    async with httpx.AsyncClient(timeout=25) as client:
                        if method == "GET":
                            params = {mode: ",".join(str(x) for x in chunk)}
                            r = await client.get(url, headers=etoro_headers(), params=params)
                        else:
                            # POST JSON body variants
                            body = {"instrumentIds": chunk}
                            r = await client.post(url, headers=etoro_headers(), json=body)
                        status = r.status_code
                        text_preview = (r.text or "")[:300]
                        data = None
                        try:
                            data = r.json()
                        except Exception:
                            data = None

                    extracted = 0
                    if status < 400 and isinstance(data, (dict, list)):
                        # Try to find list of instruments in plausible keys
                        items = None
                        if isinstance(data, dict):
                            for k in ("instruments", "items", "data", "result", "results"):
                                if isinstance(data.get(k), list):
                                    items = data.get(k)
                                    break
                        elif isinstance(data, list):
                            items = data

                        if isinstance(items, list):
                            for it in items:
                                if not isinstance(it, dict):
                                    continue
                                # Try common id keys
                                iid = it.get("instrumentId") or it.get("instrumentID") or it.get("id") or it.get("instrument_id")
                                sym = it.get("symbolFull") or it.get("internalSymbolFull") or it.get("symbol") or it.get("ticker") or it.get("displaySymbol")
                                if iid is None or not sym:
                                    continue
                                try:
                                    iid_int = int(iid)
                                except Exception:
                                    continue
                                sym_str = str(sym).strip()
                                if sym_str:
                                    newly[iid_int] = sym_str
                                    extracted += 1

                    dbg["bulk_attempts"].append({
                        "method": method,
                        "url": url,
                        "mode": mode,
                        "status": status,
                        "extracted": extracted,
                        "preview": text_preview if status >= 400 else None,
                    })

                # apply if any extracted
                if newly:
                    for k, v in newly.items():
                        if k in missing and (k not in cache or looks_like_numeric_id(cache.get(k, ""))):
                            cache[k] = v
                    dbg["mapped_new"] += len(newly)

                # refresh missing list after each candidate
                missing = [iid for iid in ids if iid not in cache or not cache[iid] or looks_like_numeric_id(cache[iid])]
                if not missing:
                    break
            except Exception as e:
                dbg["bulk_attempts"].append({
                    "method": method,
                    "url": url,
                    "mode": mode,
                    "status": "exception",
                    "error": repr(e),
                })

    # ----------------------------
    # (B) Try search endpoint in different ways (not guaranteed to support reverse lookup)
    # ----------------------------
    if missing:
        sem = asyncio.Semaphore(10)

        async def one(iid: int):
            async with sem:
                patterns = [
                    {"instrumentId": str(iid), "pageSize": "5", "pageNumber": "1"},
                    {"instrumentID": str(iid), "pageSize": "5", "pageNumber": "1"},
                    {"internalInstrumentId": str(iid), "pageSize": "5", "pageNumber": "1"},
                    {"q": str(iid), "pageSize": "5", "pageNumber": "1"},
                    {"query": str(iid), "pageSize": "5", "pageNumber": "1"},
                    {"term": str(iid), "pageSize": "5", "pageNumber": "1"},
                ]
                for params in patterns:
                    status, data = await etoro_search(params)
                    items = data.get("items") if isinstance(data, dict) else None
                    if isinstance(items, list) and items:
                        for it in items:
                            if not isinstance(it, dict):
                                continue
                            t = _extract_ticker_from_search_item(it)
                            if t and not looks_like_numeric_id(t):
                                cache[iid] = t
                                dbg["search_attempts"].append({"instrumentID": iid, "status": status, "used_params": params, "hit": True})
                                return
                    dbg["search_attempts"].append({"instrumentID": iid, "status": status, "used_params": params, "hit": False})

        await asyncio.gather(*(one(i) for i in missing))

    still_missing = [iid for iid in ids if iid not in cache or not cache[iid] or looks_like_numeric_id(cache[iid])]
    dbg["still_missing"] = len(still_missing)

    # Save cache back to state
    state["ticker_cache"] = {str(k): v for k, v in cache.items()}
    state["mapping_last_debug"] = dbg
    state["mapping"] = {"cached": len([iid for iid in ids if iid in cache and not looks_like_numeric_id(cache[iid])]), "total": len(ids)}
    save_state(state)

    return cache, dbg

# ----------------------------
# Candles (stocks only) via Stooq (no keys)
# ----------------------------
async def stooq_get_daily_candles(ticker: str, count: int = 260) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stooq CSV daily candles.
    For US stocks, stooq usually expects "aapl.us" etc.
    """
    t = (ticker or "").strip()
    dbg: Dict[str, Any] = {"provider": "stooq", "input": ticker, "requested": count, "status": "init", "http": None, "error": None}

    if not t:
        dbg["status"] = "empty_ticker"
        return [], dbg

    if looks_like_numeric_id(t):
        dbg["status"] = "numeric_not_ticker"
        return [], dbg

    s = t.lower()
    if "." not in s:
        s = f"{s}.us"

    url = "https://stooq.com/q/d/l/"
    params = {"s": s, "i": "d"}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params, headers={"accept": "text/csv"})
            dbg["http"] = r.status_code
            if r.status_code >= 400:
                dbg["status"] = "http_error"
                dbg["error"] = (r.text or "")[:300]
                return [], dbg
            text = r.text or ""
    except Exception as e:
        dbg["status"] = "exception"
        dbg["error"] = repr(e)
        return [], dbg

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2 or "date" not in lines[0].lower():
        dbg["status"] = "bad_csv"
        dbg["error"] = (text or "")[:300]
        return [], dbg

    candles: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 6:
            continue
        d, o, h, l, c, v = parts[:6]
        try:
            candles.append({
                "fromDate": d,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v) if v not in ("", "-") else 0.0,
            })
        except Exception:
            continue

    if not candles:
        dbg["status"] = "no_rows"
        return [], dbg

    candles = candles[-max(1, int(count)):]
    dbg["status"] = "ok"
    dbg["rows"] = len(candles)
    return candles, dbg

# ----------------------------
# Technicals (simple, stable)
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

async def compute_technicals_for_ids(
    instrument_ids: List[int],
    ticker_map: Dict[int, str],
    candles_count: int = 260,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Stocks-only technicals:
      - if ticker is missing/unknown/numeric => skip
      - if ticker looks crypto and EXCLUDE_CRYPTO => skip
      - else candles via Stooq => compute SMA/RSI/MACD + liquidity flag
    """
    ids = sorted(set(int(x) for x in instrument_ids if x is not None))
    tech_map: Dict[int, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {
        "provider": "stooq",
        "requested": len(ids),
        "computed": 0,
        "skipped": 0,
        "failed": 0,
        "samples": [],
    }

    sem = asyncio.Semaphore(7)

    async def one(iid: int):
        async with sem:
            ticker = (ticker_map.get(iid) or "").strip()

            if not ticker or looks_like_numeric_id(ticker):
                debug["skipped"] += 1
                if len(debug["samples"]) < 12:
                    debug["samples"].append({"instrumentID": iid, "ticker": ticker, "status": "skipped_no_ticker"})
                return

            if EXCLUDE_CRYPTO and is_crypto_ticker(ticker):
                debug["skipped"] += 1
                if len(debug["samples"]) < 12:
                    debug["samples"].append({"instrumentID": iid, "ticker": ticker, "status": "skipped_crypto"})
                return

            candles, cdbg = await stooq_get_daily_candles(ticker, count=candles_count)
            if not candles or len(candles) < 80:
                debug["failed"] += 1
                if len(debug["samples"]) < 12:
                    debug["samples"].append({"instrumentID": iid, "ticker": ticker, "status": "no_candles", "cdbg": cdbg})
                return

            closes = [normalize_number(c.get("close")) for c in candles]
            vols = [normalize_number(c.get("volume")) for c in candles]
            closes = [c for c in closes if isinstance(c, (int, float))]
            vols = [v if isinstance(v, (int, float)) else 0.0 for v in vols]

            if len(closes) < 80:
                debug["failed"] += 1
                if len(debug["samples"]) < 12:
                    debug["samples"].append({"instrumentID": iid, "ticker": ticker, "status": "not_enough_closes"})
                return

            last_close = closes[-1]
            rsi14 = rsi(closes, 14)
            macd_line, macd_sig, macd_hist = macd(closes)
            sma20 = sma(closes, 20)
            sma50 = sma(closes, 50)
            sma200 = sma(closes, 200)

            above200 = None
            if sma200 is not None:
                above200 = last_close >= sma200

            adv20 = None
            dollar_adv20 = None
            if len(vols) >= 20 and len(closes) >= 20:
                v20 = vols[-20:]
                c20 = closes[-20:]
                adv20 = sum(v20) / 20.0
                dollar_adv20 = sum(v * c for v, c in zip(v20, c20)) / 20.0

            illiquid = False
            if isinstance(dollar_adv20, (int, float)) and dollar_adv20 > 0:
                illiquid = dollar_adv20 < 1_000_000

            tech_map[iid] = {
                "close": last_close,
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
                "candles_provider": "stooq",
            }

            debug["computed"] += 1
            if len(debug["samples"]) < 6:
                debug["samples"].append({"instrumentID": iid, "ticker": ticker, "rsi14": rsi14, "above_sma200": above200, "illiquid": illiquid})

    await asyncio.gather(*(one(i) for i in ids))
    return tech_map, debug

# ----------------------------
# Portfolio rows
# ----------------------------
def build_portfolio_rows(
    agg: List[Dict[str, Any]],
    ticker_map: Dict[int, str],
    tech_map: Dict[int, Dict[str, Any]],
    exclude_crypto: bool,
) -> List[Dict[str, Any]]:
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0
    rows: List[Dict[str, Any]] = []

    for a in agg:
        iid = int(a["instrumentID"])
        ticker = (ticker_map.get(iid) or str(iid)).strip()

        if exclude_crypto and is_crypto_ticker(ticker):
            continue

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

def deterministic_brief(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No portfolio rows."

    def fnum(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    winners = sorted(rows, key=lambda r: fnum(r.get("pnl_pct")), reverse=True)[:5]
    losers = sorted(rows, key=lambda r: fnum(r.get("pnl_pct")))[:5]

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

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    state = load_state()

    last_update = state.get("date") or utc_now_iso()
    material_events = state.get("material_events") or []
    technical_exceptions = state.get("technical_exceptions") or []
    action_required = state.get("action_required") or ["Run /tasks/daily to refresh data."]

    portfolio = state.get("positions") or []
    stats = state.get("stats") or {}
    brief = state.get("brief") or ""
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
        <p><b>Universe:</b> {"Stocks only (crypto excluded)" if EXCLUDE_CRYPTO else "Mixed (crypto allowed)"}</p>
        <p><b>eToro:</b> Lots = {stats.get("lots_count","")} | Unique instruments = {stats.get("unique_instruments_count","")}</p>
        <p><b>Mapping cache:</b> {mapping.get("cached","")}/{mapping.get("total","")} cached (real tickers)</p>
        <p><b>Technicals:</b> computed {tech_debug.get("computed","")}/{max(0, int(tech_debug.get("requested",0) or 0) - int(tech_debug.get("skipped",0) or 0))} (failed {tech_debug.get("failed","")}, skipped {tech_debug.get("skipped",0)})</p>

        <h2>Material Events</h2>
        {bullets(material_events)}

        <h2>Technical Exceptions</h2>
        {bullets(technical_exceptions)}

        <h2>Action Required</h2>
        {bullets(action_required)}

        <p>Run <code>/tasks/daily</code> to refresh data.</p>

        <h2>Daily Brief</h2>
        <pre style="white-space: pre-wrap; background:#fafafa; border:1px solid #eee; padding:12px;">{brief or "No brief yet. Run /tasks/daily."}</pre>

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

        <p>API: <code>/api/portfolio</code> • <code>/api/daily-brief</code> • Debug: <code>/debug/mapping-last</code> • <code>/debug/tech-last</code> • <code>/debug/candles?ticker=CCJ</code></p>
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

    payload = await etoro_get_real_pnl()
    raw_positions = extract_positions(payload)
    agg_positions, stats = aggregate_positions_by_instrument(raw_positions)

    material_events.append(
        f"Pulled eToro successfully. Lots: {stats['lots_count']} | Unique instruments: {stats['unique_instruments_count']}"
    )

    instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]

    # 1) Map instrument IDs -> tickers (robust)
    ticker_cache, map_dbg = await map_instrument_ids_to_tickers(instrument_ids, state)

    mapped_count = state.get("mapping", {}).get("cached", 0)
    material_events.append(f"Resolved tickers: {mapped_count}/{len(instrument_ids)} (see /debug/mapping-last)")

    # 2) Technicals (stocks only)
    tech_map, tech_dbg = await compute_technicals_for_ids(instrument_ids, ticker_cache)
    state["tech_debug"] = tech_dbg

    if tech_dbg.get("failed", 0) > 0:
        technical_exceptions.append(f"Candles missing for {tech_dbg.get('failed')} instruments (see /debug/tech-last).")
    if tech_dbg.get("skipped", 0) > 0:
        material_events.append(f"Skipped {tech_dbg.get('skipped')} instruments (no ticker / crypto excluded).")

    # 3) Build portfolio rows (apply crypto exclusion here too)
    portfolio_rows = build_portfolio_rows(agg_positions, ticker_cache, tech_map, exclude_crypto=EXCLUDE_CRYPTO)

    # 4) Brief
    brief = deterministic_brief(portfolio_rows)

    # 5) Discord (optional)
    tech_done = tech_dbg.get("computed", 0)
    tech_req = tech_dbg.get("requested", len(instrument_ids))
    tech_skip = tech_dbg.get("skipped", 0)
    tech_den = max(0, int(tech_req) - int(tech_skip))
    await discord_notify(f"✅ Daily done | tickers {mapped_count}/{len(instrument_ids)} | tech {tech_done}/{tech_den}")

    # Save state for dashboard/API
    state.update({
        "date": utc_now_iso(),
        "material_events": material_events,
        "technical_exceptions": technical_exceptions,
        "action_required": ["None"],
        "positions": portfolio_rows,
        "stats": stats,
        "brief": brief,
    })
    save_state(state)

    return {
        "status": "ok",
        "lots": stats["lots_count"],
        "unique_instruments": stats["unique_instruments_count"],
        "mapped_symbols": mapped_count,
        "technicals_computed": tech_done,
        "technicals_requested": tech_den,
        "technicals_skipped": tech_skip,
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
        "brief": state.get("brief") or "",
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

@app.get("/debug/candles")
async def debug_candles(ticker: str = Query(..., description="Try: CCJ, ASML, CEG, ALB")):
    if EXCLUDE_CRYPTO and is_crypto_ticker(ticker):
        return JSONResponse({"ticker": ticker, "status": "skipped_crypto"})
    candles, dbg = await stooq_get_daily_candles(ticker, count=180)
    sample = candles[-3:] if candles else []
    return JSONResponse({"ticker": ticker, "debug": dbg, "rows": len(candles), "sample": sample})

@app.get("/debug/search-test")
async def debug_search_test(instrument_id: int = Query(..., description="Paste a numeric instrumentID from your table (e.g. 9031)")):
    # This tells us if /market-data/search can reverse-map instrument IDs in your environment.
    patterns = [
        {"instrumentId": str(instrument_id), "pageSize": "5", "pageNumber": "1"},
        {"instrumentID": str(instrument_id), "pageSize": "5", "pageNumber": "1"},
        {"internalInstrumentId": str(instrument_id), "pageSize": "5", "pageNumber": "1"},
        {"q": str(instrument_id), "pageSize": "5", "pageNumber": "1"},
        {"query": str(instrument_id), "pageSize": "5", "pageNumber": "1"},
        {"term": str(instrument_id), "pageSize": "5", "pageNumber": "1"},
    ]
    results = []
    for params in patterns:
        status, data = await etoro_search(params)
        results.append({
            "params": params,
            "status": status,
            "preview": str(data)[:600],
        })
    return JSONResponse({
        "instrumentId": instrument_id,
        "keys_present": {"ETORO_API_KEY": bool(ETORO_API_KEY), "ETORO_USER_KEY": bool(ETORO_USER_KEY)},
        "results": results,
    })

@app.post("/admin/seed-mapping")
async def admin_seed_mapping(request: Request):
    """
    If eToro doesn't expose reverse mapping in your environment, this endpoint lets you seed it once.
    POST JSON like: {"9031":"CCJ","9979":"ASML", ...}
    (Needs x-admin-token header if you set ADMIN_TOKEN)
    """
    require_admin(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be JSON object: {instrumentId: ticker}")

    state = load_state()
    cache_raw = state.get("ticker_cache") or {}
    for k, v in body.items():
        try:
            iid = int(k)
        except Exception:
            continue
        sym = str(v).strip()
        if sym:
            cache_raw[str(iid)] = sym

    state["ticker_cache"] = cache_raw
    save_state(state)
    return {"status": "ok", "seeded": len(body), "note": "Run /tasks/daily again."}
