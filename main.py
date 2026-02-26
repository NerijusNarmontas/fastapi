import os
import json
import uuid
import asyncio
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ============================================================
# GOALS
# - Stocks-only view (crypto excluded)
# - Crypto tickers like W (Wormhole) removed even if symbol collides with stock
# - Technicals: Stooq primary, eToro fallback with 429 backoff
# - Organized buckets: Energy (Oil/Gas + Midstream), Nuclear/Uranium, Tech, Quantum, etc
# - 7 key fundamentals + Energy thesis-fit score (Energy only)
# - Robust fundamentals: symbol normalization + error handling + debug endpoints
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

# Fundamentals provider: Financial Modeling Prep (optional but recommended)
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()

EXCLUDE_CRYPTO = os.getenv("EXCLUDE_CRYPTO", "true").strip().lower() in ("1", "true", "yes", "y")

CRYPTO_DENY = {
    x.strip().upper()
    for x in os.getenv(
        "CRYPTO_DENY",
        "BTC,ETH,SOL,ADA,XRP,DOT,AVAX,LINK,OP,RUNE,WLD,JTO,ARB,ATOM,NEAR,APT,SUI,CRO,SEI,STRK,PYTH,HBAR,EIGEN,HYPE,W",
    ).split(",")
    if x.strip()
}
CRYPTO_ID_MIN = int(os.getenv("CRYPTO_ID_MIN", "100000"))

STATE_PATH = "/tmp/investing_agent_state.json"

ETORO_REAL_PNL_URL = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
ETORO_SEARCH_URL = "https://public-api.etoro.com/api/v1/market-data/search"
ETORO_CANDLES_URL_TMPL = (
    "https://public-api.etoro.com/api/v1/market-data/instruments/{instrumentId}/history/candles/asc/OneDay/{count}"
)

FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_QUOTE = f"{FMP_BASE}/quote/{{symbol}}"
FMP_KEY_METRICS_TTM = f"{FMP_BASE}/key-metrics-ttm/{{symbol}}"
FMP_RATIOS_TTM = f"{FMP_BASE}/ratios-ttm/{{symbol}}"

http_client: Optional[httpx.AsyncClient] = None
_STATE_LOCK = asyncio.Lock()
_DAILY_LOCK = asyncio.Lock()
_ETORO_FALLBACK_SEM = asyncio.Semaphore(int(os.getenv("ETORO_FALLBACK_CONCURRENCY", "2")))
_FUND_SEM = asyncio.Semaphore(int(os.getenv("FUND_CONCURRENCY", "4")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=80, max_keepalive_connections=30),
        headers={"user-agent": "fastapi-investing-agent/1.2"},
    )
    yield
    await http_client.aclose()
    http_client = None


app = FastAPI(title="My AI Investing Agent", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("UNHANDLED EXCEPTION:\n", tb)
    return PlainTextResponse("Internal server error", status_code=500)


# ============================================================
# STATE
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


async def load_state() -> Dict[str, Any]:
    async with _STATE_LOCK:
        if not os.path.exists(STATE_PATH):
            return {}
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


async def save_state(state: Dict[str, Any]) -> None:
    async with _STATE_LOCK:
        tmp = STATE_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STATE_PATH)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


def require_admin(request: Request) -> None:
    if ADMIN_TOKEN:
        token = request.headers.get("x-admin-token", "")
        if token != ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Missing/invalid x-admin-token")


# ============================================================
# HELPERS
# ============================================================

def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def looks_like_numeric_id(s: str) -> bool:
    return (s or "").strip().isdigit()


def fmt(x: Any, nd: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return ""
    return f"{v:.{nd}f}"


def fmt_pct(frac: Any, nd: int = 2) -> str:
    v = safe_float(frac)
    if v is None:
        return ""
    return f"{v*100:.{nd}f}%"


def fmt_big(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return ""
    a = abs(v)
    if a >= 1e12:
        return f"{v/1e12:.2f}T"
    if a >= 1e9:
        return f"{v/1e9:.2f}B"
    if a >= 1e6:
        return f"{v/1e6:.2f}M"
    if a >= 1e3:
        return f"{v/1e3:.2f}K"
    return f"{v:.0f}"


def is_crypto_ticker(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    if not t:
        return False
    if t in CRYPTO_DENY:
        return True
    if t.endswith("-USD") or t.endswith("USD") or t.endswith("USDT"):
        return True
    return False


def is_crypto_instrument(instrument_id: int, ticker: str) -> bool:
    try:
        if int(instrument_id) >= CRYPTO_ID_MIN:
            return True
    except Exception:
        pass
    return is_crypto_ticker(ticker)


def pick_instrument_id(p: Dict[str, Any]) -> Optional[int]:
    iid = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
    try:
        return int(iid) if iid is not None else None
    except Exception:
        return None


def pick_unrealized_pnl(p: Dict[str, Any]) -> Optional[float]:
    return safe_float(p.get("unrealizedPnL") or p.get("unrealizedPnl") or p.get("unrealized_pnl"))


def pick_initial_usd(p: Dict[str, Any]) -> Optional[float]:
    return safe_float(p.get("initialAmountInDollars") or p.get("initialAmount") or p.get("initial_amount_usd"))


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


# ============================================================
# CATEGORIES
# ============================================================

DEFAULT_CATEGORIES: Dict[str, List[str]] = {
    "ENERGY_OIL_GAS": ["EOG","DVN","SM","NOG","FANG","CIVI","CTRA","MTDR","RRC","GPOR","VIST","OXY","TALO","WDS","VET","EQT"],
    "ENERGY_MIDSTREAM": ["EPD","WMB","ET","WES","TRGP"],
    "NUCLEAR_URANIUM": ["CCJ","LEU","UEC","SMR","OKLO"],
    "TECH_SEMI_DATACENTER": ["NVDA","AMD","AVGO","ASML","TSM","AMAT","LRCX","MU","INTC","ARM","MSFT","AMZN","GOOGL"],
    "QUANTUM": ["IONQ","RGTI","QBTS","ARQQ"],
    "GOLD_PGM_MINERS": ["NEM","AEM","GOLD","KGC","AU","RGLD","SBSW"],
    "OTHER": [],
}


def get_categories(state: Dict[str, Any]) -> Dict[str, List[str]]:
    raw = state.get("categories")
    if isinstance(raw, dict) and raw:
        out: Dict[str, List[str]] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, list):
                out[k.strip().upper()] = [str(x).upper().strip() for x in v if str(x).strip()]
        return out if out else DEFAULT_CATEGORIES
    return DEFAULT_CATEGORIES


def categorize_ticker(ticker: str, categories: Dict[str, List[str]]) -> str:
    t = (ticker or "").upper().strip()
    if not t:
        return "OTHER"
    for cat, syms in categories.items():
        if t in set(syms):
            return cat
    return "OTHER"


def category_label(cat: str) -> str:
    return {
        "ENERGY_OIL_GAS": "Energy — Oil & Gas",
        "ENERGY_MIDSTREAM": "Energy — Midstream",
        "NUCLEAR_URANIUM": "Nuclear & Uranium",
        "TECH_SEMI_DATACENTER": "Tech — Semis & Data Centers",
        "QUANTUM": "Quantum Computing",
        "GOLD_PGM_MINERS": "Gold & PGM Miners",
        "OTHER": "Other / Unclassified",
    }.get(cat, cat)


# ============================================================
# ENERGY THESIS + thesis-fit score
# ============================================================

ENERGY_THESIS_TEXT = (
    "Energy Thesis: post-shale peak dynamics; capital discipline; high FCF; low leverage; "
    "Tier 1 acreage; resilient cash returns; geopolitically relevant assets; gas leverage / LNG optionality; "
    "avoid refiners."
)


def energy_thesis_fit(fund: Dict[str, Any]) -> Tuple[Optional[int], List[str]]:
    if not fund or fund.get("status") != "ok":
        return None, ["NO_FUND"]

    tags: List[str] = []
    score = 50

    fcf_yield = safe_float(fund.get("fcf_yield"))
    nde = safe_float(fund.get("net_debt_ebitda"))
    roic = safe_float(fund.get("roic"))
    div_y = safe_float(fund.get("dividend_yield"))
    pe = safe_float(fund.get("pe"))

    if fcf_yield is not None:
        if fcf_yield >= 0.12:
            score += 18; tags.append("FCF++")
        elif fcf_yield >= 0.08:
            score += 10; tags.append("FCF+")
        elif fcf_yield >= 0.04:
            score += 3
        else:
            score -= 8; tags.append("FCF-")
    else:
        tags.append("FCF?")

    if nde is not None:
        if nde <= 1.0:
            score += 14; tags.append("LEV++")
        elif nde <= 2.0:
            score += 7; tags.append("LEV+")
        elif nde <= 3.0:
            score -= 3
        else:
            score -= 12; tags.append("LEV-")
    else:
        tags.append("LEV?")

    if roic is not None:
        if roic >= 0.15:
            score += 10; tags.append("ROIC+")
        elif roic >= 0.10:
            score += 5
        elif roic < 0.06:
            score -= 6; tags.append("ROIC-")
    else:
        tags.append("ROIC?")

    if div_y is not None:
        if div_y >= 0.04:
            score += 6; tags.append("DIV+")
        elif div_y >= 0.02:
            score += 3
    else:
        tags.append("DIV?")

    if pe is not None:
        if pe <= 10:
            score += 4; tags.append("VAL+")
        elif pe >= 20:
            score -= 5; tags.append("VAL-")

    score = max(0, min(100, int(round(score))))
    return score, tags


# ============================================================
# eToro HTTP
# ============================================================

def etoro_headers() -> Dict[str, str]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        return {}
    return {
        "x-api-key": ETORO_API_KEY,
        "x-user-key": ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
        "user-agent": "fastapi-investing-agent/1.2",
        "accept": "application/json",
    }


async def etoro_get_real_pnl() -> Dict[str, Any]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        raise HTTPException(status_code=400, detail="Missing ETORO_API_KEY or ETORO_USER_KEY")
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not ready")

    r = await http_client.get(ETORO_REAL_PNL_URL, headers=etoro_headers())
    if r.status_code >= 400:
        try:
            payload = r.json()
        except Exception:
            payload = {"text": r.text}
        raise HTTPException(status_code=r.status_code, detail=payload)
    return r.json()


async def etoro_search(params: Dict[str, str]) -> Tuple[int, Any]:
    if http_client is None:
        return 500, {"error": "http_client_not_ready"}

    for attempt in range(1, 6):
        r = await http_client.get(ETORO_SEARCH_URL, headers=etoro_headers(), params=params)
        if r.status_code == 429:
            await asyncio.sleep(min(16, 2 ** (attempt - 1)))
            continue
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        return r.status_code, data

    return 429, {"error": "rate_limited_exhausted", "params": params}


def _extract_ticker_from_search_item(item: Dict[str, Any]) -> str:
    for k in ("internalSymbolFull", "symbolFull", "symbol", "ticker", "displaySymbol", "internalSymbol"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


async def map_instrument_ids_to_tickers_search(instrument_ids: List[int], state: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, Any]]:
    ids = sorted(set(int(x) for x in instrument_ids if x is not None))

    cache_raw = state.get("ticker_cache") or {}
    cache: Dict[int, str] = {}
    for k, v in cache_raw.items():
        try:
            cache[int(k)] = str(v)
        except Exception:
            continue

    missing = [iid for iid in ids if iid not in cache or not cache[iid] or looks_like_numeric_id(cache[iid])]
    debug = {"requested": len(ids), "mapped": 0, "failed": 0, "samples": []}

    sem = asyncio.Semaphore(int(os.getenv("ETORO_SEARCH_CONCURRENCY", "5")))

    async def one(iid: int):
        async with sem:
            patterns = [
                {"instrumentId": str(iid), "pageSize": "5", "pageNumber": "1"},
                {"internalInstrumentId": str(iid), "pageSize": "5", "pageNumber": "1"},
                {"q": str(iid), "pageSize": "5", "pageNumber": "1"},
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
                            debug["mapped"] += 1
                            if len(debug["samples"]) < 12:
                                debug["samples"].append({"instrumentID": iid, "ticker": t, "status": status, "params": params})
                            return
            debug["failed"] += 1
            if len(debug["samples"]) < 12:
                debug["samples"].append({"instrumentID": iid, "status": "no_match"})

    await asyncio.gather(*(one(i) for i in missing))

    state["ticker_cache"] = {str(k): v for k, v in cache.items()}
    state["mapping"] = {"cached": len([iid for iid in ids if iid in cache and not looks_like_numeric_id(cache[iid])]), "total": len(ids)}
    state["mapping_last_debug"] = {"missing": len(missing), "cached_total": len(cache), "total_today": len(ids), "debug": debug}
    await save_state(state)

    return cache, debug


async def etoro_get_daily_candles(instrument_id: int, count: int = 260) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if http_client is None:
        return [], {"provider": "etoro", "instrumentId": instrument_id, "status": "http_client_not_ready"}

    url = ETORO_CANDLES_URL_TMPL.format(instrumentId=instrument_id, count=count)
    dbg = {"provider": "etoro", "instrumentId": instrument_id, "requested": count, "status": "init", "http": None, "error": None}

    for attempt in range(1, 6):
        try:
            r = await http_client.get(url, headers=etoro_headers())
            dbg["http"] = r.status_code

            if r.status_code == 429:
                dbg["status"] = "rate_limited"
                await asyncio.sleep(min(16, 2 ** (attempt - 1)))
                continue

            if r.status_code >= 400:
                dbg["status"] = "http_error"
                dbg["error"] = (r.text or "")[:300]
                return [], dbg

            data = r.json()
        except Exception as e:
            dbg["status"] = "exception"
            dbg["error"] = repr(e)
            await asyncio.sleep(0.5 * attempt)
            continue

        candles: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            if isinstance(data.get("candles"), list) and data["candles"]:
                g0 = data["candles"][0]
                if isinstance(g0, dict) and isinstance(g0.get("candles"), list):
                    candles = [c for c in g0["candles"] if isinstance(c, dict)]
            elif isinstance(data.get("items"), list):
                candles = [c for c in data["items"] if isinstance(c, dict)]

        if not candles:
            dbg["status"] = "empty"
            return [], dbg

        dbg["status"] = "ok"
        dbg["rows"] = len(candles)
        return candles, dbg

    dbg["status"] = "rate_limited_exhausted" if dbg.get("http") == 429 else dbg["status"]
    return [], dbg


# ============================================================
# Stooq candles (primary)
# ============================================================

async def stooq_get_daily_candles(ticker: str, count: int = 260) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if http_client is None:
        return [], {"provider": "stooq", "input": ticker, "status": "http_client_not_ready"}

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

    last_err = None
    text = ""
    for attempt in range(1, 4):
        try:
            r = await http_client.get(url, params=params, headers={"accept": "text/csv"})
            dbg["http"] = r.status_code
            if r.status_code >= 400:
                dbg["status"] = "http_error"
                dbg["error"] = (r.text or "")[:300]
                return [], dbg
            text = r.text or ""
            last_err = None
            break
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.6 * attempt)

    if last_err is not None:
        dbg["status"] = "exception"
        dbg["error"] = repr(last_err)
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


# ============================================================
# Technical indicators
# ============================================================

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

    if t.get("illiquid"):
        tags.append("ILLQ")

    return " | ".join(tags)


async def compute_technicals_for_ids(
    instrument_ids: List[int],
    ticker_map: Dict[int, str],
    candles_count: int = 260,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    ids = sorted(set(int(x) for x in instrument_ids if x is not None))
    tech_map: Dict[int, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {"provider": "stooq+etoro_fallback", "requested": len(ids), "computed": 0, "skipped": 0, "failed": 0, "samples": []}

    sem = asyncio.Semaphore(int(os.getenv("TECH_CONCURRENCY", "6")))

    async def one(iid: int):
        async with sem:
            ticker = (ticker_map.get(iid) or "").strip()
            if not ticker or looks_like_numeric_id(ticker):
                debug["skipped"] += 1
                return
            if EXCLUDE_CRYPTO and is_crypto_instrument(iid, ticker):
                debug["skipped"] += 1
                return

            candles, cdbg = await stooq_get_daily_candles(ticker, count=candles_count)

            if (not candles) or (len(candles) < 80):
                async with _ETORO_FALLBACK_SEM:
                    ecandles, edbg = await etoro_get_daily_candles(iid, count=candles_count)
                if ecandles and len(ecandles) >= 80:
                    candles = ecandles
                    cdbg = {"stooq": cdbg, "fallback": edbg}
                else:
                    debug["failed"] += 1
                    if len(debug["samples"]) < 12:
                        debug["samples"].append({"instrumentID": iid, "ticker": ticker, "status": "no_candles", "dbg": cdbg})
                    return

            closes = [safe_float(c.get("close")) for c in candles]
            vols = [safe_float(c.get("volume")) for c in candles]
            closes = [c for c in closes if isinstance(c, (int, float))]
            vols = [v if isinstance(v, (int, float)) else 0.0 for v in vols]

            if len(closes) < 80:
                debug["failed"] += 1
                return

            last_close = closes[-1]
            rsi14 = rsi(closes, 14)
            macd_line, macd_sig, macd_hist = macd(closes)
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
                "sma20": sma(closes, 20),
                "sma50": sma(closes, 50),
                "sma200": sma200,
                "above_sma200": above200,
                "adv20": adv20,
                "dollar_adv20": dollar_adv20,
                "illiquid": illiquid,
                "candles_debug": cdbg,
            }
            debug["computed"] += 1

    await asyncio.gather(*(one(i) for i in ids))
    return tech_map, debug


# ============================================================
# FUNDAMENTALS (7 key params) — robust
# ============================================================

FUND_KEYS = ["market_cap", "pe", "ev_ebitda", "fcf_yield", "net_debt_ebitda", "roic", "dividend_yield"]


def normalize_symbol_for_fmp(display_symbol: str) -> str:
    """
    FMP generally likes US tickers like 'CCJ', 'ASML', 'BRK-B' etc.
    eToro/search can return things like:
      - 'BRK.B'  -> 'BRK-B'
      - 'RDS.A'  -> 'RDS-A'
      - 'ASML.NV' / 'BOE.ASX' -> 'ASML' / 'BOE' (best-effort)
      - 'CEG.US' -> 'CEG'
    We keep your DISPLAY ticker unchanged in the UI, but query fundamentals on this normalized symbol.
    """
    s = (display_symbol or "").upper().strip()
    if not s:
        return ""
    # remove common exchange suffix after dot
    # keep first segment only (best effort)
    if "." in s:
        s = s.split(".", 1)[0]
    # convert class separator
    s = s.replace("/", "-")
    # BRK.B -> BRK-B
    if "." in s:
        s = s.replace(".", "-")
    return s


async def fmp_get_json(url: str, params: Dict[str, str]) -> Tuple[Optional[Any], Optional[str]]:
    if http_client is None:
        return None, "http_client_not_ready"
    try:
        r = await http_client.get(url, params=params)
        if r.status_code == 429:
            await asyncio.sleep(2)
            r = await http_client.get(url, params=params)
        if r.status_code >= 400:
            return None, f"http_{r.status_code}"
        try:
            data = r.json()
        except Exception:
            return None, "bad_json"
        # FMP errors are often dicts with "Error Message"
        if isinstance(data, dict) and ("Error Message" in data or "error" in data or "message" in data):
            return data, "provider_error"
        return data, None
    except Exception as e:
        return None, repr(e)


async def get_fundamentals_symbol(display_symbol: str, state: Dict[str, Any]) -> Dict[str, Any]:
    sym_disp = (display_symbol or "").upper().strip()
    sym = normalize_symbol_for_fmp(sym_disp)
    if not sym or looks_like_numeric_id(sym):
        return {"status": "NO_FUND"}

    cache = state.get("fund_cache") if isinstance(state.get("fund_cache"), dict) else {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cached = cache.get(sym_disp)
    if isinstance(cached, dict) and cached.get("date") == today:
        return cached

    if not FMP_API_KEY:
        out = {"status": "NO_FUND", "date": today, "symbol": sym, "display_symbol": sym_disp, "reason": "missing_FMP_API_KEY"}
        cache[sym_disp] = out
        state["fund_cache"] = cache
        return out

    async with _FUND_SEM:
        q, qerr = await fmp_get_json(FMP_QUOTE.format(symbol=sym), {"apikey": FMP_API_KEY})
        km, kmerr = await fmp_get_json(FMP_KEY_METRICS_TTM.format(symbol=sym), {"apikey": FMP_API_KEY})
        rt, rterr = await fmp_get_json(FMP_RATIOS_TTM.format(symbol=sym), {"apikey": FMP_API_KEY})

    # if any returned provider_error dict, capture it for debug
    provider_error = None
    if isinstance(q, dict) and qerr == "provider_error":
        provider_error = q
    if isinstance(km, dict) and kmerr == "provider_error":
        provider_error = provider_error or km
    if isinstance(rt, dict) and rterr == "provider_error":
        provider_error = provider_error or rt

    quote0 = q[0] if isinstance(q, list) and q else {}
    km0 = km[0] if isinstance(km, list) and km else {}
    rt0 = rt[0] if isinstance(rt, list) and rt else {}

    market_cap = safe_float(quote0.get("marketCap") or quote0.get("mktCap"))
    pe = safe_float(quote0.get("pe") or quote0.get("peRatio"))
    ev_ebitda = safe_float(km0.get("enterpriseValueOverEBITDATTM") or rt0.get("enterpriseValueMultipleTTM"))
    fcf_yield = safe_float(km0.get("freeCashFlowYieldTTM") or rt0.get("freeCashFlowYieldTTM"))
    net_debt_ebitda = safe_float(km0.get("netDebtToEBITDATTM") or rt0.get("netDebtToEBITDATTM"))
    roic = safe_float(km0.get("roicTTM") or rt0.get("returnOnCapitalEmployedTTM"))
    dividend_yield = safe_float(km0.get("dividendYieldTTM") or rt0.get("dividendYieldTTM") or quote0.get("dividendYield"))

    ok = any(v is not None for v in [market_cap, pe, ev_ebitda, fcf_yield, net_debt_ebitda, roic, dividend_yield])

    out = {
        "status": "ok" if ok else "NO_FUND",
        "date": today,
        "symbol": sym,
        "display_symbol": sym_disp,
        "market_cap": market_cap,
        "pe": pe,
        "ev_ebitda": ev_ebitda,
        "fcf_yield": fcf_yield,
        "net_debt_ebitda": net_debt_ebitda,
        "roic": roic,
        "dividend_yield": dividend_yield,
        "errors": {"quote": qerr, "key_metrics": kmerr, "ratios": rterr},
        "provider_error": provider_error,
    }

    cache[sym_disp] = out
    state["fund_cache"] = cache
    return out


async def compute_fundamentals_for_symbols(symbols: List[str], state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    syms = sorted(set((s or "").upper().strip() for s in symbols if s and not looks_like_numeric_id(s)))
    out: Dict[str, Dict[str, Any]] = {}

    async def one(sym: str):
        out[sym] = await get_fundamentals_symbol(sym, state)

    await asyncio.gather(*(one(s) for s in syms))
    await save_state(state)
    return out


# ============================================================
# PRESENTATION
# ============================================================

def build_portfolio_rows(
    agg: List[Dict[str, Any]],
    ticker_map: Dict[int, str],
    tech_map: Dict[int, Dict[str, Any]],
    fund_map: Dict[str, Dict[str, Any]],
    categories: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0
    rows: List[Dict[str, Any]] = []

    for a in agg:
        iid = int(a["instrumentID"])
        ticker = (ticker_map.get(iid) or str(iid)).strip()
        sym = ticker.upper().strip()

        if EXCLUDE_CRYPTO and is_crypto_instrument(iid, ticker):
            continue

        initial = safe_float(a.get("initialAmountInDollars"))
        unreal = safe_float(a.get("unrealizedPnL"))

        weight_pct = (initial / total_initial * 100.0) if (total_initial > 0 and initial and initial > 0) else None
        pnl_pct = (unreal / initial * 100.0) if (initial and initial != 0 and unreal is not None) else None

        tech = tech_map.get(iid)
        tech_status = (tech_tags(tech) or "OK") if tech else "NO_TECH"

        fund = fund_map.get(sym) or {"status": "NO_FUND"}
        cat = categorize_ticker(sym, categories)

        thesis_score = ""
        thesis_tags: List[str] = []
        if cat in ("ENERGY_OIL_GAS", "ENERGY_MIDSTREAM"):
            s, tags = energy_thesis_fit(fund)
            thesis_tags = tags
            thesis_score = str(s) if s is not None else ""

        rows.append({
            "ticker": ticker,
            "instrumentID": str(iid),
            "category": cat,
            "lots": str(a.get("lots", "")),
            "weight_pct": f"{weight_pct:.2f}" if weight_pct is not None else "",
            "pnl_pct": f"{pnl_pct:.2f}" if pnl_pct is not None else "",
            "thesis_score": thesis_score,
            "thesis_tags": thesis_tags,
            "tech_status": tech_status,
            "tech": tech or {},
            "fundamentals": fund,
        })

    def fnum(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    rows.sort(key=lambda r: (r.get("category", "OTHER"), -fnum(r.get("weight_pct"))))
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

    below200 = [r["ticker"] for r in rows if r.get("tech", {}).get("above_sma200") is False][:12]
    illq = [r["ticker"] for r in rows if r.get("tech", {}).get("illiquid")][:12]

    energy = [r for r in rows if r.get("category") in ("ENERGY_OIL_GAS", "ENERGY_MIDSTREAM")]
    best_thesis = sorted(
        [(r["ticker"], safe_int(r.get("thesis_score"), -1), r.get("thesis_tags", [])) for r in energy if r.get("thesis_score")],
        key=lambda x: x[1],
        reverse=True
    )[:5]

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
    if best_thesis:
        lines.append("Energy thesis-fit leaders: " + ", ".join([f"{t} {s} ({' '.join(tags)})" for t, s, tags in best_thesis]))
    lines.append("")
    if below200:
        lines.append("Below SMA200 (watch trend): " + ", ".join(below200))
    if illq:
        lines.append("Liquidity flags (low $ADV20): " + ", ".join(illq))
    lines.append("")
    lines.append("Energy thesis: " + ENERGY_THESIS_TEXT)
    lines.append("Notes: No buy/sell instructions. Use as a checklist for research.")
    return "\n".join(lines)


async def discord_notify(text: str) -> None:
    if not DISCORD_WEBHOOK_URL or http_client is None:
        return
    try:
        await http_client.post(DISCORD_WEBHOOK_URL, json={"content": text})
    except Exception:
        pass


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    state = await load_state()
    categories = get_categories(state)

    last_update = state.get("date") or utc_now_iso()
    material_events = state.get("material_events") or []
    technical_exceptions = state.get("technical_exceptions") or []
    action_required = state.get("action_required") or ["Run /tasks/daily to refresh data."]

    portfolio = state.get("positions") or []
    stats = state.get("stats") or {}
    brief = state.get("brief") or ""
    mapping = state.get("mapping") or {}
    tech_debug = state.get("tech_debug") or {}
    fund_debug = state.get("fund_debug") or {}

    def bullets(items: List[str]) -> str:
        lis = "".join([f"<li>{x}</li>" for x in items]) if items else "<li>None</li>"
        return f"<ul>{lis}</ul>"

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in portfolio:
        grouped[r.get("category") or "OTHER"].append(r)

    req = safe_int(tech_debug.get("requested", 0))
    skp = safe_int(tech_debug.get("skipped", 0))
    computed = safe_int(tech_debug.get("computed", 0))
    failed = safe_int(tech_debug.get("failed", 0))
    denom = max(0, req - skp)

    fund_provider = fund_debug.get("provider") or ("FMP" if FMP_API_KEY else "None")
    fund_ok = safe_int(fund_debug.get("ok", 0))
    fund_total = safe_int(fund_debug.get("requested", 0))

    def fund_cells(f: Dict[str, Any]) -> str:
        if not isinstance(f, dict) or f.get("status") != "ok":
            # show why (helps immediately)
            reason = (f or {}).get("reason") or ""
            perr = (f or {}).get("provider_error")
            hint = ""
            if reason:
                hint = f" ({reason})"
            elif perr:
                hint = " (provider_error)"
            return f"<td colspan='7'>NO_FUND{hint}</td>"
        return (
            f"<td>{fmt_big(f.get('market_cap'))}</td>"
            f"<td>{fmt(f.get('pe'),2)}</td>"
            f"<td>{fmt(f.get('ev_ebitda'),2)}</td>"
            f"<td>{fmt_pct(f.get('fcf_yield'),2)}</td>"
            f"<td>{fmt(f.get('net_debt_ebitda'),2)}</td>"
            f"<td>{fmt_pct(f.get('roic'),2)}</td>"
            f"<td>{fmt_pct(f.get('dividend_yield'),2)}</td>"
        )

    sections_html = ""
    order = ["ENERGY_OIL_GAS","ENERGY_MIDSTREAM","NUCLEAR_URANIUM","TECH_SEMI_DATACENTER","QUANTUM","GOLD_PGM_MINERS","OTHER"]
    for cat in order:
        rows = grouped.get(cat) or []
        if not rows:
            continue

        body = ""
        for r in rows[:400]:
            thesis = r.get("thesis_score", "")
            ttags = " ".join(r.get("thesis_tags") or [])
            thesis_cell = f"{thesis} {ttags}".strip()
            body += (
                "<tr>"
                f"<td>{r.get('ticker','')}</td>"
                f"<td>{r.get('lots','')}</td>"
                f"<td>{r.get('weight_pct','')}</td>"
                f"<td>{r.get('pnl_pct','')}</td>"
                f"<td>{thesis_cell}</td>"
                f"<td>{r.get('tech_status','')}</td>"
                f"{fund_cells(r.get('fundamentals') or {})}"
                "</tr>"
            )
        if not body:
            body = "<tr><td colspan='13'>No positions in this category.</td></tr>"

        sections_html += f"""
        <h2>{category_label(cat)}</h2>
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Lots</th>
              <th>Weight %</th>
              <th>P&amp;L %</th>
              <th>Thesis (Energy)</th>
              <th>Tech</th>
              <th>MktCap</th>
              <th>P/E</th>
              <th>EV/EBITDA</th>
              <th>FCF Yield</th>
              <th>NetDebt/EBITDA</th>
              <th>ROIC</th>
              <th>Div Yield</th>
            </tr>
          </thead>
          <tbody>
            {body}
          </tbody>
        </table>
        """

    if not sections_html:
        sections_html = "<p>No positions saved yet. Run <code>/tasks/daily</code>.</p>"

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>My AI Investing Agent</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; margin: 24px; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
          th {{ background: #f5f5f5; text-align: left; position: sticky; top: 0; }}
          code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 4px; }}
          .small {{ color: #666; font-size: 12px; }}
        </style>
      </head>
      <body>
        <h1>My AI Investing Agent</h1>
        <p><b>Last update:</b> {last_update}</p>
        <p><b>Universe:</b> {"Stocks only (crypto excluded)" if EXCLUDE_CRYPTO else "Mixed (crypto allowed)"}</p>
        <p><b>Crypto filter:</b> instrumentId &gt;= {CRYPTO_ID_MIN} OR ticker in denylist</p>
        <p><b>eToro:</b> Lots = {stats.get("lots_count","")} | Unique instruments = {stats.get("unique_instruments_count","")}</p>
        <p><b>Mapping cache:</b> {mapping.get("cached","")}/{mapping.get("total","")} cached (real tickers)</p>
        <p><b>Technicals:</b> computed {computed}/{denom} (failed {failed}, skipped {skp})</p>
        <p><b>Fundamentals:</b> provider {fund_provider} | ok {fund_ok}/{fund_total}</p>

        <h2>Energy Thesis</h2>
        <p class="small">{ENERGY_THESIS_TEXT}</p>

        <h2>Material Events</h2>
        {bullets(material_events)}

        <h2>Technical Exceptions</h2>
        {bullets(technical_exceptions)}

        <h2>Action Required</h2>
        {bullets(action_required)}

        <p>Run <code>/tasks/daily</code> to refresh data.</p>

        <h2>Daily Brief</h2>
        <pre style="white-space: pre-wrap; background:#fafafa; border:1px solid #eee; padding:12px;">{brief or "No brief yet. Run /tasks/daily."}</pre>

        {sections_html}

        <p class="small">
          API: <code>/api/portfolio</code> • <code>/api/daily-brief</code> • <code>/api/categories</code> •
          Debug: <code>/debug/mapping-last</code> • <code>/debug/tech-last</code> •
          <code>/debug/fund?symbol=CCJ</code> • <code>/debug/fund-last</code>
        </p>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/tasks/daily")
async def run_daily():
    async with _DAILY_LOCK:
        state = await load_state()
        categories = get_categories(state)

        material_events: List[str] = []
        technical_exceptions: List[str] = []

        material_events.append(
            f"System check: OpenAI={'True' if bool(OPENAI_API_KEY) else 'False'}, "
            f"Discord={'True' if bool(DISCORD_WEBHOOK_URL) else 'False'}, "
            f"eToro keys={'True' if (bool(ETORO_API_KEY) and bool(ETORO_USER_KEY)) else 'False'}, "
            f"Fundamentals={'FMP' if bool(FMP_API_KEY) else 'None'}."
        )

        payload = await etoro_get_real_pnl()
        raw_positions = extract_positions(payload)
        agg_positions, stats = aggregate_positions_by_instrument(raw_positions)

        material_events.append(
            f"Pulled eToro successfully. Lots: {stats['lots_count']} | Unique instruments: {stats['unique_instruments_count']}"
        )

        instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]

        # 1) mapping
        ticker_cache, _ = await map_instrument_ids_to_tickers_search(instrument_ids, state)
        mapping = state.get("mapping") or {"cached": 0, "total": len(instrument_ids)}
        material_events.append(f"Resolved tickers: {mapping.get('cached',0)}/{mapping.get('total',len(instrument_ids))}")

        # 2) technicals
        tech_map, tech_dbg = await compute_technicals_for_ids(instrument_ids, ticker_cache)
        state["tech_debug"] = tech_dbg

        if safe_int(tech_dbg.get("failed", 0)) > 0:
            technical_exceptions.append(f"Candles missing for {tech_dbg.get('failed')} instruments (see /debug/tech-last).")
        if safe_int(tech_dbg.get("skipped", 0)) > 0:
            material_events.append(f"Skipped {tech_dbg.get('skipped')} instruments (no ticker / crypto excluded).")

        # 3) fundamentals
        tickers: List[str] = []
        for iid in instrument_ids:
            t = (ticker_cache.get(iid) or "").strip().upper()
            if not t or looks_like_numeric_id(t):
                continue
            if EXCLUDE_CRYPTO and is_crypto_instrument(iid, t):
                continue
            tickers.append(t)

        fund_map = await compute_fundamentals_for_symbols(tickers, state)

        # debug counters
        uniq = sorted(set(tickers))
        ok = 0
        provider_errors = 0
        missing_key = 0
        samples = []
        for s in uniq:
            f = fund_map.get(s) or {}
            if f.get("status") == "ok":
                ok += 1
            if f.get("reason") == "missing_FMP_API_KEY":
                missing_key += 1
            if f.get("provider_error"):
                provider_errors += 1
            if len(samples) < 6:
                samples.append({"display": s, "symbol": f.get("symbol"), "status": f.get("status"), "errors": f.get("errors"), "provider_error": bool(f.get("provider_error"))})

        state["fund_debug"] = {
            "provider": "FMP" if FMP_API_KEY else "None",
            "requested": len(uniq),
            "ok": ok,
            "missing_key": missing_key,
            "provider_errors": provider_errors,
            "samples": samples,
        }

        # 4) rows
        portfolio_rows = build_portfolio_rows(agg_positions, ticker_cache, tech_map, fund_map, categories)

        # 5) brief
        brief = deterministic_brief(portfolio_rows)

        # 6) notify
        tech_done = safe_int(tech_dbg.get("computed", 0))
        tech_req = safe_int(tech_dbg.get("requested", len(instrument_ids)))
        tech_skip = safe_int(tech_dbg.get("skipped", 0))
        tech_den = max(0, tech_req - tech_skip)
        await discord_notify(
            f"✅ Daily done | tickers {mapping.get('cached',0)}/{len(instrument_ids)} | tech {tech_done}/{tech_den} | fund {ok}/{len(uniq)}"
        )

        state.update({
            "date": utc_now_iso(),
            "material_events": material_events,
            "technical_exceptions": technical_exceptions,
            "action_required": ["None"],
            "positions": portfolio_rows,
            "stats": stats,
            "brief": brief,
            "mapping": mapping,
            "energy_thesis": ENERGY_THESIS_TEXT,
        })
        await save_state(state)

        return {
            "status": "ok",
            "lots": stats["lots_count"],
            "unique_instruments": stats["unique_instruments_count"],
            "mapped_symbols": mapping.get("cached", 0),
            "technicals_computed": tech_done,
            "technicals_requested": tech_den,
            "technicals_skipped": tech_skip,
            "fundamentals_ok": ok,
            "fundamentals_requested": len(uniq),
            "fund_provider": state["fund_debug"]["provider"],
            "fund_provider_errors": provider_errors,
            "fund_missing_key": missing_key,
        }


@app.get("/api/portfolio")
async def api_portfolio():
    state = await load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "stats": state.get("stats") or {},
        "mapping": state.get("mapping") or {},
        "tech_debug": state.get("tech_debug") or {},
        "fund_debug": state.get("fund_debug") or {},
        "energy_thesis": state.get("energy_thesis") or ENERGY_THESIS_TEXT,
        "positions": state.get("positions") or [],
    })


@app.get("/api/daily-brief")
async def api_daily_brief():
    state = await load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "material_events": state.get("material_events") or [],
        "technical_exceptions": state.get("technical_exceptions") or [],
        "action_required": state.get("action_required") or [],
        "brief": state.get("brief") or "",
        "stats": state.get("stats") or {},
        "mapping": state.get("mapping") or {},
        "tech_debug": state.get("tech_debug") or {},
        "fund_debug": state.get("fund_debug") or {},
        "energy_thesis": state.get("energy_thesis") or ENERGY_THESIS_TEXT,
    })


@app.get("/api/categories")
async def api_categories():
    state = await load_state()
    cats = get_categories(state)
    return JSONResponse({
        "categories": cats,
        "labels": {k: category_label(k) for k in cats.keys()},
        "note": "Use POST /admin/set-categories to override.",
    })


@app.get("/debug/mapping-last")
async def debug_mapping_last():
    state = await load_state()
    return JSONResponse(state.get("mapping_last_debug") or {"note": "Run /tasks/daily first."})


@app.get("/debug/tech-last")
async def debug_tech_last():
    state = await load_state()
    return JSONResponse(state.get("tech_debug") or {"note": "Run /tasks/daily first."})


@app.get("/debug/fund-last")
async def debug_fund_last():
    state = await load_state()
    return JSONResponse(state.get("fund_debug") or {"note": "Run /tasks/daily first."})


@app.get("/debug/fund")
async def debug_fund(
    symbol: str = Query(..., description="Try: CCJ, LEU, EQT, WMB, ASML, RGTI"),
):
    state = await load_state()
    f = await get_fundamentals_symbol(symbol, state)
    await save_state(state)
    return JSONResponse({"input": symbol, "normalized": normalize_symbol_for_fmp(symbol), "fundamentals": f})


@app.post("/admin/seed-mapping")
async def admin_seed_mapping(request: Request):
    require_admin(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be JSON object: {instrumentId: ticker}")
    state = await load_state()
    cache_raw = state.get("ticker_cache") or {}
    seeded = 0
    for k, v in body.items():
        try:
            iid = int(k)
        except Exception:
            continue
        sym = str(v).strip()
        if sym:
            cache_raw[str(iid)] = sym
            seeded += 1
    state["ticker_cache"] = cache_raw
    await save_state(state)
    return {"status": "ok", "seeded": seeded, "note": "Run /tasks/daily again."}


@app.post("/admin/set-categories")
async def admin_set_categories(request: Request):
    require_admin(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be JSON object of {category: [tickers]}")
    cleaned: Dict[str, List[str]] = {}
    for k, v in body.items():
        if isinstance(k, str) and isinstance(v, list):
            cleaned[k.strip().upper()] = [str(x).upper().strip() for x in v if str(x).strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="No valid categories provided.")
    state = await load_state()
    state["categories"] = cleaned
    await save_state(state)
    return {"status": "ok", "categories": cleaned, "note": "Refresh /tasks/daily to re-render grouped tables."}
