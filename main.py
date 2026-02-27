import os
import json
import uuid
import asyncio
import traceback
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from xml.etree import ElementTree as ET

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ============================================================
# WHAT THIS VERSION FIXES / ADDS (your request)
# ✅ Keep P/L % (you already have it; kept)
# ✅ Remove Market Cap column (removed)
# ✅ Remove winners/losers section from brief (removed)
# ✅ Add FREE news section (Google News RSS per ticker, cached daily)
# ✅ Add SEC filings section (EDGAR submissions JSON per ticker->CIK, cached daily)
# ✅ Add short resumes (summaries) for filings using OpenAI if key exists,
#    otherwise a simple fallback (no full-text reading required by you)
# ✅ Still: stocks-only view, crypto excluded, tech indicators, categories, energy thesis fit
# ============================================================

# ----------------------------
# ENV VARS
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()  # safe default; change if you want

ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

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

# SEC requires a User-Agent with contact info. Provide yours via env if you want.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "fastapi-investing-agent/1.0 (contact: admin@local)").strip()

STATE_PATH = "/tmp/investing_agent_state.json"

# ----------------------------
# eToro endpoints
# ----------------------------
ETORO_REAL_PNL_URL = "https://public-api.etoro.com/api/v1/trading/info/real/pnl"
ETORO_SEARCH_URL = "https://public-api.etoro.com/api/v1/market-data/search"
ETORO_CANDLES_URL_TMPL = (
    "https://public-api.etoro.com/api/v1/market-data/instruments/{instrumentId}/history/candles/asc/OneDay/{count}"
)

# ----------------------------
# FREE NEWS (RSS)
# ----------------------------
# Google News RSS is free and gives you source + link + title + time.
# Query form: https://news.google.com/rss/search?q=CCJ%20stock&hl=en-US&gl=US&ceid=US:en
GOOGLE_NEWS_RSS_TMPL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

# ----------------------------
# SEC / EDGAR (free)
# ----------------------------
# Ticker->CIK map (free file)
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# Company submissions JSON (free)
SEC_SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik10}.json"
# Filing index link (human-friendly)
SEC_ARCHIVES_INDEX_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_nodash}/{primary_doc}"

# ============================================================
# GLOBALS: shared HTTP + locks
# ============================================================
http_client: Optional[httpx.AsyncClient] = None

_STATE_LOCK = asyncio.Lock()
_DAILY_LOCK = asyncio.Lock()

_ETORO_FALLBACK_SEM = asyncio.Semaphore(int(os.getenv("ETORO_FALLBACK_CONCURRENCY", "2")))
_NEWS_SEM = asyncio.Semaphore(int(os.getenv("NEWS_CONCURRENCY", "6")))
_SEC_SEM = asyncio.Semaphore(int(os.getenv("SEC_CONCURRENCY", "3")))
_SUMMARY_SEM = asyncio.Semaphore(int(os.getenv("SUMMARY_CONCURRENCY", "2")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=90, max_keepalive_connections=30),
        headers={"user-agent": "fastapi-investing-agent/1.3"},
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


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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
    "ENERGY_OIL_GAS": ["EOG", "DVN", "SM", "NOG", "FANG", "CIVI", "CTRA", "MTDR", "RRC", "GPOR", "VIST", "OXY", "TALO", "WDS", "VET", "EQT"],
    "ENERGY_MIDSTREAM": ["EPD", "WMB", "ET", "WES", "TRGP"],
    "NUCLEAR_URANIUM": ["CCJ", "LEU", "UEC", "SMR", "OKLO"],
    "TECH_SEMI_DATACENTER": ["NVDA", "AMD", "AVGO", "ASML", "TSM", "AMAT", "LRCX", "MU", "INTC", "ARM", "MSFT", "AMZN", "GOOGL"],
    "QUANTUM": ["IONQ", "RGTI", "QBTS", "ARQQ"],
    "GOLD_PGM_MINERS": ["NEM", "AEM", "GOLD", "KGC", "AU", "RGLD", "SBSW"],
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
# ENERGY THESIS + thesis-fit (uses fundamentals if available; if not, shows NO_FUND)
# (kept minimal; you can later plug in a fundamentals provider again)
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
# eTORO HTTP
# ============================================================

def etoro_headers() -> Dict[str, str]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        return {}
    return {
        "x-api-key": ETORO_API_KEY,
        "x-user-key": ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
        "user-agent": "fastapi-investing-agent/1.3",
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
# STOOQ candles (primary)
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
# TECHNICAL INDICATORS
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
# NEWS (FREE) — Google News RSS
# ============================================================

def google_news_query_for_ticker(ticker: str) -> str:
    # Keep query simple; Google returns source fields
    # You can tune by adding "site:" but keep it broad for now.
    return f"{ticker} stock"


def parse_google_news_rss(xml_text: str, max_items: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    # RSS: channel/item
    channel = root.find("channel")
    if channel is None:
        return out

    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        out.append({"title": title, "link": link, "pubDate": pub, "source": source})
    return out


async def fetch_google_news_for_ticker(ticker: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if http_client is None:
        return [], {"status": "http_client_not_ready"}

    q = google_news_query_for_ticker(ticker)
    url = GOOGLE_NEWS_RSS_TMPL.format(q=httpx.QueryParams({"q": q})["q"])  # simple escape
    # Note: above trick only encodes; actual params are embedded already

    # Better: build ourselves:
    url = GOOGLE_NEWS_RSS_TMPL.format(q=httpx.URL("").copy_with(params={"q": q}).params["q"])

    dbg = {"provider": "google_news_rss", "ticker": ticker, "status": "init", "http": None, "error": None}

    async with _NEWS_SEM:
        try:
            r = await http_client.get(url, headers={"accept": "application/rss+xml, text/xml"})
            dbg["http"] = r.status_code
            if r.status_code >= 400:
                dbg["status"] = "http_error"
                dbg["error"] = (r.text or "")[:200]
                return [], dbg
            items = parse_google_news_rss(r.text or "", max_items=6)
            dbg["status"] = "ok"
            dbg["items"] = len(items)
            return items, dbg
        except Exception as e:
            dbg["status"] = "exception"
            dbg["error"] = repr(e)
            return [], dbg


async def get_news_for_tickers(tickers: List[str], state: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Cached daily in state['news_cache'][TICKER] = {date, items}
    """
    today = utc_today()
    cache = state.get("news_cache") if isinstance(state.get("news_cache"), dict) else {}
    out: Dict[str, List[Dict[str, Any]]] = {}

    uniq = sorted(set(t.upper().strip() for t in tickers if t and not looks_like_numeric_id(t)))
    dbg = {"provider": "google_news_rss", "requested": len(uniq), "fetched": 0, "cached": 0, "errors": 0, "samples": []}

    async def one(t: str):
        nonlocal cache
        cached = cache.get(t)
        if isinstance(cached, dict) and cached.get("date") == today and isinstance(cached.get("items"), list):
            out[t] = cached["items"]
            dbg["cached"] += 1
            return

        items, idbg = await fetch_google_news_for_ticker(t)
        if items:
            cache[t] = {"date": today, "items": items}
            out[t] = items
            dbg["fetched"] += 1
        else:
            out[t] = []
            dbg["errors"] += 1
        if len(dbg["samples"]) < 6:
            dbg["samples"].append({"ticker": t, "debug": idbg})

    await asyncio.gather(*(one(t) for t in uniq))

    state["news_cache"] = cache
    state["news_debug"] = dbg
    await save_state(state)
    return out, dbg


# ============================================================
# SEC FILINGS (FREE) — ticker->CIK map + submissions JSON
# ============================================================

def sec_headers() -> Dict[str, str]:
    return {
        "user-agent": SEC_USER_AGENT,
        "accept-encoding": "gzip, deflate, br",
        "accept": "application/json,text/plain,*/*",
        "connection": "keep-alive",
    }


async def fetch_sec_ticker_map(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns {TICKER: CIK10}
    Cached daily in state['sec_ticker_map']
    """
    today = utc_today()
    cached = state.get("sec_ticker_map")
    if isinstance(cached, dict) and cached.get("date") == today and isinstance(cached.get("map"), dict):
        return cached["map"]

    if http_client is None:
        return {}

    async with _SEC_SEM:
        try:
            r = await http_client.get(SEC_TICKER_MAP_URL, headers=sec_headers())
            if r.status_code >= 400:
                state["sec_ticker_map"] = {"date": today, "map": {}, "error": f"http_{r.status_code}"}
                await save_state(state)
                return {}
            data = r.json()
        except Exception as e:
            state["sec_ticker_map"] = {"date": today, "map": {}, "error": repr(e)}
            await save_state(state)
            return {}

    # company_tickers.json is usually dict keyed by index with fields: cik_str, ticker, title
    out: Dict[str, str] = {}
    if isinstance(data, dict):
        for _, row in data.items():
            if not isinstance(row, dict):
                continue
            t = str(row.get("ticker") or "").upper().strip()
            cik = row.get("cik_str")
            if not t or cik is None:
                continue
            try:
                cik10 = f"{int(cik):010d}"
            except Exception:
                continue
            out[t] = cik10

    state["sec_ticker_map"] = {"date": today, "map": out, "error": None}
    await save_state(state)
    return out


def normalize_ticker_for_sec(ticker: str) -> str:
    """
    eToro tickers can contain suffixes/classes; SEC tickers are usually base.
    - BRK.B -> BRK.B (SEC uses BRK.B sometimes), but mapping file often uses BRK.B
    - If suffix like ASML.NV -> ASML
    """
    t = (ticker or "").upper().strip()
    if not t:
        return ""
    if "." in t and len(t.split(".", 1)[1]) in (2, 3, 4):  # heuristic suffix
        # keep BRK.B / RDS.A as-is (class shares), but drop exchange suffixes like .NV .ASX .L
        base, suf = t.split(".", 1)
        if suf in ("A", "B", "C", "D"):
            return t
        return base
    return t


async def fetch_sec_submissions(cik10: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if http_client is None:
        return None, {"status": "http_client_not_ready"}

    url = SEC_SUBMISSIONS_URL_TMPL.format(cik10=cik10)
    dbg = {"provider": "sec_submissions", "cik10": cik10, "status": "init", "http": None, "error": None}

    async with _SEC_SEM:
        try:
            r = await http_client.get(url, headers=sec_headers())
            dbg["http"] = r.status_code
            if r.status_code >= 400:
                dbg["status"] = "http_error"
                dbg["error"] = (r.text or "")[:200]
                return None, dbg
            dbg["status"] = "ok"
            return r.json(), dbg
        except Exception as e:
            dbg["status"] = "exception"
            dbg["error"] = repr(e)
            return None, dbg


def build_filing_link(cik10: str, accession_no: str, primary_doc: str) -> str:
    cik_int = str(int(cik10))  # remove leading zeros
    acc_no_nodash = accession_no.replace("-", "")
    return SEC_ARCHIVES_INDEX_TMPL.format(cik_int=cik_int, acc_no_nodash=acc_no_nodash, primary_doc=primary_doc)


def extract_recent_filings(submissions: Dict[str, Any], max_items: int = 6) -> List[Dict[str, Any]]:
    """
    Uses submissions['filings']['recent'] arrays. Returns list of filing dicts:
    {form, filingDate, accessionNumber, primaryDocument, primaryDocDescription, link}
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(submissions, dict):
        return out

    recent = (submissions.get("filings") or {}).get("recent") if isinstance(submissions.get("filings"), dict) else None
    if not isinstance(recent, dict):
        return out

    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accs = recent.get("accessionNumber") or []
    prim_docs = recent.get("primaryDocument") or []
    prim_desc = recent.get("primaryDocDescription") or []

    n = min(len(forms), len(dates), len(accs), len(prim_docs))
    for i in range(min(n, max_items)):
        out.append({
            "form": str(forms[i]),
            "filingDate": str(dates[i]),
            "accessionNumber": str(accs[i]),
            "primaryDocument": str(prim_docs[i]),
            "primaryDocDescription": str(prim_desc[i]) if i < len(prim_desc) else "",
        })
    return out


def filing_priority(form: str) -> int:
    # Higher priority (lower number) = more important for your use
    f = (form or "").upper().strip()
    if f == "8-K":
        return 1
    if f in ("10-Q", "10-K"):
        return 2
    if f in ("DEF 14A", "20-F", "6-K"):
        return 3
    if f.startswith("S-"):
        return 4
    if f in ("4", "13D", "13G"):
        return 5
    return 9


def simple_filing_resume(form: str, desc: str) -> str:
    """
    No-OpenAI fallback: short, readable, based on form type + description.
    """
    f = (form or "").upper().strip()
    d = (desc or "").strip()
    if f == "8-K":
        base = "8-K: material update (earnings release, guidance change, deal, financing, leadership, etc.)."
    elif f == "10-Q":
        base = "10-Q: quarterly financial update (margins, cash flow, guidance language changes)."
    elif f == "10-K":
        base = "10-K: annual report (full business + risks + capital structure)."
    elif f == "DEF 14A":
        base = "Proxy (DEF 14A): executive pay + incentives + governance signals."
    elif f == "4":
        base = "Form 4: insider transaction (signal, not a standalone reason)."
    elif f.startswith("S-") or f.startswith("424"):
        base = "Offering-related filing: possible dilution / financing / shelf registration."
    else:
        base = f"{f}: filing update."
    if d:
        return f"{base} Doc: {d}"
    return base


def strip_html_to_text(html: str, max_chars: int = 8000) -> str:
    # Very lightweight: remove scripts/styles + tags + collapse whitespace
    if not html:
        return ""
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html[:max_chars]


async def fetch_filing_primary_text(cik10: str, accession_no: str, primary_doc: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch the primary document HTML and return stripped text.
    """
    if http_client is None:
        return "", {"status": "http_client_not_ready"}

    url = build_filing_link(cik10, accession_no, primary_doc)
    dbg = {"provider": "sec_primary_doc", "url": url, "status": "init", "http": None, "error": None}

    async with _SEC_SEM:
        try:
            r = await http_client.get(url, headers={**sec_headers(), "accept": "text/html,*/*"})
            dbg["http"] = r.status_code
            if r.status_code >= 400:
                dbg["status"] = "http_error"
                dbg["error"] = (r.text or "")[:200]
                return "", dbg
            txt = strip_html_to_text(r.text or "", max_chars=9000)
            dbg["status"] = "ok"
            dbg["chars"] = len(txt)
            return txt, dbg
        except Exception as e:
            dbg["status"] = "exception"
            dbg["error"] = repr(e)
            return "", dbg


async def openai_resume(text: str, context: str) -> Tuple[str, Optional[str]]:
    """
    Summarize with OpenAI if available. Returns (summary, error).
    Kept short & decision-oriented.
    """
    if not OPENAI_API_KEY:
        return "", "missing_openai_key"
    if http_client is None:
        return "", "http_client_not_ready"
    if not text.strip():
        return "", "empty_text"

    prompt = (
        "You are an investing assistant. Summarize the following SEC filing content for a portfolio dashboard.\n\n"
        f"Context: {context}\n\n"
        "Output format:\n"
        "- 1 sentence: what happened\n"
        "- 2–4 bullet points: what changed / why it matters\n"
        "- 1 line: what to read next inside the filing (section hints)\n\n"
        "Be specific. Avoid hype. If uncertain, say so.\n\n"
        f"FILING TEXT:\n{text[:8000]}"
    )

    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
        "max_output_tokens": 260,
    }

    try:
        # Responses API (modern). If your key/account only supports ChatCompletions,
        # you can switch endpoint + payload.
        r = await http_client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        if r.status_code >= 400:
            return "", f"openai_http_{r.status_code}"
        data = r.json()
        # responses API returns output in data["output"][...]
        text_out = ""
        if isinstance(data, dict):
            out = data.get("output")
            if isinstance(out, list):
                # find first text chunk
                for item in out:
                    if isinstance(item, dict):
                        content = item.get("content")
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    text_out = (c.get("text") or "").strip()
                                    break
                    if text_out:
                        break
        if not text_out:
            return "", "openai_no_text"
        return text_out, None
    except Exception as e:
        return "", repr(e)


async def get_filings_for_tickers(tickers: List[str], state: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Cached daily in:
      state['sec_cache'][TICKER] = {date, cik10, filings:[...]}
      state['filing_summaries'][accession] = {date, summary}
    """
    today = utc_today()
    sec_cache = state.get("sec_cache") if isinstance(state.get("sec_cache"), dict) else {}
    summaries = state.get("filing_summaries") if isinstance(state.get("filing_summaries"), dict) else {}

    ticker_map = await fetch_sec_ticker_map(state)

    uniq = sorted(set(t.upper().strip() for t in tickers if t and not looks_like_numeric_id(t)))
    out: Dict[str, List[Dict[str, Any]]] = {}

    dbg = {"provider": "sec_edgar", "requested": len(uniq), "cached": 0, "fetched": 0, "summarized": 0, "errors": 0, "samples": []}

    async def one(t: str):
        nonlocal sec_cache, summaries
        t_sec = normalize_ticker_for_sec(t)
        cik10 = ticker_map.get(t_sec)

        if not cik10:
            out[t] = []
            dbg["errors"] += 1
            if len(dbg["samples"]) < 6:
                dbg["samples"].append({"ticker": t, "status": "no_cik"})
            return

        cached = sec_cache.get(t)
        if isinstance(cached, dict) and cached.get("date") == today and isinstance(cached.get("filings"), list):
            out[t] = cached["filings"]
            dbg["cached"] += 1
            return

        subs, sdbg = await fetch_sec_submissions(cik10)
        if not subs:
            out[t] = []
            dbg["errors"] += 1
            if len(dbg["samples"]) < 6:
                dbg["samples"].append({"ticker": t, "status": "no_submissions", "debug": sdbg})
            return

        filings = extract_recent_filings(subs, max_items=8)
        # sort: 8-K, 10-Q, 10-K, then rest
        filings.sort(key=lambda f: (filing_priority(f.get("form", "")), f.get("filingDate", "")), reverse=False)

        # add link and summary (cached by accession)
        enriched: List[Dict[str, Any]] = []
        for f in filings[:6]:
            acc = f.get("accessionNumber", "")
            prim = f.get("primaryDocument", "")
            link = build_filing_link(cik10, acc, prim) if acc and prim else ""
            form = f.get("form", "")
            desc = f.get("primaryDocDescription", "")

            # summary cache key: cik+acc+prim
            skey = f"{t}|{acc}|{prim}"
            cached_sum = summaries.get(skey)
            summary_text = ""
            sum_error = None

            if isinstance(cached_sum, dict) and cached_sum.get("date") == today and isinstance(cached_sum.get("summary"), str):
                summary_text = cached_sum["summary"]
            else:
                # If OpenAI available, fetch primary doc text and summarize.
                # If not, use fallback.
                if OPENAI_API_KEY:
                    async with _SUMMARY_SEM:
                        txt, _tdbg = await fetch_filing_primary_text(cik10, acc, prim)
                        context = f"{t} {form} filed {f.get('filingDate','')}. {desc}"
                        s, err = await openai_resume(txt, context=context)
                        if s:
                            summary_text = s
                            summaries[skey] = {"date": today, "summary": s}
                            dbg["summarized"] += 1
                        else:
                            sum_error = err or "summary_failed"
                            summary_text = simple_filing_resume(form, desc)
                            summaries[skey] = {"date": today, "summary": summary_text, "error": sum_error}
                else:
                    summary_text = simple_filing_resume(form, desc)
                    summaries[skey] = {"date": today, "summary": summary_text, "error": "no_openai"}

            enriched.append({
                **f,
                "cik10": cik10,
                "link": link,
                "summary": summary_text,
            })

        sec_cache[t] = {"date": today, "cik10": cik10, "filings": enriched}
        out[t] = enriched
        dbg["fetched"] += 1

        if len(dbg["samples"]) < 6:
            dbg["samples"].append({"ticker": t, "cik10": cik10, "filings": len(enriched)})

    await asyncio.gather(*(one(t) for t in uniq))

    state["sec_cache"] = sec_cache
    state["filing_summaries"] = summaries
    state["sec_debug"] = dbg
    await save_state(state)
    return out, dbg


# ============================================================
# PRESENTATION
# ============================================================

def build_portfolio_rows(
    agg: List[Dict[str, Any]],
    ticker_map: Dict[int, str],
    tech_map: Dict[int, Dict[str, Any]],
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

        cat = categorize_ticker(sym, categories)

        # fundamentals removed for now (you said you don't need market cap now; also your current setup lacks a provider)
        # keep placeholders so table is stable; you can re-add fundamentals later.
        fund = {"status": "NO_FUND"}

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
        })

    def fnum(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    rows.sort(key=lambda r: (r.get("category", "OTHER"), -fnum(r.get("weight_pct"))))
    return rows


def deterministic_brief(rows: List[Dict[str, Any]]) -> str:
    """
    You asked to remove winners/losers. Kept only actionable market structure signals.
    """
    if not rows:
        return "No portfolio rows."

    rsis = [(r["ticker"], r.get("tech", {}).get("rsi14")) for r in rows]
    rsis_clean = [(t, float(v)) for (t, v) in rsis if isinstance(v, (int, float))]
    overbought = sorted(rsis_clean, key=lambda x: x[1], reverse=True)[:6]
    oversold = sorted(rsis_clean, key=lambda x: x[1])[:6]

    below200 = [r["ticker"] for r in rows if r.get("tech", {}).get("above_sma200") is False][:12]
    illq = [r["ticker"] for r in rows if r.get("tech", {}).get("illiquid")][:12]

    lines = []
    lines.append("DAILY BRIEF (read-only)")
    lines.append("")
    if overbought:
        lines.append("RSI overbought: " + ", ".join([f"{t} {v:.1f}" for t, v in overbought]))
    if oversold:
        lines.append("RSI oversold:  " + ", ".join([f"{t} {v:.1f}" for t, v in oversold]))
    lines.append("")
    if below200:
        lines.append("Below SMA200 (trend watch): " + ", ".join(below200))
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
    news_debug = state.get("news_debug") or {}
    sec_debug = state.get("sec_debug") or {}

    news_cache = state.get("news_cache") or {}
    sec_cache = state.get("sec_cache") or {}

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

    # Build category tables + news + filings
    sections_html = ""
    order = ["ENERGY_OIL_GAS", "ENERGY_MIDSTREAM", "NUCLEAR_URANIUM", "TECH_SEMI_DATACENTER", "QUANTUM", "GOLD_PGM_MINERS", "OTHER"]

    for cat in order:
        rows = grouped.get(cat) or []
        if not rows:
            continue

        body = ""
        tickers_in_cat = []
        for r in rows[:400]:
            t = r.get("ticker", "")
            if t:
                tickers_in_cat.append(t.upper().strip())

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
                "</tr>"
            )

        if not body:
            body = "<tr><td colspan='6'>No positions in this category.</td></tr>"

        # News block
        news_html = "<p class='small'>No news cached yet. Run <code>/tasks/daily</code>.</p>"
        if isinstance(news_cache, dict):
            # build per ticker list
            parts = []
            for t in sorted(set(tickers_in_cat)):
                cached = news_cache.get(t)
                items = cached.get("items") if isinstance(cached, dict) else None
                if not isinstance(items, list) or not items:
                    continue
                lis = []
                for it in items[:4]:
                    title = (it.get("title") or "").strip()
                    link = (it.get("link") or "").strip()
                    source = (it.get("source") or "").strip()
                    pub = (it.get("pubDate") or "").strip()
                    if title and link:
                        lis.append(f"<li><a href='{link}' target='_blank' rel='noopener'>{title}</a> <span class='small'>({source or 'source'} • {pub})</span></li>")
                if lis:
                    parts.append(f"<div><b>{t}</b><ul>{''.join(lis)}</ul></div>")
            if parts:
                news_html = "".join(parts)

        # Filings block
        filings_html = "<p class='small'>No filings cached yet. Run <code>/tasks/daily</code>.</p>"
        if isinstance(sec_cache, dict):
            parts = []
            for t in sorted(set(tickers_in_cat)):
                cached = sec_cache.get(t)
                filings = cached.get("filings") if isinstance(cached, dict) else None
                if not isinstance(filings, list) or not filings:
                    continue
                lis = []
                for f in filings[:4]:
                    form = (f.get("form") or "").strip()
                    dt = (f.get("filingDate") or "").strip()
                    link = (f.get("link") or "").strip()
                    summ = (f.get("summary") or "").strip()
                    if link:
                        lis.append(
                            "<li>"
                            f"<b>{form}</b> <span class='small'>{dt}</span> — "
                            f"<a href='{link}' target='_blank' rel='noopener'>Open filing</a>"
                            f"<div class='small' style='margin-top:6px; white-space: pre-wrap;'>{summ}</div>"
                            "</li>"
                        )
                    else:
                        lis.append(
                            f"<li><b>{form}</b> <span class='small'>{dt}</span><div class='small' style='white-space: pre-wrap;'>{summ}</div></li>"
                        )
                if lis:
                    parts.append(f"<div><b>{t}</b><ul>{''.join(lis)}</ul></div>")
            if parts:
                filings_html = "".join(parts)

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
            </tr>
          </thead>
          <tbody>
            {body}
          </tbody>
        </table>

        <h3>News (free)</h3>
        {news_html}

        <h3>SEC Filings (resumes)</h3>
        {filings_html}
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
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 14px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
          th {{ background: #f5f5f5; text-align: left; position: sticky; top: 0; }}
          code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 4px; }}
          .small {{ color: #666; font-size: 12px; }}
          h3 {{ margin-top: 10px; }}
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

        <h2>Energy Thesis</h2>
        <p class="small">{ENERGY_THESIS_TEXT}</p>

        <h2>Material Events</h2>
        {bullets(material_events)}

        <h2>Technical Exceptions</h2>
        {bullets(technical_exceptions)}

        <h2>Action Required</h2>
        {bullets(action_required)}

        <p>Run <code>/tasks/daily</code> to refresh data (positions + tech + news + filings).</p>

        <h2>Daily Brief</h2>
        <pre style="white-space: pre-wrap; background:#fafafa; border:1px solid #eee; padding:12px;">{brief or "No brief yet. Run /tasks/daily."}</pre>

        {sections_html}

        <p class="small">
          API: <code>/api/portfolio</code> • <code>/api/daily-brief</code> • <code>/api/news</code> • <code>/api/filings</code> • <code>/api/categories</code><br/>
          Debug: <code>/debug/mapping-last</code> • <code>/debug/tech-last</code> • <code>/debug/news-last</code> • <code>/debug/sec-last</code>
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
            f"eToro keys={'True' if (bool(ETORO_API_KEY) and bool(ETORO_USER_KEY)) else 'False'}."
        )

        payload = await etoro_get_real_pnl()
        raw_positions = extract_positions(payload)
        agg_positions, stats = aggregate_positions_by_instrument(raw_positions)

        material_events.append(f"Pulled eToro successfully. Lots: {stats['lots_count']} | Unique instruments: {stats['unique_instruments_count']}")

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

        # Build portfolio rows
        portfolio_rows = build_portfolio_rows(agg_positions, ticker_cache, tech_map, categories)

        # tickers in use (for news + filings)
        tickers = []
        for r in portfolio_rows:
            t = (r.get("ticker") or "").upper().strip()
            if t and not looks_like_numeric_id(t):
                tickers.append(t)

        tickers = sorted(set(tickers))

        # 3) news (free)
        news_map, news_dbg = await get_news_for_tickers(tickers, state)

        # 4) filings + resumes
        filings_map, sec_dbg = await get_filings_for_tickers(tickers, state)

        # 5) brief (no winners/losers)
        brief = deterministic_brief(portfolio_rows)

        # 6) discord (optional)
        tech_done = safe_int(tech_dbg.get("computed", 0))
        tech_req = safe_int(tech_dbg.get("requested", len(instrument_ids)))
        tech_skip = safe_int(tech_dbg.get("skipped", 0))
        tech_den = max(0, tech_req - tech_skip)

        await discord_notify(
            f"✅ Daily done | tickers {mapping.get('cached',0)}/{len(instrument_ids)} | tech {tech_done}/{tech_den} | news {news_dbg.get('fetched',0)}/{news_dbg.get('requested',0)} | filings {sec_dbg.get('fetched',0)}/{sec_dbg.get('requested',0)}"
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
            # news/filings are stored in cache sections
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
            "news_requested": news_dbg.get("requested", 0),
            "news_fetched": news_dbg.get("fetched", 0),
            "filings_requested": sec_dbg.get("requested", 0),
            "filings_fetched": sec_dbg.get("fetched", 0),
            "filings_summarized_today": sec_dbg.get("summarized", 0),
        }


@app.get("/api/portfolio")
async def api_portfolio():
    state = await load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "stats": state.get("stats") or {},
        "mapping": state.get("mapping") or {},
        "tech_debug": state.get("tech_debug") or {},
        "news_debug": state.get("news_debug") or {},
        "sec_debug": state.get("sec_debug") or {},
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
        "news_debug": state.get("news_debug") or {},
        "sec_debug": state.get("sec_debug") or {},
        "energy_thesis": state.get("energy_thesis") or ENERGY_THESIS_TEXT,
    })


@app.get("/api/news")
async def api_news():
    state = await load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "news_debug": state.get("news_debug") or {},
        "news_cache": state.get("news_cache") or {},
        "note": "News is Google News RSS (free). Items include source, time, and direct links.",
    })


@app.get("/api/filings")
async def api_filings():
    state = await load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "sec_debug": state.get("sec_debug") or {},
        "sec_cache": state.get("sec_cache") or {},
        "note": "Filings from SEC EDGAR submissions JSON; summaries cached daily.",
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


@app.get("/debug/news-last")
async def debug_news_last():
    state = await load_state()
    return JSONResponse(state.get("news_debug") or {"note": "Run /tasks/daily first."})


@app.get("/debug/sec-last")
async def debug_sec_last():
    state = await load_state()
    return JSONResponse(state.get("sec_debug") or {"note": "Run /tasks/daily first."})


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
