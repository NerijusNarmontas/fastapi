# main.py
# Robust single-file FastAPI investing agent (no more silent 500s):
# - Portfolio from eToro (/tasks/daily)
# - Invested $ column + Weight % + P&L % (green/red)
# - Compact TECH line per ticker: 1M / 200DMA trend / RSI / Liquidity
# - Energy Universe Scanner (/tasks/scan-energy) using free sources:
#     Universe: StockAnalysis Energy sector list + URA holdings (uranium complex)
#     Technicals: Stooq
#     Fundamentals (US-first): SEC companyfacts (requires SEC_UA)
# - Scanner results visible on / and /scanner
# - Built-in debugging: /debug/last-error, /debug/env, /debug/state-keys
#
# Railway startCommand:
#   hypercorn main:app --bind "0.0.0.0:$PORT"

import os
import re
import json
import uuid
import math
import time
import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

app = FastAPI(title="My AI Investing Agent")

# ----------------------------
# Config
# ----------------------------
ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

STATE_PATH = os.getenv("STATE_PATH", "/tmp/investing_agent_state.json")

ETORO_REAL_PNL_URL = os.getenv(
    "ETORO_REAL_PNL_URL",
    "https://public-api.etoro.com/api/v1/trading/info/real/pnl",
).strip()

ETORO_SEARCH_URL = os.getenv(
    "ETORO_SEARCH_URL",
    "https://public-api.etoro.com/api/v1/market-data/search",
).strip()

DEFAULT_UA = os.getenv(
    "DEFAULT_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
).strip()

SEC_UA = os.getenv("SEC_UA", "").strip()

NEWS_PER_TICKER = int(os.getenv("NEWS_PER_TICKER", "6"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "12"))

# Energy scanner sizing (to avoid timeouts on free infra)
ENERGY_UNIVERSE_MAX = int(os.getenv("ENERGY_UNIVERSE_MAX", "900"))  # stored universe size
ENERGY_SCAN_MAX = int(os.getenv("ENERGY_SCAN_MAX", "250"))          # scanned per run (keep stable)

# Universe sources
ENERGY_SCAN_MODE = os.getenv("ENERGY_SCAN_MODE", "stockanalysis+ura").strip().lower()
ENERGY_UNIVERSE_EXTRA = [x.strip().upper() for x in os.getenv("ENERGY_UNIVERSE_EXTRA", "").split(",") if x.strip()]

# Exclude refiners by default (your thesis)
REFINER_EXCLUDE = set(
    x.strip().upper() for x in os.getenv(
        "REFINER_EXCLUDE",
        "VLO,PSX,MPC,PBF,HFC,DINO,DK,CLMT"
    ).split(",") if x.strip()
)

# Crypto handling (kept)
CRYPTO_EXCLUDE = set(s.strip().upper() for s in os.getenv("CRYPTO_EXCLUDE", "W").split(",") if s.strip())
CRYPTO_TICKERS = set(
    s.strip().upper()
    for s in os.getenv(
        "CRYPTO_TICKERS",
        "BTC,ETH,SOL,AVAX,OP,ARB,JTO,RUNE,W,EIGEN,STRK,ONDO,SEI,WLD,PYTH,HBAR,CRO,HYPE,RPL,ARBE",
    ).split(",")
    if s.strip()
)
CRYPTO_INSTRUMENT_IDS = set(
    int(x.strip())
    for x in os.getenv("CRYPTO_INSTRUMENT_IDS", "100000").split(",")
    if x.strip().isdigit()
)

ENERGY_THESIS = {
    "avoid": ["refining"],
    "preferred": ["gas", "oil", "midstream", "offshore (select)", "uranium", "nuclear fuel", "smrs"],
    "style": ["capital discipline", "high FCF", "low debt", "shareholder return", "liquidity OK"],
}

# ----------------------------
# Error capture middleware (prevents blank 500)
# ----------------------------
_LAST_ERROR = {"when": None, "where": None, "trace": None}

@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        tb = traceback.format_exc()
        _LAST_ERROR["when"] = utc_now_iso()
        _LAST_ERROR["where"] = f"{request.method} {request.url.path}"
        _LAST_ERROR["trace"] = tb

        try:
            st = load_state()
            st["last_error"] = _LAST_ERROR
            save_state(st)
        except Exception:
            pass

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "where": _LAST_ERROR["where"],
                "when": _LAST_ERROR["when"],
                "trace": tb[-9000:],
            },
        )

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

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)
    return t

def html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

def normalize_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        return f"{x:,.0f}"
    except Exception:
        return ""

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def tradingview_url(ticker: str, exchange: str = "") -> str:
    t = normalize_ticker(ticker)
    ex = (exchange or "").strip().upper()
    if ex:
        return f"https://www.tradingview.com/chart/?symbol={ex}:{t}"
    return f"https://www.tradingview.com/symbols/{t}/"

def edgar_search_link(ticker: str) -> str:
    return f"https://www.sec.gov/edgar/search/#/q={quote_plus(ticker)}&sort=desc"

def is_crypto_ticker(ticker: str, instrument_id: Optional[int] = None) -> bool:
    t = normalize_ticker(ticker)
    if t in CRYPTO_TICKERS:
        return True
    if instrument_id is not None and instrument_id in CRYPTO_INSTRUMENT_IDS:
        return True
    if t.endswith("-USD") and t[:-4] in CRYPTO_TICKERS:
        return True
    return False

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
        "user-agent": DEFAULT_UA,
        "accept": "application/json",
    }

async def etoro_get_real_pnl() -> Dict[str, Any]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        raise HTTPException(status_code=400, detail="Missing ETORO_API_KEY or ETORO_USER_KEY")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(ETORO_REAL_PNL_URL, headers=etoro_headers())
        if r.status_code >= 400:
            try:
                payload = r.json()
            except Exception:
                payload = {"text": r.text}
            raise HTTPException(status_code=r.status_code, detail=payload)
        return r.json()

async def etoro_search(params: Dict[str, str]) -> Tuple[int, Any]:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(ETORO_SEARCH_URL, headers=etoro_headers(), params=params)
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        return r.status_code, data

def extract_positions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    cp = payload.get("clientPortfolio") or payload.get("ClientPortfolio") or {}
    if isinstance(cp, dict) and isinstance(cp.get("positions"), list):
        return cp["positions"]
    if isinstance(payload.get("positions"), list):
        return payload["positions"]
    return []

def pick_instrument_id(p: Dict[str, Any]) -> Optional[int]:
    iid = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
    try:
        return int(iid) if iid is not None else None
    except Exception:
        return None

def pick_unrealized_pnl(p: Dict[str, Any]) -> Optional[float]:
    up = p.get("unrealizedPnL") or p.get("unrealizedPnl") or p.get("unrealized_pnl")
    if isinstance(up, dict):
        return normalize_number(up.get("pnL"))
    return normalize_number(up)

def pick_initial_usd(p: Dict[str, Any]) -> Optional[float]:
    return normalize_number(p.get("initialAmountInDollars") or p.get("initialAmount") or p.get("initial_amount_usd"))

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

def _extract_ticker_from_search_item(item: Dict[str, Any]) -> str:
    for k in ("internalSymbolFull", "symbolFull", "symbol"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _extract_exchange_from_search_item(item: Dict[str, Any]) -> str:
    for k in ("exchange", "exchangeName", "exchangeCode", "marketName", "market"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

async def map_instrument_ids_to_tickers(instrument_ids: List[int]) -> Tuple[Dict[int, str], Dict[str, Any]]:
    ids = sorted(set(int(x) for x in instrument_ids if x is not None))
    out: Dict[int, str] = {}
    instrument_meta: Dict[str, Dict[str, str]] = {}
    debug: Dict[str, Any] = {"requested": len(ids), "mapped": 0, "failed": 0, "samples": []}
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
                        ex = _extract_exchange_from_search_item(it)
                        instrument_meta[str(iid)] = {"exchange": ex}
                        debug["mapped"] += 1
                        if len(debug["samples"]) < 10:
                            debug["samples"].append({"instrumentID": iid, "ticker": t, "exchange": ex, "via": "instrumentId", "status": status})
                        return

            status2, data2 = await etoro_search({"internalInstrumentId": str(iid), "pageSize": "5", "pageNumber": "1"})
            items2 = data2.get("items") if isinstance(data2, dict) else None
            if isinstance(items2, list) and items2:
                for it in items2:
                    if not isinstance(it, dict):
                        continue
                    t = _extract_ticker_from_search_item(it)
                    if t:
                        out[iid] = t
                        ex = _extract_exchange_from_search_item(it)
                        instrument_meta[str(iid)] = {"exchange": ex}
                        debug["mapped"] += 1
                        if len(debug["samples"]) < 10:
                            debug["samples"].append({"instrumentID": iid, "ticker": t, "exchange": ex, "via": "internalInstrumentId", "status": status2})
                        return

            debug["failed"] += 1
            if len(debug["samples"]) < 10:
                debug["samples"].append({"instrumentID": iid, "status": "no_match_or_no_ticker"})

    await asyncio.gather(*(one(i) for i in ids))
    debug["instrument_meta"] = instrument_meta
    return out, debug

def build_portfolio_rows(agg: List[Dict[str, Any]], ticker_map: Dict[int, str]) -> List[Dict[str, Any]]:
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0
    rows: List[Dict[str, Any]] = []

    for a in agg:
        iid = int(a["instrumentID"])
        ticker_raw = ticker_map.get(iid) or str(iid)
        ticker = normalize_ticker(ticker_raw)

        if ticker in CRYPTO_EXCLUDE and ticker not in CRYPTO_TICKERS:
            continue

        initial = normalize_number(a.get("initialAmountInDollars"))
        unreal = normalize_number(a.get("unrealizedPnL"))

        weight_pct = (initial / total_initial * 100.0) if (total_initial > 0 and initial and initial > 0) else None
        pnl_pct = (unreal / initial * 100.0) if (initial and initial != 0 and unreal is not None) else None

        rows.append({
            "ticker": ticker,
            "lots": int(a.get("lots", 0) or 0),
            "instrumentID": str(iid),
            "invested_usd": float(initial or 0.0),
            "unreal_usd": float(unreal or 0.0),
            "weight_pct_num": float(weight_pct or 0.0),
            "pnl_pct_num": float(pnl_pct or 0.0) if pnl_pct is not None else 0.0,
            "pnl_pct_is_na": pnl_pct is None,
            "invested_disp": fmt_money(float(initial or 0.0)),
            "weight_pct": f"{weight_pct:.2f}" if weight_pct is not None else "",
            "pnl_pct": f"{pnl_pct:.2f}" if pnl_pct is not None else "",
        })
    return rows

# ----------------------------
# TECH (Stooq)
# ----------------------------
def _stooq_symbol_candidates(ticker: str) -> List[str]:
    t = normalize_ticker(ticker)
    if t.endswith(".US"):
        base = t[:-3]
        return [f"{base}.US", base, t]
    return [t, t.replace("-", "."), t.replace(".", "-")]

async def fetch_stooq_daily(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, float]]:
    for sym in _stooq_symbol_candidates(ticker):
        url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
        r = await client.get(url, headers={"user-agent": DEFAULT_UA})
        if r.status_code != 200 or "Date,Open,High,Low,Close,Volume" not in r.text:
            continue
        lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        if len(lines) < 30:
            continue
        out = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) < 6:
                continue
            try:
                h = float(parts[2]); l = float(parts[3]); c = float(parts[4])
                v = float(parts[5]) if parts[5] not in ("", "nan", "NaN") else 0.0
            except Exception:
                continue
            out.append({"high": h, "low": l, "close": c, "volume": v})
        if len(out) >= 30:
            return out
    return []

def sma(values: List[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n

def rsi14(closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    gains, losses = [], []
    for i in range(-n, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0:
            gains.append(ch); losses.append(0.0)
        else:
            gains.append(0.0); losses.append(-ch)
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def monthly_return_pct(closes: List[float], lookback_days: int = 21) -> Optional[float]:
    if len(closes) < lookback_days + 1:
        return None
    a = closes[-lookback_days - 1]; b = closes[-1]
    if a == 0:
        return None
    return (b / a - 1.0) * 100.0

def liquidity_flag(bars: List[Dict[str, float]], days: int = 20) -> Tuple[str, Optional[float]]:
    if len(bars) < days:
        return ("NA", None)
    recent = bars[-days:]
    dvs = [(x["close"] * (x.get("volume") or 0.0)) for x in recent]
    avg_dv = (sum(dvs) / days) if dvs else None
    if avg_dv is None:
        return ("NA", None)
    return ("OK" if avg_dv >= 2_000_000 else "LOW", avg_dv)

def build_compact_line(ticker: str, bars: List[Dict[str, float]]) -> Dict[str, Any]:
    closes = [x["close"] for x in bars]
    last = closes[-1] if closes else None
    mret = monthly_return_pct(closes)
    sma200 = sma(closes, 200)
    trend = None
    if last is not None and sma200 is not None:
        trend = "above" if last >= sma200 else "below"
    rsi = rsi14(closes, 14)
    liq, avg_dv = liquidity_flag(bars, 20)
    warn = bool((rsi is not None and rsi <= 30) or (liq == "LOW"))

    parts = []
    if mret is not None:
        parts.append(f"1M {mret:+.0f}%")
    if trend:
        parts.append(f"Trend {trend} 200DMA")
    if rsi is not None:
        parts.append(f"RSI {rsi:.0f}")
    parts.append(f"Liq {liq}")

    return {
        "ticker": ticker,
        "last_close": last,
        "mret_1m_pct": mret,
        "sma200": sma200,
        "trend_vs_200": trend,
        "rsi14": rsi,
        "liq": liq,
        "avg_dollar_vol": avg_dv,
        "compact": " | ".join(parts),
        "warn": warn,
    }

async def compute_tech_cache(tickers: List[str]) -> Dict[str, Any]:
    sem = asyncio.Semaphore(CONCURRENCY)
    tech_cache: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        async def do_one(t: str):
            async with sem:
                try:
                    bars = await fetch_stooq_daily(client, t)
                    if bars:
                        tech_cache[t] = build_compact_line(t, bars)
                    else:
                        tech_cache[t] = {"ticker": t, "compact": "Tech NA", "warn": False, "liq": "NA"}
                except Exception:
                    tech_cache[t] = {"ticker": t, "compact": "Tech ERR", "warn": True, "liq": "NA"}
        await asyncio.gather(*(do_one(t) for t in tickers))
    return tech_cache

# ----------------------------
# News (Google RSS) - portfolio only
# ----------------------------
def parse_rss_items(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    channel = root.find("channel")
    if channel is None:
        return []
    out: List[Dict[str, Any]] = []
    for item in channel.findall("item"):
        out.append({
            "title": (item.findtext("title") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "pubDate": (item.findtext("pubDate") or "").strip(),
            "source": (item.findtext("source") or "").strip(),
        })
    return out

def resume_news(title: str) -> str:
    t = (title or "").lower()
    tags = []
    if any(k in t for k in ["earnings", "guidance", "revenue", "eps"]): tags.append("Earnings")
    if any(k in t for k in ["acquire", "acquisition", "merger", "buyout"]): tags.append("M&A")
    if any(k in t for k in ["offering", "convertible", "debt", "notes", "atm"]): tags.append("Financing")
    if any(k in t for k in ["lawsuit", "probe", "investigation"]): tags.append("Legal")
    if any(k in t for k in ["upgrade", "downgrade", "price target"]): tags.append("Analyst")
    return " • ".join(tags) if tags else "Headline"

async def fetch_google_news_for_ticker(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, Any]]:
    q = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = await client.get(url, headers={"accept": "application/rss+xml,*/*", "user-agent": DEFAULT_UA})
    if r.status_code != 200:
        return []
    items = parse_rss_items(r.text)
    out: List[Dict[str, Any]] = []
    seen = set()
    for it in items:
        link = (it.get("link") or "").strip()
        title = (it.get("title") or "").strip()
        if not link or not title or link in seen:
            continue
        seen.add(link)
        out.append({
            "ticker": ticker,
            "title": title,
            "link": link,
            "source": it.get("source") or "Google News",
            "published": it.get("pubDate") or "",
            "resume": resume_news(title),
        })
        if len(out) >= NEWS_PER_TICKER:
            break
    return out

async def compute_news(tickers: List[str]) -> Dict[str, Any]:
    sem = asyncio.Semaphore(CONCURRENCY)
    cache: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        async def do_one(t: str):
            async with sem:
                try:
                    cache[t] = await fetch_google_news_for_ticker(client, t)
                except Exception:
                    cache[t] = []
        await asyncio.gather(*(do_one(t) for t in tickers))
    return cache

# ----------------------------
# Energy Universe (free)
# ----------------------------
STOCKANALYSIS_ENERGY_URL = "https://stockanalysis.com/stocks/sector/energy/"
GLOBALX_URA_CSV = "https://www.globalxetfs.com/funds/ura/?download=holdings"

def parse_stockanalysis_energy_tickers(html: str, limit: int) -> List[str]:
    # Best-effort: pull /stocks/{symbol}/ links
    syms = re.findall(r'href="/stocks/([a-z0-9\.\-]+)/"', html, flags=re.I)
    out, seen = [], set()
    for s in syms:
        t = normalize_ticker(s)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out

def parse_csv_tickers(csv_text: str) -> List[str]:
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    if "ticker" not in header:
        return []
    idx = header.index("ticker")
    out, seen = [], set()
    for ln in lines[1:]:
        parts = [p.strip().strip('"') for p in ln.split(",")]
        if idx >= len(parts):
            continue
        t = normalize_ticker(parts[idx])
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

async def build_energy_universe(state: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    cache = state.get("energy_universe_cache") or {}
    meta = cache.get("meta") or {}
    ts = float(meta.get("ts") or 0.0)
    if cache.get("tickers") and (now - ts) < (7 * 24 * 3600):
        return cache

    tickers: List[str] = []
    sources: List[str] = []
    warnings: List[str] = []

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        if "stockanalysis" in ENERGY_SCAN_MODE:
            try:
                r = await client.get(STOCKANALYSIS_ENERGY_URL, headers={"user-agent": DEFAULT_UA})
                if r.status_code == 200:
                    tickers.extend(parse_stockanalysis_energy_tickers(r.text, ENERGY_UNIVERSE_MAX))
                    sources.append("stockanalysis_energy")
                else:
                    warnings.append(f"stockanalysis status {r.status_code}")
            except Exception as e:
                warnings.append(f"stockanalysis error: {repr(e)}")

        if "ura" in ENERGY_SCAN_MODE:
            try:
                r = await client.get(GLOBALX_URA_CSV, headers={"user-agent": DEFAULT_UA, "accept": "text/csv,*/*"})
                if r.status_code == 200:
                    tickers.extend(parse_csv_tickers(r.text))
                    sources.append("ura_holdings")
                else:
                    warnings.append(f"ura status {r.status_code}")
            except Exception as e:
                warnings.append(f"ura error: {repr(e)}")

    tickers.extend(ENERGY_UNIVERSE_EXTRA)

    # normalize + dedupe + exclude refiners
    out, seen = [], set()
    for t in tickers:
        tt = normalize_ticker(t)
        if not tt or tt in seen:
            continue
        seen.add(tt)
        if tt in REFINER_EXCLUDE:
            continue
        out.append(tt)

    cache = {
        "tickers": out,
        "sources": sources,
        "warnings": warnings,
        "meta": {"ts": now, "date": utc_now_iso(), "count": len(out)},
    }
    state["energy_universe_cache"] = cache
    save_state(state)
    return cache

# ----------------------------
# SEC Fundamentals (US-first, free)
# ----------------------------
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

CFO_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquireOilAndGasPropertyPlantAndEquipment",
]
SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]

async def get_sec_ticker_map(state: Dict[str, Any]) -> Dict[str, str]:
    now = time.time()
    cache = state.get("sec_ticker_map") or {}
    meta = state.get("sec_ticker_map_meta") or {}
    ts = float(meta.get("ts") or 0.0)
    if cache and (now - ts) < (7 * 24 * 3600):
        return cache

    if not SEC_UA:
        return cache

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(SEC_TICKER_MAP_URL, headers={"user-agent": SEC_UA, "accept": "application/json"})
        if r.status_code != 200:
            return cache
        data = r.json()

    out: Dict[str, str] = {}
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            t = normalize_ticker(row.get("ticker") or "")
            cik = str(row.get("cik") or "").strip()
            if not t or not cik.isdigit():
                continue
            out[t] = cik.zfill(10)

    state["sec_ticker_map"] = out
    state["sec_ticker_map_meta"] = {"ts": now, "date": utc_now_iso()}
    save_state(state)
    return out

def _pick_latest_quarters(facts: Dict[str, Any], concept_names: List[str]) -> List[float]:
    try:
        usgaap = facts.get("facts", {}).get("us-gaap", {})
    except Exception:
        return []
    series = None
    for c in concept_names:
        if c in usgaap:
            series = usgaap.get(c)
            break
    if not series:
        return []
    units = series.get("units") or {}
    candidates = units.get("USD") or units.get("usd") or []
    if not isinstance(candidates, list):
        return []
    vals = []
    for x in candidates:
        if not isinstance(x, dict):
            continue
        v = x.get("val")
        if v is None:
            continue
        form = (x.get("form") or "").upper()
        fp = (x.get("fp") or "").upper()
        if form not in ("10-Q", "10-K"):
            continue
        if fp and fp not in ("Q1", "Q2", "Q3", "Q4", "FY"):
            continue
        try:
            vals.append((x.get("end") or "", float(v)))
        except Exception:
            continue
    vals.sort(key=lambda z: z[0], reverse=True)
    return [v for _, v in vals[:8]]

def _pick_latest_shares(facts: Dict[str, Any]) -> List[float]:
    try:
        usgaap = facts.get("facts", {}).get("us-gaap", {})
    except Exception:
        return []
    series = None
    for c in SHARES_CONCEPTS:
        if c in usgaap:
            series = usgaap.get(c)
            break
    if not series:
        return []
    units = series.get("units") or {}
    candidates = units.get("shares") or units.get("SHARES") or units.get("Shares") or []
    if not isinstance(candidates, list):
        return []
    vals = []
    for x in candidates:
        if not isinstance(x, dict):
            continue
        v = x.get("val")
        if v is None:
            continue
        form = (x.get("form") or "").upper()
        if form not in ("10-Q", "10-K"):
            continue
        try:
            vals.append((x.get("end") or "", float(v)))
        except Exception:
            continue
    vals.sort(key=lambda z: z[0], reverse=True)
    return [v for _, v in vals[:8]]

async def fetch_sec_companyfacts(client: httpx.AsyncClient, cik10: str) -> Optional[Dict[str, Any]]:
    if not SEC_UA:
        return None
    url = SEC_COMPANYFACTS_URL.format(cik=cik10)
    r = await client.get(url, headers={"user-agent": SEC_UA, "accept": "application/json"})
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

def compute_fundamentals_from_facts(facts: Dict[str, Any], last_price: Optional[float]) -> Dict[str, Any]:
    cfo_q = _pick_latest_quarters(facts, CFO_CONCEPTS)
    capex_q = _pick_latest_quarters(facts, CAPEX_CONCEPTS)
    shares = _pick_latest_shares(facts)

    ttm_cfo = sum(cfo_q[:4]) if len(cfo_q) >= 4 else None
    ttm_capex = sum(capex_q[:4]) if len(capex_q) >= 4 else None
    fcf = (ttm_cfo - ttm_capex) if (ttm_cfo is not None and ttm_capex is not None) else None

    sh_latest = shares[0] if shares else None
    sh_old = shares[4] if len(shares) >= 5 else (shares[-1] if len(shares) >= 2 else None)

    dilution_yoy = None
    if sh_latest and sh_old and sh_old != 0:
        dilution_yoy = (sh_latest / sh_old - 1.0) * 100.0

    mktcap = None
    if last_price is not None and sh_latest is not None:
        mktcap = last_price * sh_latest

    fcf_yield = None
    if fcf is not None and mktcap and mktcap != 0:
        fcf_yield = (fcf / mktcap) * 100.0

    return {
        "ttm_cfo": ttm_cfo,
        "ttm_capex": ttm_capex,
        "fcf": fcf,
        "shares": sh_latest,
        "dilution_yoy_pct": dilution_yoy,
        "mktcap_est": mktcap,
        "fcf_yield_pct": fcf_yield,
        "compact": fundamentals_compact({
            "fcf_yield_pct": fcf_yield,
            "dilution_yoy_pct": dilution_yoy,
        })
    }

def fundamentals_compact(f: Dict[str, Any]) -> str:
    fy = f.get("fcf_yield_pct")
    dil = f.get("dilution_yoy_pct")
    parts = []
    if fy is not None and math.isfinite(fy):
        parts.append(f"FCF yld {fy:.1f}%")
    else:
        parts.append("FCF yld NA")
    if dil is not None and math.isfinite(dil):
        parts.append(f"Shares {dil:+.1f}% YoY")
    else:
        parts.append("Shares NA")
    return " | ".join(parts)

# ----------------------------
# Scanner scoring
# ----------------------------
def score_candidate(ticker: str, tech: Dict[str, Any], fund: Dict[str, Any]) -> Dict[str, Any]:
    liq = tech.get("liq") or "NA"
    trend = tech.get("trend_vs_200")
    rsi = tech.get("rsi14")
    fy = fund.get("fcf_yield_pct")
    dil = fund.get("dilution_yoy_pct")

    score = 50.0
    tags = []

    if liq == "OK":
        score += 10; tags.append("Liq OK")
    elif liq == "LOW":
        score -= 25; tags.append("Liq LOW")
    else:
        score -= 10; tags.append("Liq NA")

    if fy is None or not math.isfinite(fy):
        score -= 5; tags.append("FCF NA")
    else:
        if fy >= 12: score += 20; tags.append("FCF strong")
        elif fy >= 8: score += 14; tags.append("FCF good")
        elif fy >= 4: score += 6; tags.append("FCF ok")
        elif fy >= 0: score -= 6; tags.append("FCF weak")
        else: score -= 15; tags.append("FCF negative")

    if dil is None or not math.isfinite(dil):
        tags.append("Shares NA")
    else:
        if dil <= -2: score += 8; tags.append("Buyback")
        elif dil <= 2: score += 2; tags.append("No dilution")
        else:
            score -= min(15, dil); tags.append("Dilution risk")

    if trend == "above":
        score += 6; tags.append("Above 200DMA")
    elif trend == "below":
        score -= 6; tags.append("Below 200DMA")

    if rsi is not None and math.isfinite(rsi):
        if rsi <= 30: tags.append("RSI<30")
        elif rsi >= 70: tags.append("RSI>70")

    bucket = "Satellite"
    if (liq == "OK") and (fy is not None and math.isfinite(fy) and fy >= 8) and (dil is None or dil <= 2):
        bucket = "Core"
    if liq == "LOW" or (fy is not None and math.isfinite(fy) and fy < 0) or (ticker in REFINER_EXCLUDE):
        bucket = "Avoid"

    score = max(0.0, min(100.0, score))
    return {
        "ticker": ticker,
        "score": round(score, 1),
        "bucket": bucket,
        "tags": tags[:6],
        "tech": tech,
        "fund": fund,
    }

async def run_energy_scan(state: Dict[str, Any]) -> Dict[str, Any]:
    uni = await build_energy_universe(state)
    universe = (uni.get("tickers") or [])[:max(10, ENERGY_SCAN_MAX)]

    tech_cache = await compute_tech_cache(universe)

    sec_map: Dict[str, str] = {}
    sec_warning = None
    try:
        sec_map = await get_sec_ticker_map(state)
    except Exception as e:
        sec_warning = f"SEC map error: {repr(e)}"
        sec_map = {}

    fund_cache: Dict[str, Any] = {}
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        async def do_one(t: str):
            async with sem:
                tech = tech_cache.get(t) or {}
                last_price = tech.get("last_close")
                cik = sec_map.get(t)
                if not cik:
                    fund_cache[t] = {"note": "Fund NA (no SEC CIK)"}
                    return
                facts = await fetch_sec_companyfacts(client, cik)
                if not facts:
                    fund_cache[t] = {"note": "Fund NA (SEC fetch fail)"}
                    return
                fund_cache[t] = compute_fundamentals_from_facts(facts, last_price)

        # If SEC_UA missing, skip SEC calls completely
        if SEC_UA and sec_map:
            await asyncio.gather(*(do_one(t) for t in universe))
        else:
            for t in universe:
                fund_cache[t] = {"note": "Fund NA (SEC_UA missing)" if not SEC_UA else "Fund NA"}

    rows = []
    for t in universe:
        rows.append(score_candidate(
            t,
            tech_cache.get(t) or {"compact": "Tech NA", "liq": "NA"},
            fund_cache.get(t) or {"note": "Fund NA"},
        ))

    bucket_rank = {"Core": 0, "Satellite": 1, "Avoid": 2}
    rows.sort(key=lambda r: (bucket_rank.get(r.get("bucket"), 9), -float(r.get("score") or 0.0), r.get("ticker") or ""))

    result = {
        "date": utc_now_iso(),
        "universe_count": len(uni.get("tickers") or []),
        "scanned": len(universe),
        "sources": uni.get("sources") or [],
        "warnings": (uni.get("warnings") or []) + ([sec_warning] if sec_warning else []),
        "top": rows[:200],
        "all_count": len(rows),
    }
    state["energy_scan"] = result
    state["energy_thesis"] = ENERGY_THESIS
    save_state(state)
    return result

# ----------------------------
# Sorting helpers (portfolio)
# ----------------------------
SORT_KEYS = {
    "ticker": lambda r: (r.get("ticker") or ""),
    "invested": lambda r: float(r.get("invested_usd") or 0.0),
    "weight": lambda r: float(r.get("weight_pct_num") or 0.0),
    "pnl": lambda r: (-1e18 if r.get("pnl_pct_is_na") else float(r.get("pnl_pct_num") or 0.0)),
    "lots": lambda r: int(r.get("lots") or 0),
}
def sort_rows(rows: List[Dict[str, Any]], sort_by: str, direction: str) -> List[Dict[str, Any]]:
    key_fn = SORT_KEYS.get(sort_by, SORT_KEYS["weight"])
    rev = (direction or "desc").lower() != "asc"
    return sorted(rows, key=key_fn, reverse=rev)

# ----------------------------
# Routes
# ----------------------------
@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"

@app.get("/debug/last-error")
async def debug_last_error():
    st = load_state()
    return JSONResponse(st.get("last_error") or _LAST_ERROR or {"note": "no error captured yet"})

@app.get("/debug/env")
async def debug_env():
    return JSONResponse({
        "has_ETORO_API_KEY": bool(ETORO_API_KEY),
        "has_ETORO_USER_KEY": bool(ETORO_USER_KEY),
        "has_SEC_UA": bool(SEC_UA),
        "HTTP_TIMEOUT": HTTP_TIMEOUT,
        "CONCURRENCY": CONCURRENCY,
        "ENERGY_UNIVERSE_MAX": ENERGY_UNIVERSE_MAX,
        "ENERGY_SCAN_MAX": ENERGY_SCAN_MAX,
        "ENERGY_SCAN_MODE": ENERGY_SCAN_MODE,
    })

@app.get("/debug/state-keys")
async def debug_state_keys():
    st = load_state()
    return JSONResponse({"keys": sorted(list(st.keys()))})

@app.get("/", response_class=HTMLResponse)
async def dashboard(sort: str = "weight", dir: str = "desc"):
    state = load_state()
    last_update = state.get("date") or "never"
    stats = state.get("stats") or {}
    mapping = state.get("mapping") or {}
    positions = state.get("positions") or []
    instrument_meta = state.get("instrument_meta") or {}
    tech_cache = state.get("tech_cache") or {}
    scan = state.get("energy_scan") or {}

    sort_by = (sort or "weight").lower()
    direction = (dir or "desc").lower()
    if sort_by not in SORT_KEYS:
        sort_by = "weight"
    if direction not in ("asc", "desc"):
        direction = "desc"

    def hdr(label: str, key: str) -> str:
        next_dir = "asc" if (sort_by == key and direction == "desc") else "desc"
        arrow = ""
        if sort_by == key:
            arrow = " ▲" if direction == "asc" else " ▼"
        return f"<a href='/?sort={html_escape(key)}&dir={html_escape(next_dir)}'>{html_escape(label)}{arrow}</a>"

    def pnl_class(pnl_num: Optional[float]) -> str:
        if pnl_num is None:
            return "na"
        if pnl_num > 0:
            return "pos"
        if pnl_num < 0:
            return "neg"
        return "flat"

    def render_portfolio(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "<div class='muted'>No positions (run <code>/tasks/daily</code>).</div>"
        rows_sorted = sort_rows(rows, sort_by, direction)
        out = []
        out.append("<table><thead><tr>"
                   f"<th>{hdr('Ticker','ticker')}</th>"
                   f"<th>{hdr('Lots','lots')}</th>"
                   f"<th>{hdr('Invested $','invested')}</th>"
                   f"<th>{hdr('Weight %','weight')}</th>"
                   f"<th>{hdr('P&L %','pnl')}</th>"
                   "</tr></thead><tbody>")
        for r in rows_sorted[:400]:
            t = r.get("ticker", "")
            iid = (r.get("instrumentID") or "0")
            ex = (instrument_meta.get(str(iid)) or {}).get("exchange", "")
            tv = tradingview_url(t, ex)

            pnl_num = None if r.get("pnl_pct_is_na") else safe_float(r.get("pnl_pct_num"))
            pc = pnl_class(pnl_num)

            tech = tech_cache.get(t, {})
            compact = tech.get("compact") or ""
            warn = bool(tech.get("warn"))
            icon = "⚠️ " if warn else ""
            techline = f"<div class='muted techline'>{icon}{html_escape(compact)}</div>" if compact else ""

            out.append(
                "<tr>"
                "<td>"
                f"<a href='/t/{html_escape(t)}' target='_blank' rel='noopener noreferrer'>{html_escape(t)}</a>"
                f"<sup style='margin-left:6px;'>"
                f"<a href='{html_escape(tv)}' target='_blank' rel='noopener noreferrer' "
                f"title='Open in TradingView' style='font-size:11px; text-decoration:none;'>tv</a>"
                f"</sup>"
                f"{techline}"
                "</td>"
                f"<td>{html_escape(str(r.get('lots','')))}</td>"
                f"<td>{html_escape(r.get('invested_disp',''))}</td>"
                f"<td>{html_escape(r.get('weight_pct',''))}</td>"
                f"<td class='{pc}'>{html_escape(r.get('pnl_pct',''))}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
        return "".join(out)

    def render_scan_top(scan_obj: Dict[str, Any]) -> str:
        top = scan_obj.get("top") or []
        if not top:
            return "<div class='muted'>No scan yet. Run <code>/tasks/scan-energy</code>.</div>"
        out = []
        out.append("<table><thead><tr>"
                   "<th>Ticker</th><th>Bucket</th><th>Score</th><th>Fundamentals</th><th>Tech</th>"
                   "</tr></thead><tbody>")
        for r in top[:25]:
            t = r.get("ticker")
            bucket = r.get("bucket")
            score = r.get("score")
            fund = r.get("fund") or {}
            tech = r.get("tech") or {}
            fline = fund.get("compact") or fund.get("note") or "Fund NA"
            tline = tech.get("compact") or "Tech NA"
            out.append(
                "<tr>"
                f"<td><a href='/t/{html_escape(t)}' target='_blank' rel='noopener noreferrer'>{html_escape(t)}</a></td>"
                f"<td>{html_escape(bucket)}</td>"
                f"<td>{html_escape(str(score))}</td>"
                f"<td class='muted' style='font-size:12px;'>{html_escape(fline)}</td>"
                f"<td class='muted' style='font-size:12px;'>{html_escape(tline)}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
        return "".join(out)

    scan_date = scan.get("date") or "never"
    scan_sources = ", ".join(scan.get("sources") or [])
    scan_warn = scan.get("warnings") or []
    warn_html = ""
    if scan_warn:
        warn_html = "<div class='muted' style='margin-top:6px; font-size:12px;'>" + html_escape(" | ".join(scan_warn[:3])) + "</div>"

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>My AI Investing Agent</title>
        <style>
          body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
            margin: 24px;
            background: #ffffff;
            color: #000000;
          }}
          a {{ color: #0b57d0; text-decoration:none; }}
          a:hover {{ text-decoration:underline; }}
          .muted {{ color:#555; }}
          .wrap {{ max-width: 1250px; }}
          .topline {{ display:flex; gap:18px; flex-wrap:wrap; align-items:baseline; }}
          code {{ background:#f3f3f3; border:1px solid #ddd; padding:2px 6px; border-radius:8px; }}

          table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; vertical-align: top; }}
          th {{ background: #f5f5f5; text-align: left; }}
          th a {{ color:#111; }}

          .section {{
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 10px 12px;
            margin: 12px 0;
            background: #fff;
          }}
          .pos {{ color: #0a7a2f; font-weight: 700; }}
          .neg {{ color: #b00020; font-weight: 700; }}
          .flat {{ color: #111; font-weight: 700; }}
          .na {{ color: #555; }}

          .techline {{ font-size:12px; margin-top:3px; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <h1>My AI Investing Agent</h1>

          <div class="topline">
            <div><b>Last portfolio update:</b> {html_escape(str(last_update))}</div>
            <div><b>Lots:</b> {html_escape(str(stats.get("lots_count","")))} · <b>Unique:</b> {html_escape(str(stats.get("unique_instruments_count","")))}</div>
            <div><b>Mapping:</b> {html_escape(str(mapping.get("mapped","")))} / {html_escape(str(mapping.get("requested","")))}</div>
            <div>Refresh portfolio: <code>/tasks/daily</code></div>
            <div>Scan energy: <code>/tasks/scan-energy</code></div>
            <div><a href="/scanner" target="_self">Open Scanner</a></div>
            <div class="muted"><a href="/debug/env" target="_self">env</a> · <a href="/debug/last-error" target="_self">last-error</a></div>
          </div>

          <div class="section">
            <div style="font-weight:800; font-size:16px;">Energy Scanner (Top 25)</div>
            <div class="muted" style="margin-top:4px;">
              Last scan: {html_escape(scan_date)} · Sources: {html_escape(scan_sources)}
            </div>
            {warn_html}
            <div style="margin-top:10px;">{render_scan_top(scan)}</div>
          </div>

          <div class="section">
            <div style="font-weight:800; font-size:16px;">Portfolio</div>
            <div class="muted" style="margin-top:4px;">Sort with headers · green=positive, red=negative.</div>
            <div style="margin-top:10px;">{render_portfolio(positions)}</div>
          </div>

          <p class="muted">Click ticker for links. Click <b>tv</b> for TradingView.</p>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/scanner", response_class=HTMLResponse)
async def scanner():
    state = load_state()
    scan = state.get("energy_scan") or {}
    rows = scan.get("top") or []
    date = scan.get("date") or "never"
    sources = ", ".join(scan.get("sources") or [])
    uni_count = scan.get("universe_count") or 0
    scanned = scan.get("scanned") or 0
    warnings = scan.get("warnings") or []

    if not rows:
        body = "<div class='muted'>No scan yet. Run <code>/tasks/scan-energy</code>.</div>"
    else:
        out = []
        out.append("<table><thead><tr>"
                   "<th>Ticker</th><th>Bucket</th><th>Score</th><th>Tags</th><th>Fundamentals</th><th>Tech</th>"
                   "</tr></thead><tbody>")
        for r in rows[:200]:
            t = r.get("ticker")
            bucket = r.get("bucket")
            score = r.get("score")
            tags = ", ".join(r.get("tags") or [])
            fund = r.get("fund") or {}
            tech = r.get("tech") or {}
            fline = fund.get("compact") or fund.get("note") or "Fund NA"
            tline = tech.get("compact") or "Tech NA"
            out.append(
                "<tr>"
                f"<td><a href='/t/{html_escape(t)}' target='_blank' rel='noopener noreferrer'>{html_escape(t)}</a></td>"
                f"<td>{html_escape(bucket)}</td>"
                f"<td>{html_escape(str(score))}</td>"
                f"<td class='muted' style='font-size:12px;'>{html_escape(tags)}</td>"
                f"<td class='muted' style='font-size:12px;'>{html_escape(fline)}</td>"
                f"<td class='muted' style='font-size:12px;'>{html_escape(tline)}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
        body = "".join(out)

    warn_html = ""
    if warnings:
        warn_html = "<div class='muted' style='margin-top:8px; font-size:12px;'>" + html_escape(" | ".join(warnings[:5])) + "</div>"

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Energy Scanner</title>
        <style>
          body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
            margin: 24px;
            background: #ffffff;
            color: #000000;
          }}
          a {{ color: #0b57d0; text-decoration:none; }}
          a:hover {{ text-decoration:underline; }}
          .muted {{ color:#555; }}
          code {{ background:#f3f3f3; border:1px solid #ddd; padding:2px 6px; border-radius:8px; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; vertical-align: top; }}
          th {{ background: #f5f5f5; text-align: left; }}
          .top {{ display:flex; gap:12px; flex-wrap:wrap; align-items:baseline; }}
          .btn {{
            display:inline-block; padding:6px 10px; border:1px solid #ddd;
            border-radius:10px; background:#f5f5f5; color:#111; font-size:13px;
          }}
        </style>
      </head>
      <body>
        <div class="top">
          <h1 style="margin:0;">Energy Scanner</h1>
          <a class="btn" href="/" target="_self">Back</a>
          <span class="muted">Last scan: {html_escape(date)} · Sources: {html_escape(sources)} · Universe: {uni_count} · Scanned: {scanned}</span>
          <span class="muted">Run: <code>/tasks/scan-energy</code></span>
        </div>
        {warn_html}
        {body}
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/t/{ticker}", response_class=HTMLResponse)
async def ticker_page(ticker: str):
    state = load_state()
    t = normalize_ticker(ticker)
    news_cache = state.get("news_cache") or {}
    news_items = (news_cache.get(t) or [])[:NEWS_PER_TICKER]
    tv = tradingview_url(t, "")

    def render_news():
        if not news_items:
            return "<div class='muted'>No cached news for this ticker (news is cached for portfolio tickers only).</div>"
        out = ["<ul>"]
        for it in news_items:
            out.append(
                "<li>"
                f"<a href='{html_escape(it.get('link',''))}' target='_blank' rel='noopener noreferrer'>{html_escape(it.get('title',''))}</a>"
                f"<div class='muted' style='font-size:12px;'>{html_escape(it.get('source',''))} · {html_escape(it.get('published',''))}</div>"
                f"<div style='font-size:12px;'>{html_escape(it.get('resume',''))}</div>"
                "</li>"
            )
        out.append("</ul>")
        return "".join(out)

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{html_escape(t)} — Links</title>
        <style>
          body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
            margin: 24px;
            background: #ffffff;
            color: #000000;
          }}
          a {{ color: #0b57d0; text-decoration:none; }}
          a:hover {{ text-decoration:underline; }}
          .muted {{ color:#555; }}
          .btn {{
            display:inline-block; padding:6px 10px; border:1px solid #ddd;
            border-radius:10px; background:#f5f5f5; color:#111; font-size:13px;
          }}
          .card {{ border:1px solid #ddd; border-radius:12px; padding:12px; margin:12px 0; }}
        </style>
      </head>
      <body>
        <div>
          <h1 style="margin:0;">{html_escape(t)}</h1>
          <div style="margin-top:10px;">
            <a class="btn" href="/" target="_self">Back</a>
            <a class="btn" href="{html_escape(tv)}" target="_blank" rel="noopener noreferrer">TradingView</a>
            <a class="btn" href="{html_escape(edgar_search_link(t))}" target="_blank" rel="noopener noreferrer">EDGAR</a>
          </div>
        </div>
        <div class="card">
          <div style="font-weight:800; font-size:13px; text-transform:uppercase; letter-spacing:.08em;">News</div>
          {render_news()}
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/tasks/daily")
async def task_daily():
    state = load_state()
    state["date"] = utc_now_iso()

    # eToro pull
    payload = await etoro_get_real_pnl()
    raw_positions = extract_positions(payload)
    agg_positions, stats = aggregate_positions_by_instrument(raw_positions)

    instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]
    ticker_map, map_debug = await map_instrument_ids_to_tickers(instrument_ids)

    portfolio_rows = build_portfolio_rows(agg_positions, ticker_map)
    tickers = sorted({r["ticker"] for r in portfolio_rows if r.get("ticker")})

    # Tech cache (never crash the whole task)
    try:
        tech_cache = await compute_tech_cache(tickers)
    except Exception as e:
        tech_cache = {}
        state["daily_warning_tech"] = repr(e)

    # News cache (never crash the whole task)
    try:
        news_cache = await compute_news(tickers)
    except Exception as e:
        news_cache = {}
        state["daily_warning_news"] = repr(e)

    state.update({
        "positions": portfolio_rows,
        "stats": stats,
        "mapping": {"requested": map_debug.get("requested"), "mapped": map_debug.get("mapped")},
        "mapping_last_debug": map_debug,
        "instrument_meta": map_debug.get("instrument_meta") or {},
        "tech_cache": tech_cache,
        "news_cache": news_cache,
        "energy_thesis": ENERGY_THESIS,
    })
    save_state(state)

    return {
        "status": "ok",
        "lots": stats["lots_count"],
        "unique_instruments": stats["unique_instruments_count"],
        "mapped_symbols": map_debug.get("mapped"),
        "portfolio_tickers": len(tickers),
    }

@app.get("/tasks/scan-energy")
async def task_scan_energy():
    state = load_state()
    res = await run_energy_scan(state)
    return {
        "status": "ok",
        "date": res.get("date"),
        "universe_count": res.get("universe_count"),
        "scanned": res.get("scanned"),
        "sources": res.get("sources"),
        "warnings": res.get("warnings"),
        "top_count": len(res.get("top") or []),
        "note": "Best fundamentals for US SEC filers; set SEC_UA for full SEC access.",
    }

@app.get("/api/scanner")
async def api_scanner():
    state = load_state()
    return JSONResponse(state.get("energy_scan") or {"note": "Run /tasks/scan-energy"})

@app.get("/api/portfolio")
async def api_portfolio():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "stats": state.get("stats") or {},
        "positions": state.get("positions") or [],
        "mapping": state.get("mapping") or {},
        "warnings": {
            "daily_warning_tech": state.get("daily_warning_tech"),
            "daily_warning_news": state.get("daily_warning_news"),
        }
    })
