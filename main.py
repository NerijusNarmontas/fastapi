# main.py
# Single-file FastAPI app that:
# - Pulls eToro REAL PnL + portfolio positions
# - Aggregates by instrumentID
# - Maps instrumentID -> ticker via eToro /market-data/search
# - Separates Crypto vs Stocks (crypto list includes your tickers)
# - Categorizes stocks into industry-like groups (manual mappings included)
# - CLEAN dashboard (white background / black text)
# - Adds TECH line next to each ticker:
#     "1M -12% ⚠️ | Trend below 200DMA | RSI 28 | Liq OK"
# - Negative values shown in RED, positive in GREEN
# - Adds "Invested $" column (so you know how much you invested per ticker)
# - Sorting: click table headers (or use ?sort=...&dir=...)
# - News/SEC links shown ONLY when you click a ticker:
#     /t/{TICKER} opens a page with cached News + SEC links
# - Adds tiny superscript "tv" link next to each ticker to open TradingView live
# - Fetches FREE news (Google News RSS) + SEC links (cached by /tasks/daily)
#
# Railway startCommand:
#   hypercorn main:app --bind "0.0.0.0:$PORT"
#
# Railway Variables REQUIRED:
#   ETORO_API_KEY=...
#   ETORO_USER_KEY=...
#   SEC_UA=YourName AppName (email@domain.com)
#
# Optional:
#   OPENAI_API_KEY=... (optional AI brief; safe to leave empty)
#   CRYPTO_EXCLUDE=W
#   CRYPTO_TICKERS=... (extend list)
#   CRYPTO_INSTRUMENT_IDS=100000
#   STATE_PATH=/tmp/investing_agent_state.json
#   NEWS_PER_TICKER=6
#   SEC_PER_TICKER=6
#   DEFAULT_UA=...
#   ETORO_REAL_PNL_URL=...
#   ETORO_SEARCH_URL=...
#   CIK_MAP={"AAPL":"0000320193"}  # optional for richer SEC lists (CIK-based Atom feed)

import os
import re
import json
import uuid
import math
import asyncio
from dataclasses import dataclass
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
# Config (ENV VARS on Railway)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

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
CIK_MAP_JSON = os.getenv("CIK_MAP", "").strip()

# kept: ticker collision excludes (default W)
CRYPTO_EXCLUDE = set(
    s.strip().upper() for s in os.getenv("CRYPTO_EXCLUDE", "W").split(",") if s.strip()
)

# crypto list (includes your crypto tickers; extendable via env)
CRYPTO_TICKERS = set(
    s.strip().upper()
    for s in os.getenv(
        "CRYPTO_TICKERS",
        "BTC,ETH,SOL,AVAX,OP,ARB,JTO,RUNE,W,EIGEN,STRK,ONDO,"
        "SEI,WLD,PYTH,HBAR,CRO,HYPE,RPL,ARBE",
    ).split(",")
    if s.strip()
)

# optional instrumentID heuristic for crypto (BTC on eToro often 100000)
CRYPTO_INSTRUMENT_IDS = set(
    int(x.strip())
    for x in os.getenv("CRYPTO_INSTRUMENT_IDS", "100000").split(",")
    if x.strip().isdigit()
)

NEWS_PER_TICKER = int(os.getenv("NEWS_PER_TICKER", "6"))
SEC_PER_TICKER = int(os.getenv("SEC_PER_TICKER", "6"))

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))

# ----------------------------
# Industry-like categories (manual overrides first)
# ----------------------------
CATEGORY_OVERRIDES: Dict[str, Tuple[str, str]] = {
    # Your explicit mapping
    "CEG": ("Energy", "Power / Nuclear-linked"),
    "BKSY": ("Space", "Earth Observation"),
    "FLY": ("Space", "Aerospace"),
    "SPIR": ("Space Tech", "Satcom / Data"),
    "ABAT": ("Rare Metals", "Battery / Materials"),
    "USAR": ("Rare Metals", "Rare Earth / Materials"),
    "TMC": ("Rare Metals", "Seabed Metals"),
    "FERMI": ("Nuclear", "Nuclear"),
    "LTBR": ("Nuclear", "Nuclear Components"),
    "BOE.ASX": ("Nuclear", "Nuclear (ASX)"),
    "CRWV": ("Tech", "Software / Infra"),
    "S.US": ("Tech", "Semis / Chips"),
    "INTC": ("Tech", "Semis / Chips"),
    "AR.US": ("Gas", "Gas E&P"),
    "RGLD": ("Miners", "Gold / Royalty"),
    "ARQQ": ("Quantum", "Quantum / Security"),
    "RGTI": ("Quantum", "Quantum Computing"),

    # Energy / gas / midstream examples
    "EQT": ("Gas", "Gas E&P"),
    "CTRA": ("Gas", "Gas E&P"),
    "RRC": ("Gas", "Gas E&P"),
    "DVN": ("Energy", "Oil & Gas E&P"),
    "EOG": ("Energy", "Oil & Gas E&P"),
    "FANG": ("Energy", "Oil & Gas E&P"),
    "OXY": ("Energy", "Oil & Gas E&P"),
    "EPD": ("Midstream", "Pipes / Terminals"),
    "WMB": ("Midstream", "Pipes / Terminals"),
    "TRGP": ("Midstream", "Pipes / Processing"),
    "ET": ("Midstream", "Pipes / Terminals"),
    "WES": ("Midstream", "Midstream"),

    "CCJ": ("Nuclear", "Uranium"),
    "UEC": ("Nuclear", "Uranium"),
    "LEU": ("Nuclear", "Nuclear Fuel"),
    "SMR": ("Nuclear", "SMR"),
    "OKLO": ("Nuclear", "SMR"),
}

ENERGY_THESIS = {
    "focus": ["post-shale peak", "capital discipline", "FCF/ROIC", "gas tightness", "geopolitics", "nuclear buildout"],
    "avoid": ["refining"],
    "preferred": ["gas", "midstream", "uranium", "nuclear fuel", "smrs"],
}

CATEGORY_ORDER = [
    "Gas",
    "Energy",
    "Midstream",
    "Nuclear",
    "Quantum",
    "Space",
    "Space Tech",
    "Tech",
    "Rare Metals",
    "Miners",
    "Other",
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


def pick_unrealized_pnl(p: Dict[str, Any]) -> Optional[float]:
    up = p.get("unrealizedPnL") or p.get("unrealizedPnl") or p.get("unrealized_pnl")
    if isinstance(up, dict):
        return normalize_number(up.get("pnL"))
    return normalize_number(up)


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
    stats = {"lots_count": len(raw_positions), "unique_instruments_count": len(aggregated)}
    return aggregated, stats


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


def is_crypto_ticker(ticker: str, instrument_id: Optional[int] = None) -> bool:
    t = normalize_ticker(ticker)
    if t in CRYPTO_TICKERS:
        return True
    if instrument_id is not None and instrument_id in CRYPTO_INSTRUMENT_IDS:
        return True
    if t.endswith("-USD") and t[:-4] in CRYPTO_TICKERS:
        return True
    return False


def categorize_ticker(ticker: str) -> Tuple[str, str]:
    t = normalize_ticker(ticker)
    if t in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[t]

    # conservative fallbacks
    if t in ("CCJ", "UEC", "LEU", "SMR", "OKLO", "LTBR", "FERMI", "BOE.ASX"):
        return ("Nuclear", "Nuclear")
    if t in ("ARQQ", "RGTI", "IONQ", "QBTS"):
        return ("Quantum", "Quantum")
    if t in ("BKSY", "SPIR", "RDW", "FLY"):
        return ("Space", "Space")
    if t in ("RGLD", "AU", "NEM", "KGC", "SBSW", "MP"):
        return ("Miners", "Miners")
    if t in ("ABAT", "USAR", "TMC", "ALB"):
        return ("Rare Metals", "Materials")
    if t in ("INTC", "S.US", "CRWV"):
        return ("Tech", "Tech")
    if t.endswith(".US") and t.startswith("AR"):
        return ("Gas", "Gas")
    return ("Other", "Unclassified")


def ordered_industries(groups: Dict[str, Any]) -> List[str]:
    present = set(groups.keys())
    out = [x for x in CATEGORY_ORDER if x in present]
    for k in sorted(present):
        if k not in out:
            out.append(k)
    return out


def group_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      groups: { industry: {"rows":[...], "tickers":[...]} }
      crypto_rows: [rows]
    """
    groups: Dict[str, Dict[str, Any]] = {}
    crypto_rows: List[Dict[str, Any]] = []

    for r in rows:
        t = r.get("ticker", "")
        iid = None
        try:
            iid = int(r.get("instrumentID")) if r.get("instrumentID") is not None else None
        except Exception:
            iid = None

        if is_crypto_ticker(t, iid):
            crypto_rows.append(r)
            continue

        industry, _sub = categorize_ticker(t)
        if industry not in groups:
            groups[industry] = {"rows": [], "tickers": []}
        groups[industry]["rows"].append(r)
        if t:
            groups[industry]["tickers"].append(t)

    for industry in groups:
        groups[industry]["rows"].sort(key=lambda x: float(x.get("weight_pct_num") or 0), reverse=True)
        groups[industry]["tickers"] = list(dict.fromkeys(groups[industry]["tickers"]))

    crypto_rows.sort(key=lambda x: float(x.get("weight_pct_num") or 0), reverse=True)
    return groups, crypto_rows


def tradingview_url(ticker: str, exchange: str = "") -> str:
    t = normalize_ticker(ticker)
    ex = (exchange or "").strip().upper()
    if ex:
        return f"https://www.tradingview.com/chart/?symbol={ex}:{t}"
    return f"https://www.tradingview.com/symbols/{t}/"


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


# ----------------------------
# TECH METRICS (Stooq)
# ----------------------------
def _stooq_symbol_candidates(ticker: str) -> List[str]:
    t = normalize_ticker(ticker)
    if t.endswith(".US"):
        base = t[:-3]
        return [f"{base}.US", base, t]
    return [t, t.replace("-", "."), t.replace(".", "-")]


async def fetch_stooq_daily(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, float]]:
    """
    Returns list of bars oldest->newest with keys: close, high, low, volume
    """
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
                h = float(parts[2])
                l = float(parts[3])
                c = float(parts[4])
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
    gains = []
    losses = []
    for i in range(-n, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0:
            gains.append(ch)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-ch)
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def monthly_return_pct(closes: List[float], lookback_days: int = 21) -> Optional[float]:
    if len(closes) < lookback_days + 1:
        return None
    a = closes[-lookback_days - 1]
    b = closes[-1]
    if a == 0:
        return None
    return (b / a - 1.0) * 100.0


def liquidity_flag(bars: List[Dict[str, float]], days: int = 20) -> Tuple[str, Optional[float]]:
    """
    Returns ("OK"/"LOW"/"NA", avg_dollar_vol)
    """
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

    warn = False
    if rsi is not None and rsi <= 30:
        warn = True
    if liq == "LOW":
        warn = True

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
        "mret_1m_pct": mret,
        "sma200": sma200,
        "trend_vs_200": trend,
        "rsi14": rsi,
        "liq": liq,
        "avg_dollar_vol": avg_dv,
        "compact": " | ".join(parts),
        "warn": warn,
    }


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
    """
    instrumentId -> ticker, and best-effort meta (exchange) for TradingView links.
    Returns: ticker_map, debug (includes debug["instrument_meta"]).
    """
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


# ----------------------------
# Portfolio rows + PnL% + Invested$
# ----------------------------
def build_portfolio_rows(agg: List[Dict[str, Any]], ticker_map: Dict[int, str]) -> List[Dict[str, Any]]:
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0
    rows: List[Dict[str, Any]] = []

    for a in agg:
        iid = int(a["instrumentID"])
        ticker_raw = ticker_map.get(iid) or str(iid)
        ticker = normalize_ticker(ticker_raw)

        # collision exclude for non-crypto only
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

            # numeric fields for sorting
            "invested_usd": float(initial or 0.0),
            "unreal_usd": float(unreal or 0.0),
            "weight_pct_num": float(weight_pct or 0.0),
            "pnl_pct_num": float(pnl_pct or 0.0) if pnl_pct is not None else 0.0,
            "pnl_pct_is_na": pnl_pct is None,

            # display fields
            "invested_disp": fmt_money(float(initial or 0.0)),
            "weight_pct": f"{weight_pct:.2f}" if weight_pct is not None else "",
            "pnl_pct": f"{pnl_pct:.2f}" if pnl_pct is not None else "",
        })

    return rows


# ----------------------------
# News (Google News RSS)
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
    if any(k in t for k in ["earnings", "guidance", "revenue", "eps"]):
        tags.append("Earnings / guidance")
    if any(k in t for k in ["acquire", "acquisition", "merger", "buyout", "takeover"]):
        tags.append("M&A")
    if any(k in t for k in ["offering", "private placement", "convertible", "debt", "notes", "atm", "shelf"]):
        tags.append("Financing")
    if any(k in t for k in ["contract", "award", "partnership", "collaboration"]):
        tags.append("Commercial deal")
    if any(k in t for k in ["lawsuit", "probe", "investigation", "fraud", "settlement"]):
        tags.append("Legal / investigation")
    if any(k in t for k in ["upgrade", "downgrade", "price target", "analyst"]):
        tags.append("Analyst move")
    if any(k in t for k in ["sec", "8-k", "10-q", "10-k", "s-1", "f-1", "edgar"]):
        tags.append("Filing-related")
    return " • ".join(tags) if tags else "Headline update"


async def fetch_google_news_for_ticker(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, Any]]:
    q = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = await client.get(
        url,
        headers={
            "accept": "application/rss+xml, text/xml;q=0.9,*/*;q=0.8",
            "user-agent": DEFAULT_UA,
        },
    )
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


# ----------------------------
# SEC (links + short resumes)
# ----------------------------
@dataclass
class SecItem:
    ticker: str
    form: str
    title: str
    link: str
    filed: str
    resume: str


def resume_sec(form: str) -> str:
    f = (form or "").upper().strip()
    if f == "8-K":
        return "Material event."
    if f == "10-Q":
        return "Quarterly report."
    if f == "10-K":
        return "Annual report."
    if f in ("S-1", "F-1", "424B", "424B2", "424B3", "424B5"):
        return "Registration / offering."
    return "Filing update."


def parse_atom_entries(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    ns = "{http://www.w3.org/2005/Atom}"
    out = []
    for entry in root.findall(f".//{ns}entry"):
        title = (entry.findtext(f"{ns}title") or "").strip()
        updated = (entry.findtext(f"{ns}updated") or "").strip()
        link_el = entry.find(f"{ns}link")
        link = link_el.attrib.get("href", "").strip() if link_el is not None else ""
        out.append({"title": title, "updated": updated, "link": link})
    return out


async def fetch_sec_for_ticker(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, Any]]:
    cik_map: Dict[str, str] = {}
    if CIK_MAP_JSON:
        try:
            cik_map = json.loads(CIK_MAP_JSON)
        except Exception:
            cik_map = {}

    cik = (cik_map.get(ticker) or "").strip()
    if cik:
        cik = cik.zfill(10)
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&owner=exclude&count=40&output=atom"
        r = await client.get(
            url,
            headers={
                "accept": "application/atom+xml, application/xml;q=0.9,*/*;q=0.8",
                "user-agent": SEC_UA or DEFAULT_UA,
            },
        )
        if r.status_code != 200:
            return []

        entries = parse_atom_entries(r.text)
        out: List[Dict[str, Any]] = []
        seen = set()
        for e in entries:
            title = e.get("title") or ""
            link = e.get("link") or ""
            if not title or not link or link in seen:
                continue
            seen.add(link)
            m = re.search(r"\b(8-K|10-Q|10-K|S-1|F-1|424B\d*)\b", title, re.I)
            form = (m.group(1).upper() if m else "Filing")
            out.append({
                "ticker": ticker,
                "form": form,
                "title": title,
                "link": link,
                "filed": e.get("updated") or "",
                "resume": resume_sec(form),
            })
            if len(out) >= SEC_PER_TICKER:
                break
        return out

    search_link = f"https://www.sec.gov/edgar/search/#/q={quote_plus(ticker)}&sort=desc"
    return [{
        "ticker": ticker,
        "form": "EDGAR",
        "title": f"EDGAR search for {ticker}",
        "link": search_link,
        "filed": "",
        "resume": "Open EDGAR search.",
    }]


# ----------------------------
# Optional OpenAI brief (disable by leaving OPENAI_API_KEY empty)
# ----------------------------
async def generate_openai_brief(portfolio_rows: List[Dict[str, Any]]) -> str:
    if not OPENAI_API_KEY:
        return "AI brief disabled (no key)."

    top = portfolio_rows[:20]
    lines = [f"{r['ticker']}: invested=${r.get('invested_disp','')} w={r.get('weight_pct','')}% pnl={r.get('pnl_pct','')}%" for r in top]
    portfolio_text = "\n".join(lines) if lines else "(no positions)"

    prompt = (
        "You are an investing assistant. READ-ONLY.\n"
        "No buy/sell calls. No price predictions.\n"
        "Write 6-10 bullets only: material events, risks, watchlist.\n\n"
        f"Portfolio:\n{portfolio_text}\n"
    )

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": "gpt-5-mini", "input": prompt, "max_output_tokens": 350}

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=body)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"text": r.text}
            # show clean message (your screenshot showed 429 insufficient_quota)
            return f"OpenAI error {r.status_code}: {err}"
        data = r.json()
        return data.get("output_text") or "AI brief generated (output_text missing)."


# ----------------------------
# Daily Task: tickers -> tech + news + sec
# ----------------------------
async def compute_news_and_sec(tickers: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sem = asyncio.Semaphore(CONCURRENCY)
    news_cache: Dict[str, Any] = {}
    sec_cache: Dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:

        async def do_news(t: str):
            async with sem:
                news_cache[t] = await fetch_google_news_for_ticker(client, t)

        async def do_sec(t: str):
            async with sem:
                sec_cache[t] = await fetch_sec_for_ticker(client, t)

        await asyncio.gather(*(do_news(t) for t in tickers))
        await asyncio.gather(*(do_sec(t) for t in tickers))

    return news_cache, sec_cache


async def compute_tech_cache(tickers: List[str]) -> Dict[str, Any]:
    sem = asyncio.Semaphore(CONCURRENCY)
    tech_cache: Dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:

        async def do_one(t: str):
            async with sem:
                bars = await fetch_stooq_daily(client, t)
                if bars:
                    tech_cache[t] = build_compact_line(t, bars)
                else:
                    tech_cache[t] = {"ticker": t, "compact": "Tech NA (no data)", "warn": False}

        await asyncio.gather(*(do_one(t) for t in tickers))

    return tech_cache


# ----------------------------
# Sorting helpers
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


@app.get("/", response_class=HTMLResponse)
async def dashboard(sort: str = "weight", dir: str = "desc"):
    state = load_state()

    last_update = state.get("date") or utc_now_iso()
    material_events = state.get("material_events") or []
    technical_exceptions = state.get("technical_exceptions") or []
    action_required = state.get("action_required") or ["Run /tasks/daily once to refresh data."]

    stats = state.get("stats") or {}
    ai_brief = state.get("ai_brief") or ""

    mapping = state.get("mapping") or {}
    mapped = mapping.get("mapped", "")
    requested = mapping.get("requested", "")

    groups = state.get("groups") or {}
    crypto_rows = state.get("crypto_rows") or []
    instrument_meta = state.get("instrument_meta") or {}  # {instrumentID(str): {"exchange": "..."}}
    tech_cache = state.get("tech_cache") or {}

    sort_by = (sort or "weight").lower()
    direction = (dir or "desc").lower()
    if sort_by not in SORT_KEYS:
        sort_by = "weight"
    if direction not in ("asc", "desc"):
        direction = "desc"

    def bullets(items: List[str]) -> str:
        lis = "".join([f"<li>{html_escape(x)}</li>" for x in items]) if items else "<li>None</li>"
        return f"<ul>{lis}</ul>"

    def hdr(label: str, key: str) -> str:
        next_dir = "asc" if (sort_by == key and direction == "desc") else "desc"
        arrow = ""
        if sort_by == key:
            arrow = " ▲" if direction == "asc" else " ▼"
        return f"<a href='/?sort={html_escape(key)}&dir={html_escape(next_dir)}'>{html_escape(label)}{arrow}</a>"

    def render_table(rows: List[Dict[str, Any]], is_crypto: bool = False) -> str:
        if not rows:
            return "<div class='muted'>No positions.</div>"

        rows_sorted = sort_rows(rows, sort_by, direction)

        out = []
        out.append("<table><thead><tr>"
                   f"<th>{hdr('Ticker','ticker')}</th>"
                   f"<th>{hdr('Lots','lots')}</th>"
                   f"<th>{hdr('Invested $','invested')}</th>"
                   f"<th>{hdr('Weight %','weight')}</th>"
                   f"<th>{hdr('P&L %','pnl')}</th>"
                   "</tr></thead><tbody>")

        for r in rows_sorted[:300]:
            t = r.get("ticker", "")
            iid = (r.get("instrumentID") or "0")
            ex = (instrument_meta.get(str(iid)) or {}).get("exchange", "")
            tv = tradingview_url(t, ex)

            pnl_num = None if r.get("pnl_pct_is_na") else safe_float(r.get("pnl_pct_num"))
            pnl_class = "na"
            if pnl_num is not None:
                pnl_class = "pos" if pnl_num > 0 else ("neg" if pnl_num < 0 else "flat")

            tech = tech_cache.get(t, {})
            compact = tech.get("compact") or ""
            warn = bool(tech.get("warn"))
            mret = tech.get("mret_1m_pct", None)

            # Style 1M return red/green inside the compact line (only for that part)
            compact_html = ""
            if compact:
                icon = "⚠️ " if warn else ""
                if mret is not None:
                    mret_class = "pos" if mret > 0 else ("neg" if mret < 0 else "flat")
                    # Replace first "1M ..." part by colored span
                    # Best-effort: find prefix until first "|"
                    first = compact.split("|")[0].strip()
                    rest = " | ".join([p.strip() for p in compact.split("|")[1:]]) if "|" in compact else ""
                    colored_first = f"<span class='{mret_class}'>{html_escape(first)}</span>"
                    compact_html = f"<div class='muted techline'>{icon}{colored_first}"
                    if rest:
                        compact_html += f" <span class='muted'>| {html_escape(rest)}</span>"
                    compact_html += "</div>"
                else:
                    compact_html = f"<div class='muted techline'>{icon}{html_escape(compact)}</div>"

            ticker_link = f"<a href='/t/{html_escape(t)}' target='_blank' rel='noopener noreferrer'>{html_escape(t)}</a>"
            if is_crypto:
                # crypto page still works for tv + cached links page (news/sec may be empty)
                pass

            out.append(
                "<tr>"
                "<td>"
                f"{ticker_link}"
                f"<sup style='margin-left:6px;'>"
                f"<a href='{html_escape(tv)}' target='_blank' rel='noopener noreferrer' "
                f"title='Open in TradingView' style='font-size:11px; text-decoration:none;'>tv</a>"
                f"</sup>"
                f"{compact_html}"
                "</td>"
                f"<td>{html_escape(str(r.get('lots','')))}</td>"
                f"<td>{html_escape(r.get('invested_disp',''))}</td>"
                f"<td>{html_escape(r.get('weight_pct',''))}</td>"
                f"<td class='{pnl_class}'>{html_escape(r.get('pnl_pct',''))}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
        return "".join(out)

    # Build "All Stocks" list (ticker + invested) for quick overview
    all_stock_rows: List[Dict[str, Any]] = []
    for industry in groups:
        all_stock_rows.extend((groups.get(industry) or {}).get("rows") or [])
    all_stock_rows = sort_rows(all_stock_rows, "invested", "desc")

    all_stocks_html = ""
    if all_stock_rows:
        out = []
        out.append("<table><thead><tr>"
                   "<th>Ticker</th><th>Invested $</th><th>Weight %</th>"
                   "</tr></thead><tbody>")
        for r in all_stock_rows[:400]:
            t = r.get("ticker", "")
            out.append(
                "<tr>"
                f"<td>{html_escape(t)}</td>"
                f"<td>{html_escape(r.get('invested_disp',''))}</td>"
                f"<td>{html_escape(r.get('weight_pct',''))}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
        all_stocks_html = "".join(out)

    category_html = ""
    for industry in ordered_industries(groups):
        rows = (groups.get(industry) or {}).get("rows") or []
        category_html += f"""
        <details class="section" open>
          <summary>{html_escape(industry)} <span class="muted">({len(rows)} positions)</span></summary>
          <div class="block">{render_table(rows, is_crypto=False)}</div>
        </details>
        """

    crypto_html = ""
    if crypto_rows:
        crypto_html = f"""
        <details class="section">
          <summary>Crypto <span class="muted">({len(crypto_rows)} positions)</span></summary>
          <div class="block">{render_table(crypto_rows, is_crypto=True)}</div>
          <div class="muted">Crypto news/SEC not fetched by default. Click ticker for links page, and “tv” for TradingView.</div>
        </details>
        """

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
          .wrap {{ max-width: 1200px; }}

          .topline {{ display:flex; gap:18px; flex-wrap:wrap; align-items:baseline; }}
          code {{ background:#f3f3f3; border:1px solid #ddd; padding:2px 6px; border-radius:8px; }}

          table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; vertical-align: top; }}
          th {{ background: #f5f5f5; text-align: left; }}
          th a {{ color:#111; }}
          th a:hover {{ text-decoration:underline; }}

          .section {{
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 10px 12px;
            margin: 12px 0;
            background: #fff;
          }}
          summary {{
            cursor: pointer;
            font-weight: 700;
            font-size: 16px;
          }}
          .block {{ margin-top: 10px; }}

          pre {{
            white-space: pre-wrap;
            background:#f7f7f7;
            border:1px solid #ddd;
            padding:12px;
            border-radius:12px;
          }}

          .pos {{ color: #0a7a2f; font-weight: 700; }}
          .neg {{ color: #b00020; font-weight: 700; }}
          .flat {{ color: #111; font-weight: 700; }}
          .na {{ color: #555; }}

          .techline {{ font-size:12px; margin-top:3px; }}
          .grid {{
            display:grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 14px;
          }}
          @media (max-width: 980px) {{
            .grid {{ grid-template-columns: 1fr; }}
          }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <h1>My AI Investing Agent</h1>
          <div class="topline">
            <div><b>Last update:</b> {html_escape(last_update)}</div>
            <div><b>Lots:</b> {stats.get("lots_count","")} · <b>Unique instruments:</b> {stats.get("unique_instruments_count","")}</div>
            <div><b>Mapping:</b> {mapped}/{requested}</div>
            <div>Refresh: <code>/tasks/daily</code></div>
            <div class="muted">Sort: <code>?sort=ticker|invested|weight|pnl|lots&dir=asc|desc</code></div>
          </div>

          <h2>Material Events</h2>
          {bullets(material_events)}

          <h2>Technical Exceptions</h2>
          {bullets(technical_exceptions)}

          <h2>Action Required</h2>
          {bullets(action_required)}

          <h2>AI Brief</h2>
          <pre>{html_escape(ai_brief or "No brief yet. Run /tasks/daily.")}</pre>

          <div class="grid">
            <div>
              {crypto_html}
              <h2>Stocks (industry groups)</h2>
              {category_html if category_html else "<div class='muted'>No categorized stock positions yet. Run /tasks/daily.</div>"}
            </div>
            <div>
              <div class="section" open>
                <div style="font-weight:700; font-size:16px; margin-bottom:6px;">All Stock Tickers (Invested)</div>
                <div class="muted" style="margin-bottom:8px;">Quick view: which tickers you have + how much $ you put in.</div>
                <div class="block">{all_stocks_html if all_stocks_html else "<div class='muted'>No stock positions.</div>"}</div>
              </div>
            </div>
          </div>

          <p class="muted">Click ticker for links page (News + SEC). Click <b>tv</b> to open TradingView.</p>
          <p class="muted">API: <code>/api/portfolio</code> • <code>/api/news</code> • <code>/api/sec</code> • <code>/api/daily-brief</code> • Debug: <code>/debug/mapping-last</code></p>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/t/{ticker}", response_class=HTMLResponse)
async def ticker_page(ticker: str):
    state = load_state()
    t = normalize_ticker(ticker)

    news_cache = state.get("news_cache") or {}
    sec_cache = state.get("sec_cache") or {}

    news_items = (news_cache.get(t) or [])[:NEWS_PER_TICKER]
    sec_items = (sec_cache.get(t) or [])[:SEC_PER_TICKER]

    tv = tradingview_url(t, "")

    def render_news():
        if not news_items:
            return "<div class='muted'>No news cached for this ticker. Run /tasks/daily.</div>"
        out = ["<ul class='lst'>"]
        for it in news_items:
            out.append(
                "<li>"
                f"<a href='{html_escape(it.get('link',''))}' target='_blank' rel='noopener noreferrer'>{html_escape(it.get('title',''))}</a>"
                f"<div class='meta'>{html_escape(it.get('source',''))} · {html_escape(it.get('published',''))}</div>"
                f"<div class='tag'>{html_escape(it.get('resume',''))}</div>"
                "</li>"
            )
        out.append("</ul>")
        return "".join(out)

    def render_sec():
        if not sec_items:
            return "<div class='muted'>No SEC cached for this ticker. (CIK_MAP improves this.)</div>"
        out = ["<ul class='lst'>"]
        for it in sec_items:
            out.append(
                "<li>"
                f"<span class='badge'>{html_escape(it.get('form',''))}</span>"
                f"<a href='{html_escape(it.get('link',''))}' target='_blank' rel='noopener noreferrer'>{html_escape(it.get('title',''))}</a>"
                f"<div class='meta'>{html_escape(it.get('filed',''))}</div>"
                f"<div class='tag'>{html_escape(it.get('resume',''))}</div>"
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
          .wrap {{ max-width: 1000px; }}
          .top {{ display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }}
          .btn {{
            display:inline-block;
            padding:6px 10px;
            border:1px solid #ddd;
            border-radius:10px;
            background:#f5f5f5;
            color:#111;
            font-size:13px;
          }}
          .card {{
            border:1px solid #ddd;
            border-radius:12px;
            padding:12px;
            margin:12px 0;
          }}
          .h {{
            font-weight:800;
            font-size:13px;
            text-transform:uppercase;
            letter-spacing:.08em;
            margin-bottom:8px;
          }}
          ul.lst {{ margin: 8px 0 0 18px; padding:0; }}
          ul.lst li {{ margin: 10px 0; line-height:1.25; }}
          .meta {{ font-size:12px; color:#555; margin-top:2px; }}
          .tag {{ font-size:12px; color:#222; margin-top:2px; }}
          .badge {{
            display:inline-block;
            font-size:11px;
            padding:2px 8px;
            border-radius:999px;
            border:1px solid #ddd;
            background:#f5f5f5;
            margin-right:6px;
          }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="top">
            <h1 style="margin:0;">{html_escape(t)}</h1>
            <a class="btn" href="/" target="_self">Back</a>
            <a class="btn" href="{html_escape(tv)}" target="_blank" rel="noopener noreferrer">Open TradingView</a>
            <span class="muted">Last update: {html_escape(state.get("date") or utc_now_iso())}</span>
          </div>

          <div class="card">
            <div class="h">News</div>
            {render_news()}
          </div>

          <div class="card">
            <div class="h">SEC</div>
            {render_sec()}
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
            "news_cache": {},
            "sec_cache": {},
            "tech_cache": {},
            "groups": {},
            "crypto_rows": [],
            "instrument_meta": {},
            "energy_thesis": ENERGY_THESIS,
        })
        save_state(state)
        return {"status": "error", "detail": e.detail}

    instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]

    # Map instrumentId -> ticker (+ exchange meta)
    ticker_map, map_debug = await map_instrument_ids_to_tickers(instrument_ids)
    material_events.append(f"Mapped tickers via search: {map_debug['mapped']}/{map_debug['requested']}")

    portfolio_rows = build_portfolio_rows(agg_positions, ticker_map)

    # Group industry + crypto
    grouped, crypto_rows = group_rows(portfolio_rows)

    # tickers for tech/news/sec: stocks only (crypto excluded)
    stock_tickers: List[str] = []
    for industry in grouped:
        stock_tickers.extend(grouped[industry].get("tickers") or [])
    stock_tickers = list(dict.fromkeys(stock_tickers))

    # TECH
    tech_cache = await compute_tech_cache(stock_tickers)

    # NEWS + SEC
    news_cache, sec_cache = await compute_news_and_sec(stock_tickers)

    technical_exceptions.append("If you see Tech NA: symbol not available on Stooq for that ticker format (e.g., non-US / OTC).")
    ai_brief = await generate_openai_brief(portfolio_rows) if portfolio_rows else "No positions."

    state.update({
        "date": utc_now_iso(),
        "material_events": material_events,
        "technical_exceptions": technical_exceptions,
        "action_required": ["None"],
        "positions": portfolio_rows,
        "stats": stats,
        "ai_brief": ai_brief,
        "mapping": {"requested": map_debug["requested"], "mapped": map_debug["mapped"]},
        "mapping_last_debug": map_debug,
        "instrument_meta": map_debug.get("instrument_meta") or {},
        "tech_cache": tech_cache,
        "news_cache": news_cache,
        "sec_cache": sec_cache,
        "tickers": stock_tickers,
        "groups": grouped,
        "crypto_rows": crypto_rows,
        "energy_thesis": ENERGY_THESIS,
    })
    save_state(state)

    return {
        "status": "ok",
        "lots": stats["lots_count"],
        "unique_instruments": stats["unique_instruments_count"],
        "mapped_symbols": map_debug["mapped"],
        "stock_tickers": len(stock_tickers),
        "crypto_positions": len(crypto_rows),
        "tech_total": len(tech_cache),
        "news_total": sum(len(v) for v in news_cache.values()),
        "sec_total": sum(len(v) for v in sec_cache.values()),
    }


@app.get("/api/portfolio")
async def api_portfolio():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "stats": state.get("stats") or {},
        "positions": state.get("positions") or [],
        "tickers": state.get("tickers") or [],
        "groups": state.get("groups") or {},
        "crypto_rows": state.get("crypto_rows") or [],
        "instrument_meta": state.get("instrument_meta") or {},
        "tech_cache": state.get("tech_cache") or {},
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


@app.get("/api/news")
async def api_news():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "tickers": state.get("tickers") or [],
        "news_cache": state.get("news_cache") or {},
    })


@app.get("/api/sec")
async def api_sec():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "tickers": state.get("tickers") or [],
        "sec_cache": state.get("sec_cache") or {},
    })


@app.get("/debug/mapping-last")
async def debug_mapping_last():
    state = load_state()
    return JSONResponse(state.get("mapping_last_debug") or {"note": "Run /tasks/daily first."})
