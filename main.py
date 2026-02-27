# main.py (lighter / faster)
# FastAPI single-file dashboard:
# - eToro REAL PnL pull
# - Aggregate by instrumentID
# - instrumentID -> ticker mapping with persistent cache (ticker_cache)
# - TECH line (Stooq) cached with TTL
# - Google News RSS cached with TTL
# - SEC links cached with TTL (CIK_MAP optional)
# - Clean white dashboard + per-ticker page

import os, re, json, uuid, asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

app = FastAPI(title="My AI Investing Agent (Lite)")

# ----------------------------
# ENV
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
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
).strip()

SEC_UA = os.getenv("SEC_UA", "").strip()
CIK_MAP_JSON = os.getenv("CIK_MAP", "").strip()

# crypto collision / list
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
    int(x.strip()) for x in os.getenv("CRYPTO_INSTRUMENT_IDS", "100000").split(",")
    if x.strip().isdigit()
)

# perf knobs
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "12"))

NEWS_PER_TICKER = int(os.getenv("NEWS_PER_TICKER", "6"))
SEC_PER_TICKER = int(os.getenv("SEC_PER_TICKER", "6"))

# TTLs (seconds) – tune for speed vs freshness
TTL_TECH = int(os.getenv("TTL_TECH", str(6 * 60 * 60)))      # 6h
TTL_NEWS = int(os.getenv("TTL_NEWS", str(3 * 60 * 60)))      # 3h
TTL_SEC  = int(os.getenv("TTL_SEC",  str(12 * 60 * 60)))     # 12h
TTL_MAP  = int(os.getenv("TTL_MAP",  str(7 * 24 * 60 * 60))) # 7d (rarely changes)

# ----------------------------
# State I/O
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
# Small helpers
# ----------------------------
def html_escape(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    return re.sub(r"[^A-Z0-9\.\-]", "", t)

def safe_float(x: Any) -> float:
    try: return float(x)
    except Exception: return 0.0

def fmt_money(x: Optional[float]) -> str:
    if x is None: return ""
    try: return f"{x:,.0f}"
    except Exception: return ""

def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def is_stale(ts: Optional[int], ttl: int) -> bool:
    if not ts: return True
    return (now_ts() - int(ts)) > ttl

def is_crypto_ticker(ticker: str, instrument_id: Optional[int] = None) -> bool:
    t = normalize_ticker(ticker)
    if t in CRYPTO_TICKERS: return True
    if instrument_id is not None and instrument_id in CRYPTO_INSTRUMENT_IDS: return True
    if t.endswith("-USD") and t[:-4] in CRYPTO_TICKERS: return True
    return False

def tradingview_url(ticker: str, exchange: str = "") -> str:
    t = normalize_ticker(ticker)
    ex = (exchange or "").strip().upper()
    if ex:
        return f"https://www.tradingview.com/chart/?symbol={ex}:{t}"
    return f"https://www.tradingview.com/symbols/{t}/"

# ----------------------------
# eToro
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

async def etoro_get_real_pnl(client: httpx.AsyncClient) -> Dict[str, Any]:
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        raise HTTPException(status_code=400, detail="Missing ETORO_API_KEY or ETORO_USER_KEY")
    r = await client.get(ETORO_REAL_PNL_URL, headers=etoro_headers())
    if r.status_code >= 400:
        try: payload = r.json()
        except Exception: payload = {"text": r.text}
        raise HTTPException(status_code=r.status_code, detail=payload)
    return r.json()

async def etoro_search(client: httpx.AsyncClient, params: Dict[str, str]) -> Tuple[int, Any]:
    r = await client.get(ETORO_SEARCH_URL, headers=etoro_headers(), params=params)
    try: data = r.json()
    except Exception: data = {"text": r.text}
    return r.status_code, data

def extract_positions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict): return []
    cp = payload.get("clientPortfolio") or payload.get("ClientPortfolio") or {}
    if isinstance(cp, dict) and isinstance(cp.get("positions"), list):
        return cp["positions"]
    if isinstance(payload.get("positions"), list):
        return payload["positions"]
    return []

def pick_instrument_id(p: Dict[str, Any]) -> Optional[int]:
    iid = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
    try: return int(iid) if iid is not None else None
    except Exception: return None

def pick_unrealized_pnl(p: Dict[str, Any]) -> float:
    up = p.get("unrealizedPnL") or p.get("unrealizedPnl") or p.get("unrealized_pnl")
    if isinstance(up, dict):
        return safe_float(up.get("pnL"))
    return safe_float(up)

def pick_initial_usd(p: Dict[str, Any]) -> float:
    return safe_float(p.get("initialAmountInDollars") or p.get("initialAmount") or p.get("initial_amount_usd"))

def aggregate_positions_by_instrument(raw_positions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    buckets = defaultdict(list)
    for p in raw_positions:
        iid = pick_instrument_id(p)
        if iid is None: continue
        buckets[iid].append(p)

    agg: List[Dict[str, Any]] = []
    for iid, lots in buckets.items():
        total_initial = sum(pick_initial_usd(x) for x in lots)
        total_unreal = sum(pick_unrealized_pnl(x) for x in lots)
        agg.append({
            "instrumentID": iid,
            "lots": len(lots),
            "initialAmountInDollars": total_initial,
            "unrealizedPnL": total_unreal,
        })

    agg.sort(key=lambda x: safe_float(x.get("initialAmountInDollars")), reverse=True)
    stats = {"lots_count": len(raw_positions), "unique_instruments_count": len(agg)}
    return agg, stats

def _extract_ticker(item: Dict[str, Any]) -> str:
    for k in ("internalSymbolFull", "symbolFull", "symbol"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _extract_exchange(item: Dict[str, Any]) -> str:
    for k in ("exchange", "exchangeName", "exchangeCode", "marketName", "market"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

async def map_instrument_ids_to_tickers_cached(
    client: httpx.AsyncClient,
    state: Dict[str, Any],
    instrument_ids: List[int],
) -> Tuple[Dict[int, str], Dict[str, Any], Dict[str, Dict[str, str]]]:
    """
    Returns:
      ticker_map: instrumentID -> ticker
      debug: mapping debug
      instrument_meta: {"123": {"exchange":"..."}}
    Keeps persistent cache in state["ticker_cache"] with TTL.
    """
    cache = state.get("ticker_cache") or {}  # {"123": {"ticker":"AAPL","exchange":"NASDAQ","ts":...}}
    instrument_meta: Dict[str, Dict[str, str]] = state.get("instrument_meta") or {}

    ids = sorted(set(int(x) for x in instrument_ids if x is not None))
    ticker_map: Dict[int, str] = {}

    # fill from cache first
    for iid in ids:
        entry = cache.get(str(iid)) or {}
        if entry.get("ticker") and not is_stale(entry.get("ts"), TTL_MAP):
            ticker_map[iid] = entry["ticker"]
            if entry.get("exchange"):
                instrument_meta[str(iid)] = {"exchange": entry["exchange"]}

    missing = [iid for iid in ids if iid not in ticker_map]

    debug = {"requested": len(ids), "cached": len(ticker_map), "mapped": 0, "failed": 0, "samples": []}

    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(iid: int):
        async with sem:
            # try instrumentId
            status, data = await etoro_search(client, {"instrumentId": str(iid), "pageSize": "5", "pageNumber": "1"})
            items = data.get("items") if isinstance(data, dict) else None
            if isinstance(items, list) and items:
                for it in items:
                    if not isinstance(it, dict): continue
                    t = _extract_ticker(it)
                    if t:
                        ex = _extract_exchange(it)
                        ticker_map[iid] = t
                        cache[str(iid)] = {"ticker": t, "exchange": ex, "ts": now_ts()}
                        if ex:
                            instrument_meta[str(iid)] = {"exchange": ex}
                        debug["mapped"] += 1
                        if len(debug["samples"]) < 8:
                            debug["samples"].append({"instrumentID": iid, "ticker": t, "exchange": ex, "via": "instrumentId", "status": status})
                        return

            # try internalInstrumentId
            status2, data2 = await etoro_search(client, {"internalInstrumentId": str(iid), "pageSize": "5", "pageNumber": "1"})
            items2 = data2.get("items") if isinstance(data2, dict) else None
            if isinstance(items2, list) and items2:
                for it in items2:
                    if not isinstance(it, dict): continue
                    t = _extract_ticker(it)
                    if t:
                        ex = _extract_exchange(it)
                        ticker_map[iid] = t
                        cache[str(iid)] = {"ticker": t, "exchange": ex, "ts": now_ts()}
                        if ex:
                            instrument_meta[str(iid)] = {"exchange": ex}
                        debug["mapped"] += 1
                        if len(debug["samples"]) < 8:
                            debug["samples"].append({"instrumentID": iid, "ticker": t, "exchange": ex, "via": "internalInstrumentId", "status": status2})
                        return

            debug["failed"] += 1
            if len(debug["samples"]) < 8:
                debug["samples"].append({"instrumentID": iid, "status": "no_match_or_no_ticker"})

    if missing:
        await asyncio.gather(*(one(i) for i in missing))

    state["ticker_cache"] = cache
    state["instrument_meta"] = instrument_meta
    state["mapping_last_debug"] = debug
    return ticker_map, debug, instrument_meta

# ----------------------------
# Build rows
# ----------------------------
def build_portfolio_rows(agg: List[Dict[str, Any]], ticker_map: Dict[int, str]) -> List[Dict[str, Any]]:
    total_initial = sum(safe_float(x.get("initialAmountInDollars")) for x in agg) or 0.0
    rows: List[Dict[str, Any]] = []

    for a in agg:
        iid = int(a["instrumentID"])
        ticker_raw = ticker_map.get(iid) or str(iid)
        ticker = normalize_ticker(ticker_raw)

        # collision exclude for non-crypto only
        if ticker in CRYPTO_EXCLUDE and ticker not in CRYPTO_TICKERS:
            continue

        initial = safe_float(a.get("initialAmountInDollars"))
        unreal = safe_float(a.get("unrealizedPnL"))

        weight_pct = (initial / total_initial * 100.0) if (total_initial > 0 and initial > 0) else None
        pnl_pct = (unreal / initial * 100.0) if (initial != 0 and a.get("unrealizedPnL") is not None) else None

        rows.append({
            "ticker": ticker,
            "instrumentID": str(iid),
            "lots": int(a.get("lots", 0) or 0),

            "invested_usd": float(initial),
            "unreal_usd": float(unreal),
            "weight_pct_num": float(weight_pct or 0.0),
            "pnl_pct_num": float(pnl_pct or 0.0) if pnl_pct is not None else 0.0,
            "pnl_pct_is_na": pnl_pct is None,

            "invested_disp": fmt_money(initial),
            "weight_pct": f"{weight_pct:.2f}" if weight_pct is not None else "",
            "pnl_pct": f"{pnl_pct:.2f}" if pnl_pct is not None else "",
        })
    return rows

def split_crypto(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    stocks, crypto = [], []
    for r in rows:
        t = r.get("ticker","")
        iid = None
        try: iid = int(r.get("instrumentID") or 0)
        except Exception: iid = None
        (crypto if is_crypto_ticker(t, iid) else stocks).append(r)
    return stocks, crypto

# ----------------------------
# TECH (Stooq) - cached per ticker
# ----------------------------
def sma(values: List[float], n: int) -> Optional[float]:
    if len(values) < n: return None
    return sum(values[-n:]) / n

def rsi14(closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1: return None
    gains, losses = [], []
    for i in range(-n, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0: gains.append(ch); losses.append(0.0)
        else: gains.append(0.0); losses.append(-ch)
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def monthly_return_pct(closes: List[float], lookback_days: int = 21) -> Optional[float]:
    if len(closes) < lookback_days + 1: return None
    a = closes[-lookback_days - 1]
    b = closes[-1]
    if a == 0: return None
    return (b / a - 1.0) * 100.0

def liquidity_flag(bars: List[Dict[str, float]], days: int = 20) -> str:
    if len(bars) < days: return "NA"
    recent = bars[-days:]
    dvs = [(x["close"] * (x.get("volume") or 0.0)) for x in recent]
    avg_dv = sum(dvs) / days if dvs else 0.0
    return "OK" if avg_dv >= 2_000_000 else "LOW"

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
        if len(lines) < 30: continue
        out = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) < 6: continue
            try:
                out.append({
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "volume": float(parts[5]) if parts[5] not in ("", "nan", "NaN") else 0.0,
                })
            except Exception:
                continue
        if len(out) >= 30:
            return out
    return []

def build_compact_line(ticker: str, bars: List[Dict[str, float]]) -> Dict[str, Any]:
    closes = [x["close"] for x in bars]
    last = closes[-1] if closes else None

    mret = monthly_return_pct(closes)
    sma200 = sma(closes, 200)
    trend = None
    if last is not None and sma200 is not None:
        trend = "above" if last >= sma200 else "below"

    rsi = rsi14(closes, 14)
    liq = liquidity_flag(bars, 20)

    warn = bool((rsi is not None and rsi <= 30) or (liq == "LOW"))

    parts = []
    if mret is not None: parts.append(f"1M {mret:+.0f}%")
    if trend: parts.append(f"Trend {trend} 200DMA")
    if rsi is not None: parts.append(f"RSI {rsi:.0f}")
    parts.append(f"Liq {liq}")

    return {"ticker": ticker, "compact": " | ".join(parts), "warn": warn, "mret_1m_pct": mret}

async def ensure_tech_cache(client: httpx.AsyncClient, state: Dict[str, Any], tickers: List[str]) -> Dict[str, Any]:
    tech_cache = state.get("tech_cache") or {}  # {T: {"data":..., "ts":...}}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(t: str):
        async with sem:
            entry = tech_cache.get(t) or {}
            if entry and not is_stale(entry.get("ts"), TTL_TECH):
                return
            bars = await fetch_stooq_daily(client, t)
            if bars:
                tech_cache[t] = {"data": build_compact_line(t, bars), "ts": now_ts()}
            else:
                tech_cache[t] = {"data": {"ticker": t, "compact": "Tech NA (no data)", "warn": False}, "ts": now_ts()}

    await asyncio.gather(*(one(t) for t in tickers))
    state["tech_cache"] = tech_cache
    return tech_cache

# ----------------------------
# NEWS (Google RSS) - cached per ticker
# ----------------------------
def parse_rss_items(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    channel = root.find("channel")
    if channel is None:
        return []
    out = []
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
    if any(k in t for k in ["earnings", "guidance", "revenue", "eps"]): tags.append("Earnings / guidance")
    if any(k in t for k in ["acquire", "acquisition", "merger", "buyout", "takeover"]): tags.append("M&A")
    if any(k in t for k in ["offering", "private placement", "convertible", "debt", "notes", "atm", "shelf"]): tags.append("Financing")
    if any(k in t for k in ["contract", "award", "partnership", "collaboration"]): tags.append("Commercial deal")
    if any(k in t for k in ["lawsuit", "probe", "investigation", "fraud", "settlement"]): tags.append("Legal / investigation")
    if any(k in t for k in ["upgrade", "downgrade", "price target", "analyst"]): tags.append("Analyst move")
    if any(k in t for k in ["sec", "8-k", "10-q", "10-k", "s-1", "f-1", "edgar"]): tags.append("Filing-related")
    return " • ".join(tags) if tags else "Headline update"

async def fetch_google_news(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, Any]]:
    q = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = await client.get(url, headers={"accept": "application/rss+xml, text/xml;q=0.9,*/*;q=0.8", "user-agent": DEFAULT_UA})
    if r.status_code != 200:
        return []
    items = parse_rss_items(r.text)
    out, seen = [], set()
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

async def ensure_news_cache(client: httpx.AsyncClient, state: Dict[str, Any], tickers: List[str]) -> Dict[str, Any]:
    news_cache = state.get("news_cache") or {}  # {T: {"data":[...], "ts":...}}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(t: str):
        async with sem:
            entry = news_cache.get(t) or {}
            if entry and not is_stale(entry.get("ts"), TTL_NEWS):
                return
            news_cache[t] = {"data": await fetch_google_news(client, t), "ts": now_ts()}

    await asyncio.gather(*(one(t) for t in tickers))
    state["news_cache"] = news_cache
    return news_cache

# ----------------------------
# SEC - cached per ticker
# ----------------------------
def resume_sec(form: str) -> str:
    f = (form or "").upper().strip()
    if f == "8-K": return "Material event."
    if f == "10-Q": return "Quarterly report."
    if f == "10-K": return "Annual report."
    if f in ("S-1","F-1") or f.startswith("424B"): return "Registration / offering."
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

async def fetch_sec(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, Any]]:
    cik_map: Dict[str, str] = {}
    if CIK_MAP_JSON:
        try: cik_map = json.loads(CIK_MAP_JSON)
        except Exception: cik_map = {}

    cik = (cik_map.get(ticker) or "").strip()
    if cik:
        cik = cik.zfill(10)
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&owner=exclude&count=40&output=atom"
        r = await client.get(url, headers={
            "accept": "application/atom+xml, application/xml;q=0.9,*/*;q=0.8",
            "user-agent": SEC_UA or DEFAULT_UA,
        })
        if r.status_code != 200:
            return []
        entries = parse_atom_entries(r.text)
        out, seen = [], set()
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

    # fallback
    return [{
        "ticker": ticker,
        "form": "EDGAR",
        "title": f"EDGAR search for {ticker}",
        "link": f"https://www.sec.gov/edgar/search/#/q={quote_plus(ticker)}&sort=desc",
        "filed": "",
        "resume": "Open EDGAR search.",
    }]

async def ensure_sec_cache(client: httpx.AsyncClient, state: Dict[str, Any], tickers: List[str]) -> Dict[str, Any]:
    sec_cache = state.get("sec_cache") or {}  # {T: {"data":[...], "ts":...}}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(t: str):
        async with sem:
            entry = sec_cache.get(t) or {}
            if entry and not is_stale(entry.get("ts"), TTL_SEC):
                return
            sec_cache[t] = {"data": await fetch_sec(client, t), "ts": now_ts()}

    await asyncio.gather(*(one(t) for t in tickers))
    state["sec_cache"] = sec_cache
    return sec_cache

# ----------------------------
# Sorting
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
    stats = state.get("stats") or {}
    instrument_meta = state.get("instrument_meta") or {}
    mapping_last = state.get("mapping_last_debug") or {}
    mapped = mapping_last.get("cached", 0) + mapping_last.get("mapped", 0)
    requested = mapping_last.get("requested", "")

    positions = state.get("positions") or []
    stocks, crypto = split_crypto(positions)

    sort_by = (sort or "weight").lower()
    direction = (dir or "desc").lower()
    if sort_by not in SORT_KEYS: sort_by = "weight"
    if direction not in ("asc","desc"): direction = "desc"

    tech_cache = state.get("tech_cache") or {}

    def hdr(label: str, key: str) -> str:
        next_dir = "asc" if (sort_by == key and direction == "desc") else "desc"
        arrow = ""
        if sort_by == key: arrow = " ▲" if direction == "asc" else " ▼"
        return f"<a href='/?sort={html_escape(key)}&dir={html_escape(next_dir)}'>{html_escape(label)}{arrow}</a>"

    def render_table(rows: List[Dict[str, Any]]) -> str:
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

        for r in rows_sorted[:400]:
            t = r.get("ticker","")
            iid = r.get("instrumentID","0")
            ex = (instrument_meta.get(str(iid)) or {}).get("exchange","")
            tv = tradingview_url(t, ex)

            pnl_num = None if r.get("pnl_pct_is_na") else safe_float(r.get("pnl_pct_num"))
            pnl_class = "na"
            if pnl_num is not None:
                pnl_class = "pos" if pnl_num > 0 else ("neg" if pnl_num < 0 else "flat")

            tech_entry = (tech_cache.get(t) or {}).get("data") or {}
            compact = tech_entry.get("compact") or ""
            warn = bool(tech_entry.get("warn"))
            mret = tech_entry.get("mret_1m_pct", None)

            compact_html = ""
            if compact:
                icon = "⚠️ " if warn else ""
                if mret is not None:
                    mret_class = "pos" if mret > 0 else ("neg" if mret < 0 else "flat")
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

            out.append(
                "<tr><td>"
                f"{ticker_link}"
                f"<sup style='margin-left:6px;'><a href='{html_escape(tv)}' target='_blank' rel='noopener noreferrer' "
                f"title='Open in TradingView' style='font-size:11px; text-decoration:none;'>tv</a></sup>"
                f"{compact_html}"
                "</td>"
                f"<td>{html_escape(str(r.get('lots','')))}</td>"
                f"<td>{html_escape(r.get('invested_disp',''))}</td>"
                f"<td>{html_escape(r.get('weight_pct',''))}</td>"
                f"<td class='{pnl_class}'>{html_escape(r.get('pnl_pct',''))}</td></tr>"
            )

        out.append("</tbody></table>")
        return "".join(out)

    html = f"""
    <html><head><meta charset="utf-8"/><title>My AI Investing Agent</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; margin:24px; background:#fff; color:#000; }}
      a {{ color:#0b57d0; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
      .muted {{ color:#555; }}
      table {{ border-collapse: collapse; width:100%; margin-top:8px; }}
      th, td {{ border:1px solid #ddd; padding:8px; font-size:14px; vertical-align:top; }}
      th {{ background:#f5f5f5; text-align:left; }}
      th a {{ color:#111; }}
      .pos {{ color:#0a7a2f; font-weight:700; }}
      .neg {{ color:#b00020; font-weight:700; }}
      .flat {{ color:#111; font-weight:700; }}
      .na {{ color:#555; }}
      .techline {{ font-size:12px; margin-top:3px; }}
      .section {{ border:1px solid #ddd; border-radius:12px; padding:12px; margin:12px 0; }}
      code {{ background:#f3f3f3; border:1px solid #ddd; padding:2px 6px; border-radius:8px; }}
      .topline {{ display:flex; gap:16px; flex-wrap:wrap; align-items:baseline; }}
    </style></head>
    <body>
      <h1 style="margin-bottom:10px;">My AI Investing Agent</h1>
      <div class="topline">
        <div><b>Last update:</b> {html_escape(state.get("date") or utc_now_iso())}</div>
        <div><b>Lots:</b> {stats.get("lots_count","")} · <b>Unique instruments:</b> {stats.get("unique_instruments_count","")}</div>
        <div><b>Mapping:</b> {mapped}/{requested}</div>
        <div>Refresh: <code>/tasks/daily</code></div>
        <div class="muted">Sort: <code>?sort=ticker|invested|weight|pnl|lots&dir=asc|desc</code></div>
      </div>

      <div class="section">
        <h2 style="margin:0 0 6px 0;">Stocks</h2>
        {render_table(stocks)}
      </div>

      <div class="section">
        <h2 style="margin:0 0 6px 0;">Crypto</h2>
        <div class="muted">Crypto tech/news/sec not fetched by default. Click ticker for its links page, and “tv” for TradingView.</div>
        {render_table(crypto)}
      </div>

      <p class="muted">API: <code>/api/portfolio</code> • <code>/api/news</code> • <code>/api/sec</code> • Debug: <code>/debug/mapping-last</code></p>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/t/{ticker}", response_class=HTMLResponse)
async def ticker_page(ticker: str):
    state = load_state()
    t = normalize_ticker(ticker)

    news_cache = state.get("news_cache") or {}
    sec_cache = state.get("sec_cache") or {}

    news_items = ((news_cache.get(t) or {}).get("data") or [])[:NEWS_PER_TICKER]
    sec_items  = ((sec_cache.get(t)  or {}).get("data") or [])[:SEC_PER_TICKER]

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
    <html><head><meta charset="utf-8"/><title>{html_escape(t)} — Links</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; margin:24px; background:#fff; color:#000; }}
      a {{ color:#0b57d0; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
      .muted {{ color:#555; }}
      .wrap {{ max-width: 1000px; }}
      .top {{ display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }}
      .btn {{ display:inline-block; padding:6px 10px; border:1px solid #ddd; border-radius:10px; background:#f5f5f5; color:#111; font-size:13px; }}
      .card {{ border:1px solid #ddd; border-radius:12px; padding:12px; margin:12px 0; }}
      .h {{ font-weight:800; font-size:13px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; }}
      ul.lst {{ margin: 8px 0 0 18px; padding:0; }} ul.lst li {{ margin: 10px 0; line-height:1.25; }}
      .meta {{ font-size:12px; color:#555; margin-top:2px; }} .tag {{ font-size:12px; color:#222; margin-top:2px; }}
      .badge {{ display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px; border:1px solid #ddd; background:#f5f5f5; margin-right:6px; }}
    </style></head>
    <body><div class="wrap">
      <div class="top">
        <h1 style="margin:0;">{html_escape(t)}</h1>
        <a class="btn" href="/" target="_self">Back</a>
        <a class="btn" href="{html_escape(tv)}" target="_blank" rel="noopener noreferrer">Open TradingView</a>
        <span class="muted">Last update: {html_escape(state.get("date") or utc_now_iso())}</span>
      </div>

      <div class="card"><div class="h">News</div>{render_news()}</div>
      <div class="card"><div class="h">SEC</div>{render_sec()}</div>
    </div></body></html>
    """
    return HTMLResponse(html)

@app.get("/tasks/daily")
async def run_daily(request: Request, tech: int = 1, news: int = 1, sec: int = 1):
    # if you want admin protection: uncomment
    # require_admin(request)

    state = load_state()
    material_events: List[str] = []

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        # 1) pull portfolio
        try:
            payload = await etoro_get_real_pnl(client)
            raw_positions = extract_positions(payload)
            agg_positions, stats = aggregate_positions_by_instrument(raw_positions)
            material_events.append(f"Pulled eToro OK. Lots={stats['lots_count']} Unique={stats['unique_instruments_count']}")
        except HTTPException as e:
            state.update({
                "date": utc_now_iso(),
                "material_events": material_events + [f"eToro API error: {e.status_code}", str(e.detail)],
                "positions": [],
                "stats": {"lots_count": 0, "unique_instruments_count": 0},
            })
            save_state(state)
            return {"status": "error", "detail": e.detail}

        instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]

        # 2) map instrumentIDs (cached)
        ticker_map, map_debug, instrument_meta = await map_instrument_ids_to_tickers_cached(client, state, instrument_ids)
        material_events.append(f"Mapping: cached={map_debug.get('cached',0)} new={map_debug.get('mapped',0)} failed={map_debug.get('failed',0)}")

        # 3) build rows
        portfolio_rows = build_portfolio_rows(agg_positions, ticker_map)

        # only stock tickers for tech/news/sec
        stocks, _crypto = split_crypto(portfolio_rows)
        stock_tickers = list(dict.fromkeys([r["ticker"] for r in stocks if r.get("ticker")]))

        # 4) refresh caches (selectively)
        if tech and stock_tickers:
            await ensure_tech_cache(client, state, stock_tickers)
        if news and stock_tickers:
            await ensure_news_cache(client, state, stock_tickers)
        if sec and stock_tickers:
            await ensure_sec_cache(client, state, stock_tickers)

    state.update({
        "date": utc_now_iso(),
        "material_events": material_events,
        "positions": portfolio_rows,
        "stats": stats,
        "tickers": stock_tickers,
        "instrument_meta": instrument_meta,
    })
    save_state(state)

    return {
        "status": "ok",
        "lots": stats["lots_count"],
        "unique_instruments": stats["unique_instruments_count"],
        "stock_tickers": len(stock_tickers),
        "mapping": map_debug,
        "refreshed": {"tech": bool(tech), "news": bool(news), "sec": bool(sec)},
    }

@app.get("/api/portfolio")
async def api_portfolio():
    state = load_state()
    return JSONResponse({
        "date": state.get("date") or utc_now_iso(),
        "stats": state.get("stats") or {},
        "positions": state.get("positions") or [],
        "tickers": state.get("tickers") or [],
        "instrument_meta": state.get("instrument_meta") or {},
        "tech_cache": state.get("tech_cache") or {},
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
