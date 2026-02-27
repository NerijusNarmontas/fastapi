# main.py
# Changes based on your feedback:
# ✅ Adds your manual classifications:
#   - Crypto: SEI WLD PYTH HBAR CRO HYPE RPL ARBE (+ your existing CRYPTO_TICKERS)
#   - Energy: CEG (Energy)
#   - Space: BKSY FLY SPIR (Space / Space tech)
#   - Rare metals: ABAT USAR TMC (Rare metals)
#   - Nuclear: FERMI LTBR BOE.ASX (Nuclear)
#   - Tech: CRWV S.US INTC (Tech)
#   - Quantum: ARQQ RGTI (Quantum)
#   - Miners: RGLD (Miners)
#
# ✅ Cleaner UI:
#   - White screen / black text
#   - Category sections are COLLAPSIBLE (<details>)
#   - Each category shows ONE compact table
#   - News shown AFTER each category but compact:
#       - one unified list (top headlines across tickers)
#       - not separate huge cards per ticker
#   - SEC shown as compact list (latest per category)
#
# ✅ Categories now behave more like "industries" (your “industry groups”) instead of broad sectors:
#   - Energy / Gas / Nuclear / Space / Space Tech / Rare Metals / Miners / Tech / AI Infra / Quantum / Other
#
# NOTE: OpenAI/Gemini brief remains optional; you can disable by not setting OPENAI_API_KEY.
#       (If you later swap to Gemini, only the brief function changes.)

import os
import re
import json
import uuid
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

# "Collision excludes" (kept)
CRYPTO_EXCLUDE = set(
    s.strip().upper() for s in os.getenv("CRYPTO_EXCLUDE", "W").split(",") if s.strip()
)

# crypto list (you can extend via env)
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

NEWS_PER_TICKER = int(os.getenv("NEWS_PER_TICKER", "6"))
SEC_PER_TICKER = int(os.getenv("SEC_PER_TICKER", "6"))

# NEW: compact mode limits (reduce bulk)
CATEGORY_MAX_TICKERS_FOR_NEWS = int(os.getenv("CATEGORY_MAX_TICKERS_FOR_NEWS", "10"))
CATEGORY_MAX_NEWS_ITEMS = int(os.getenv("CATEGORY_MAX_NEWS_ITEMS", "12"))
CATEGORY_MAX_SEC_ITEMS = int(os.getenv("CATEGORY_MAX_SEC_ITEMS", "10"))

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))

# ----------------------------
# "Industry-like" categories (manual overrides first)
# ----------------------------
# Format: TICKER -> (IndustryGroup, SubGroup)
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

    # Core energy portfolio examples (keep these so grouping stays clean)
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

# energy thesis (tracked, not printed)
ENERGY_THESIS = {
    "focus": ["post-shale peak", "capital discipline", "FCF/ROIC", "gas tightness", "geopolitics", "nuclear buildout"],
    "avoid": ["refining"],
    "preferred": ["gas", "midstream", "uranium", "nuclear fuel", "smrs"],
}

# category order (clean)
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
    "Crypto",
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
    if t in ("CCJ", "UEC", "LEU", "SMR", "OKLO", "LTBR"):
        return ("Nuclear", "Nuclear")
    if t in ("ARQQ", "RGTI", "IONQ", "QBTS"):
        return ("Quantum", "Quantum")
    if t in ("BKSY", "SPIR", "RDW", "FLY"):
        return ("Space", "Space")
    if t in ("AU", "NEM", "KGC", "SBSW", "RGLD", "MP"):
        return ("Miners", "Miners")
    if t in ("ABAT", "USAR", "TMC", "ALB"):
        return ("Rare Metals", "Materials")
    return ("Other", "Unclassified")


def group_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      groups: { "Industry": { "subgroups": {sub: [rows]}, "rows": [rows], "tickers": [...]} }
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

        industry, sub = categorize_ticker(t)
        if industry not in groups:
            groups[industry] = {"subgroups": defaultdict(list), "rows": [], "tickers": []}
        groups[industry]["subgroups"][sub].append(r)
        groups[industry]["rows"].append(r)
        if t:
            groups[industry]["tickers"].append(t)

    # sort rows by weight desc
    for industry in groups:
        groups[industry]["rows"].sort(key=lambda x: float(x.get("weight_pct") or 0), reverse=True)
        groups[industry]["tickers"] = list(dict.fromkeys(groups[industry]["tickers"]))

    crypto_rows.sort(key=lambda x: float(x.get("weight_pct") or 0), reverse=True)

    return groups, crypto_rows


def ordered_industries(groups: Dict[str, Any]) -> List[str]:
    present = set(groups.keys())
    out = [x for x in CATEGORY_ORDER if x in present]
    # add any unknown at end
    for k in sorted(present):
        if k not in out:
            out.append(k)
    return out


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


async def map_instrument_ids_to_tickers(instrument_ids: List[int]) -> Tuple[Dict[int, str], Dict[str, Any]]:
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


# ----------------------------
# Portfolio rows + PnL%
# ----------------------------
def build_portfolio_rows(agg: List[Dict[str, Any]], ticker_map: Dict[int, str]) -> List[Dict[str, Any]]:
    total_initial = sum(float(x.get("initialAmountInDollars") or 0) for x in agg) or 0.0
    rows: List[Dict[str, Any]] = []

    for a in agg:
        iid = int(a["instrumentID"])
        ticker_raw = ticker_map.get(iid) or str(iid)
        ticker = normalize_ticker(ticker_raw)

        # Collision exclude for non-crypto only
        if ticker in CRYPTO_EXCLUDE and ticker not in CRYPTO_TICKERS:
            continue

        initial = normalize_number(a.get("initialAmountInDollars"))
        unreal = normalize_number(a.get("unrealizedPnL"))

        weight_pct = (initial / total_initial * 100.0) if (total_initial > 0 and initial and initial > 0) else None
        pnl_pct = (unreal / initial * 100.0) if (initial and initial != 0 and unreal is not None) else None

        rows.append({
            "ticker": ticker,
            "lots": str(a.get("lots", "")),
            "weight_pct": f"{weight_pct:.2f}" if weight_pct is not None else "",
            "pnl_pct": f"{pnl_pct:.2f}" if pnl_pct is not None else "",
            "instrumentID": str(iid),
            "initial_usd": float(initial or 0),
            "unreal_usd": float(unreal or 0),
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
    if any(k in t for k in ["offering", "private placement", "convertible", "debt", "notes"]):
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
    if f in ("S-1", "F-1"):
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
            m = re.search(r"\b(8-K|10-Q|10-K|S-1|F-1)\b", title, re.I)
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

    # fallback link
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
# Optional OpenAI brief (keep as-is; you can disable by no key)
# ----------------------------
async def generate_openai_brief(portfolio_rows: List[Dict[str, Any]]) -> str:
    if not OPENAI_API_KEY:
        return "AI brief disabled (no key)."

    top = portfolio_rows[:20]
    lines = [f"{r['ticker']}: w={r.get('weight_pct','')}% pnl={r.get('pnl_pct','')}%" for r in top]
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
            return f"OpenAI error {r.status_code}: {err}"
        data = r.json()
        return data.get("output_text") or "AI brief generated (output_text missing)."


# ----------------------------
# Daily Task: eToro -> tickers -> news + sec
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


def _compact_merge_news(news_cache: Dict[str, List[Dict[str, Any]]], tickers: List[str], limit: int) -> List[Dict[str, Any]]:
    """
    Merge news across many tickers into one list, dedupe by link.
    Keeps order by ticker order, then RSS order.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for t in tickers:
        for it in (news_cache.get(t) or []):
            link = (it.get("link") or "").strip()
            if not link or link in seen:
                continue
            seen.add(link)
            out.append(it)
            if len(out) >= limit:
                return out
    return out


def _compact_merge_sec(sec_cache: Dict[str, List[Dict[str, Any]]], tickers: List[str], limit: int) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for t in tickers:
        for it in (sec_cache.get(t) or []):
            link = (it.get("link") or "").strip()
            if not link or link in seen:
                continue
            seen.add(link)
            out.append(it)
            if len(out) >= limit:
                return out
    return out


# ----------------------------
# Routes
# ----------------------------
@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
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

    groups = state.get("groups") or {}          # {industry: {"rows":[], "tickers":[]}}
    crypto_rows = state.get("crypto_rows") or []

    news_cache = state.get("news_cache") or {}
    sec_cache = state.get("sec_cache") or {}

    def bullets(items: List[str]) -> str:
        lis = "".join([f"<li>{html_escape(x)}</li>" for x in items]) if items else "<li>None</li>"
        return f"<ul>{lis}</ul>"

    def render_table(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "<div class='muted'>No positions.</div>"
        out = []
        out.append("<table><thead><tr>"
                   "<th>Ticker</th><th>Lots</th><th>Weight %</th><th>P&amp;L %</th>"
                   "</tr></thead><tbody>")
        for r in rows[:200]:
            out.append(
                "<tr>"
                f"<td>{html_escape(r.get('ticker',''))}</td>"
                f"<td>{html_escape(r.get('lots',''))}</td>"
                f"<td>{html_escape(r.get('weight_pct',''))}</td>"
                f"<td>{html_escape(r.get('pnl_pct',''))}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
        return "".join(out)

    def render_compact_news(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "<div class='muted'>No news cached.</div>"
        out = ["<ul class='news'>"]
        for it in items:
            out.append(
                "<li>"
                f"<a href='{html_escape(it.get('link',''))}' target='_blank' rel='noopener noreferrer'>{html_escape(it.get('title',''))}</a>"
                f"<span class='meta'> — {html_escape(it.get('source',''))}</span>"
                f"<span class='tag'> {html_escape(it.get('resume',''))}</span>"
                "</li>"
            )
        out.append("</ul>")
        return "".join(out)

    def render_compact_sec(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "<div class='muted'>No SEC cached.</div>"
        out = ["<ul class='news'>"]
        for it in items:
            out.append(
                "<li>"
                f"<span class='badge'>{html_escape(it.get('form',''))}</span>"
                f"<a href='{html_escape(it.get('link',''))}' target='_blank' rel='noopener noreferrer'>{html_escape(it.get('title',''))}</a>"
                f"<span class='meta'> — {html_escape(it.get('resume',''))}</span>"
                "</li>"
            )
        out.append("</ul>")
        return "".join(out)

    # Build clean category sections (collapsible)
    category_html = ""
    for industry in ordered_industries(groups):
        g = groups.get(industry) or {}
        rows = g.get("rows") or []
        tickers = g.get("tickers") or []
        tickers = tickers[:CATEGORY_MAX_TICKERS_FOR_NEWS]

        merged_news = _compact_merge_news(news_cache, tickers, CATEGORY_MAX_NEWS_ITEMS)
        merged_sec = _compact_merge_sec(sec_cache, tickers, CATEGORY_MAX_SEC_ITEMS)

        header_line = f"{industry} <span class='muted'>({len(rows)} positions)</span>"
        category_html += f"""
        <details class="section" open>
          <summary>{header_line}</summary>
          <div class="block">
            {render_table(rows)}
          </div>
          <div class="block">
            <div class="h">News</div>
            {render_compact_news(merged_news)}
          </div>
          <div class="block">
            <div class="h">SEC</div>
            {render_compact_sec(merged_sec)}
          </div>
        </details>
        """

    crypto_html = ""
    if crypto_rows:
        crypto_html = f"""
        <details class="section">
          <summary>Crypto <span class="muted">({len(crypto_rows)} positions)</span></summary>
          <div class="block">
            {render_table(crypto_rows)}
          </div>
          <div class="muted">Crypto news/SEC not fetched by default.</div>
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
          .wrap {{ max-width: 1100px; }}
          .topline {{ display:flex; gap:18px; flex-wrap:wrap; align-items:baseline; }}
          code {{ background:#f3f3f3; border:1px solid #ddd; padding:2px 6px; border-radius:8px; }}

          table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
          th {{ background: #f5f5f5; text-align: left; }}

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
          .h {{
            font-weight: 700;
            margin: 8px 0 6px;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: #333;
          }}
          ul.news {{
            margin: 6px 0 0 18px;
            padding: 0;
          }}
          ul.news li {{
            margin: 6px 0;
            line-height: 1.25;
            font-size: 14px;
          }}
          .meta {{
            color:#555;
            font-size: 12px;
          }}
          .tag {{
            color:#222;
            font-size: 12px;
          }}
          .badge {{
            display:inline-block;
            font-size:11px;
            padding:2px 8px;
            border-radius:999px;
            border:1px solid #ddd;
            background:#f5f5f5;
            margin-right:6px;
          }}
          pre {{
            white-space: pre-wrap;
            background:#f7f7f7;
            border:1px solid #ddd;
            padding:12px;
            border-radius:12px;
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
          </div>

          <h2>Material Events</h2>
          {bullets(material_events)}

          <h2>Technical Exceptions</h2>
          {bullets(technical_exceptions)}

          <h2>Action Required</h2>
          {bullets(action_required)}

          <h2>AI Brief</h2>
          <pre>{html_escape(ai_brief or "No brief yet. Run /tasks/daily.")}</pre>

          {crypto_html}

          <h2>Stocks (industry groups)</h2>
          {category_html if category_html else "<div class='muted'>No categorized stock positions yet. Run /tasks/daily.</div>"}

          <p class="muted">API: <code>/api/portfolio</code> • <code>/api/news</code> • <code>/api/sec</code> • <code>/api/daily-brief</code> • Debug: <code>/debug/mapping-last</code></p>
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
            "groups": {},
            "crypto_rows": [],
            "energy_thesis": ENERGY_THESIS,
        })
        save_state(state)
        return {"status": "error", "detail": e.detail}

    instrument_ids = [int(x["instrumentID"]) for x in agg_positions if x.get("instrumentID") is not None]

    # Map instrumentId -> ticker via Search endpoint
    ticker_map, map_debug = await map_instrument_ids_to_tickers(instrument_ids)
    material_events.append(f"Mapped tickers via search: {map_debug['mapped']}/{map_debug['requested']}")

    portfolio_rows = build_portfolio_rows(agg_positions, ticker_map)

    # Group (industry-like) + crypto
    grouped, crypto_rows = group_rows(portfolio_rows)

    # Tickers for news/sec: stocks only
    stock_tickers_for_news: List[str] = []
    for industry in grouped:
        stock_tickers_for_news.extend(grouped[industry].get("tickers") or [])
    stock_tickers_for_news = list(dict.fromkeys(stock_tickers_for_news))

    # Fetch news + sec (stocks only)
    news_cache, sec_cache = await compute_news_and_sec(stock_tickers_for_news)

    technical_exceptions.append("Next: RSI/MACD/MAs/Volume/ADV + liquidity flags.")
    ai_brief = await generate_openai_brief(portfolio_rows) if portfolio_rows else "No positions."

    # For state, keep groups as a compact structure
    groups_for_state: Dict[str, Any] = {}
    for industry, g in grouped.items():
        groups_for_state[industry] = {
            "rows": g.get("rows") or [],
            "tickers": g.get("tickers") or [],
        }

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
        "news_cache": news_cache,
        "sec_cache": sec_cache,
        "tickers": stock_tickers_for_news,
        "groups": groups_for_state,
        "crypto_rows": crypto_rows,
        "energy_thesis": ENERGY_THESIS,
    })
    save_state(state)

    return {
        "status": "ok",
        "lots": stats["lots_count"],
        "unique_instruments": stats["unique_instruments_count"],
        "mapped_symbols": map_debug["mapped"],
        "stock_tickers_news": len(stock_tickers_for_news),
        "crypto_positions": len(crypto_rows),
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
