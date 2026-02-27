# main.py
# Requires Railway startCommand: hypercorn main:app --bind "0.0.0.0:$PORT"
#
# Required Railway Variables:
#   ETORO_API_KEY=...
#   ETORO_USER_KEY=...
#   SEC_UA=YourName AppName (email@domain.com)
#
# Required for ticker mapping (instrumentID -> symbol):
#   ETORO_INSTRUMENTS_URL=...   (your working instruments lookup endpoint)
#
# Optional:
#   ETORO_PNL_URL=https://public-api.etoro.com/api/v1/trading/info/real/pnl
#   CRYPTO_EXCLUDE=W
#   NEWS_PER_TICKER=6
#   SEC_PER_TICKER=6
#   DEFAULT_UA=...

import os, re, json, uuid, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

APP_NAME = "Investing Dashboard"
STATE_PATH = os.getenv("STATE_PATH", "/tmp/investing_agent_state.json")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))

NEWS_PER_TICKER = int(os.getenv("NEWS_PER_TICKER", "6"))
SEC_PER_TICKER = int(os.getenv("SEC_PER_TICKER", "6"))

DEFAULT_UA = os.getenv(
    "DEFAULT_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
)

SEC_UA = os.getenv("SEC_UA", "InvestingDashboard (contact: you@example.com)")
CRYPTO_EXCLUDE = set(s.strip().upper() for s in os.getenv("CRYPTO_EXCLUDE", "W").split(",") if s.strip())
CIK_MAP_JSON = os.getenv("CIK_MAP", "").strip()

ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()

ETORO_PNL_URL = os.getenv(
    "ETORO_PNL_URL",
    "https://public-api.etoro.com/api/v1/trading/info/real/pnl",
).strip()

# YOU must set this to your working lookup endpoint:
ETORO_INSTRUMENTS_URL = os.getenv("ETORO_INSTRUMENTS_URL", "").strip()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)

STATE: Dict[str, Any] = load_state()
STATE.setdefault("tickers", [])
STATE.setdefault("news_cache", {})
STATE.setdefault("sec_cache", {})
STATE.setdefault("last_run", None)
STATE.setdefault("debug", {})

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    return re.sub(r"[^A-Z0-9\.\-]", "", t)

def safe_text(x: Optional[str]) -> str:
    return (x or "").replace("\x00", "").strip()

def html_escape(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def dedupe_list(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def dedupe_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        v = (it.get(key) or "").strip()
        if not v or v in seen: 
            continue
        seen.add(v); out.append(it)
    return out

def parse_rss_or_atom(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    channel = root.find("channel")
    if channel is not None:
        out=[]
        for item in channel.findall("item"):
            out.append({
                "title": safe_text(item.findtext("title")),
                "link": safe_text(item.findtext("link")),
                "published_raw": safe_text(item.findtext("pubDate")),
                "source": safe_text(item.findtext("source")),
            })
        return out
    out=[]
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = safe_text(entry.findtext("{http://www.w3.org/2005/Atom}title"))
        updated = safe_text(entry.findtext("{http://www.w3.org/2005/Atom}updated"))
        link = ""
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        if link_el is not None and "href" in link_el.attrib:
            link = safe_text(link_el.attrib["href"])
        out.append({"title": title, "link": link, "published_raw": updated, "source": ""})
    return out

def resume_news(title: str) -> str:
    t=(title or "").lower()
    tags=[]
    if any(k in t for k in ["earnings","guidance","eps","revenue","q1","q2","q3","q4"]): tags.append("Earnings/guidance")
    if any(k in t for k in ["acquisition","acquire","merger","buyout","takeover"]): tags.append("M&A")
    if any(k in t for k in ["offering","convertible","debt","notes"]): tags.append("Financing")
    if any(k in t for k in ["contract","award","partnership","collaboration"]): tags.append("Commercial deal")
    if any(k in t for k in ["lawsuit","investigation","probe","settlement"]): tags.append("Legal")
    return " | ".join(tags) if tags else "Headline update."

def resume_sec(form: str) -> str:
    f=(form or "").upper()
    if f=="8-K": return "Material event. Check earnings/financing/M&A/leadership/contracts."
    if f=="10-Q": return "Quarterly report. Margins, cash flow, guidance, balance sheet."
    if f=="10-K": return "Annual report. Risks, liquidity, business changes."
    return "Filing update."

@dataclass
class NewsItem:
    ticker: str; title: str; link: str; source: str; published_raw: str; resume: str

@dataclass
class SecItem:
    ticker: str; form: str; title: str; link: str; filed: str; resume: str


def etoro_headers() -> Dict[str, str]:
    return {
        "accept": "application/json",
        "user-agent": DEFAULT_UA,
        "x-api-key": ETORO_API_KEY,
        "x-user-key": ETORO_USER_KEY,
        "x-request-id": str(uuid.uuid4()),
    }

def extract_instrument_ids(payload: Dict[str, Any]) -> List[int]:
    ids=[]
    cp = payload.get("clientPortfolio") if isinstance(payload, dict) else None
    positions = []
    if isinstance(cp, dict) and isinstance(cp.get("positions"), list):
        positions = cp["positions"]
    elif isinstance(payload.get("positions"), list):
        positions = payload["positions"]
    for p in positions:
        v = p.get("instrumentID")
        if isinstance(v, int):
            ids.append(v)
        else:
            try:
                ids.append(int(v))
            except Exception:
                pass
    return sorted(set(ids))

async def fetch_etoro_pnl(client: httpx.AsyncClient) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    dbg={"url": ETORO_PNL_URL}
    r = await client.get(ETORO_PNL_URL, headers=etoro_headers())
    dbg["status_code"]=r.status_code
    if r.status_code != 200:
        dbg["body_head"]=r.text[:300]
        return None, dbg
    return r.json(), dbg

def parse_instruments_response(data: Any) -> Dict[int, str]:
    """
    Best-effort parsing. Different endpoints return different shapes.
    We try common patterns:
    - list of {instrumentId/id, symbol/ticker}
    - dict with 'instruments' list
    - dict keyed by id
    """
    mapping: Dict[int, str] = {}

    candidates: List[Dict[str, Any]] = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        if isinstance(data.get("instruments"), list):
            candidates = data["instruments"]
        elif isinstance(data.get("items"), list):
            candidates = data["items"]
        else:
            # maybe id->object
            for k, v in data.items():
                if isinstance(v, dict):
                    vv = v.copy()
                    vv["_key"] = k
                    candidates.append(vv)

    for it in candidates:
        raw_id = it.get("instrumentId") or it.get("instrumentID") or it.get("id") or it.get("_key")
        sym = it.get("symbol") or it.get("ticker") or it.get("Symbol") or it.get("Ticker")
        if raw_id is None or not sym:
            continue
        try:
            iid = int(raw_id)
        except Exception:
            continue
        mapping[iid] = normalize_ticker(str(sym))
    return mapping

async def fetch_instrument_map(client: httpx.AsyncClient, instrument_ids: List[int]) -> Tuple[Dict[int,str], Dict[str, Any]]:
    dbg={"url": ETORO_INSTRUMENTS_URL, "ids": len(instrument_ids)}
    if not ETORO_INSTRUMENTS_URL:
        dbg["error"]="Missing ETORO_INSTRUMENTS_URL env var"
        return {}, dbg
    if not instrument_ids:
        dbg["note"]="No instrument IDs"
        return {}, dbg

    # Try GET with ids query first: ?ids=1,2,3
    ids_csv = ",".join(str(i) for i in instrument_ids[:500])
    try:
        r = await client.get(f"{ETORO_INSTRUMENTS_URL}?ids={ids_csv}", headers=etoro_headers())
        dbg["get_status"]=r.status_code
        if r.status_code == 200:
            mp = parse_instruments_response(r.json())
            dbg["mapped"]=len(mp)
            if mp:
                dbg["sample"]=list(mp.items())[:5]
            return mp, dbg
    except Exception as e:
        dbg["get_error"]=f"{type(e).__name__}: {e}"

    # Fallback: POST JSON body {"ids":[...]}
    try:
        r = await client.post(ETORO_INSTRUMENTS_URL, headers=etoro_headers(), json={"ids": instrument_ids[:500]})
        dbg["post_status"]=r.status_code
        if r.status_code == 200:
            mp = parse_instruments_response(r.json())
            dbg["mapped"]=len(mp)
            if mp:
                dbg["sample"]=list(mp.items())[:5]
            return mp, dbg
        dbg["body_head"]=r.text[:300]
    except Exception as e:
        dbg["post_error"]=f"{type(e).__name__}: {e}"

    return {}, dbg

async def fetch_google_news_rss(client: httpx.AsyncClient, ticker: str) -> List[NewsItem]:
    q = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = await client.get(url, headers={"accept":"application/rss+xml, text/xml;q=0.9,*/*;q=0.8", "user-agent": DEFAULT_UA})
    if r.status_code != 200:
        return []
    raw = parse_rss_or_atom(r.text)
    out=[]
    for it in raw[:NEWS_PER_TICKER*3]:
        title=safe_text(it.get("title")); link=safe_text(it.get("link"))
        if not title or not link: 
            continue
        out.append(NewsItem(ticker=ticker, title=title, link=link,
                            source=safe_text(it.get("source")) or "Google News",
                            published_raw=safe_text(it.get("published_raw")) or "",
                            resume=resume_news(title)))
    ded = dedupe_by_key([x.__dict__ for x in out], "link")
    return [NewsItem(**x) for x in ded[:NEWS_PER_TICKER]]

async def fetch_sec_for_ticker(client: httpx.AsyncClient, ticker: str) -> List[SecItem]:
    cik_map={}
    if CIK_MAP_JSON:
        try: cik_map=json.loads(CIK_MAP_JSON)
        except Exception: cik_map={}
    cik=(cik_map.get(ticker) or "").strip()
    if cik:
        cik=cik.zfill(10)
        url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&owner=exclude&count=40&output=atom"
        r=await client.get(url, headers={"accept":"application/atom+xml, application/xml;q=0.9,*/*;q=0.8", "user-agent": SEC_UA})
        if r.status_code != 200:
            return []
        raw=parse_rss_or_atom(r.text)
        out=[]
        for it in raw[:SEC_PER_TICKER*3]:
            title=safe_text(it.get("title")); link=safe_text(it.get("link")); filed=safe_text(it.get("published_raw")) or ""
            if not title or not link: 
                continue
            m=re.search(r"\b(8-K|10-Q|10-K|S-1|F-1)\b", title, re.I)
            form=(m.group(1).upper() if m else "Filing")
            out.append(SecItem(ticker=ticker, form=form, title=title, link=link, filed=filed, resume=resume_sec(form)))
        ded=dedupe_by_key([x.__dict__ for x in out], "link")
        return [SecItem(**x) for x in ded[:SEC_PER_TICKER]]
    # fallback link
    link=f"https://www.sec.gov/edgar/search/#/q={quote_plus(ticker)}&sort=desc"
    return [SecItem(ticker=ticker, form="EDGAR", title=f"EDGAR search for {ticker}", link=link, filed="", resume="Click to see latest filings.")]

async def run_daily_refresh() -> Dict[str, Any]:
    started=now_utc_iso()
    sem=asyncio.Semaphore(CONCURRENCY)
    errors=[]
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        pnl, pnl_dbg = await fetch_etoro_pnl(client)
        if pnl is None:
            STATE["tickers"]=[]
            STATE["news_cache"]={}
            STATE["sec_cache"]={}
            STATE["last_run"]=started
            STATE["debug"]={"started": started, "tickers": 0, "news_total": 0, "sec_total": 0, "errors": ["etoro_pnl_failed"], "pnl_debug": pnl_dbg}
            save_state(STATE)
            return STATE["debug"]

        instrument_ids = extract_instrument_ids(pnl)
        inst_map, inst_dbg = await fetch_instrument_map(client, instrument_ids)

        tickers = []
        for iid in instrument_ids:
            sym = inst_map.get(iid)
            if sym and sym not in CRYPTO_EXCLUDE:
                tickers.append(sym)
        tickers = dedupe_list(tickers)

        STATE["tickers"] = tickers

        async def gn(t: str): 
            async with sem: 
                items = await fetch_google_news_rss(client, t)
                return t, [x.__dict__ for x in items]

        async def sec(t: str):
            async with sem:
                items = await fetch_sec_for_ticker(client, t)
                return t, [x.__dict__ for x in items]

        news_cache={}
        sec_cache={}
        try:
            news_results = await asyncio.gather(*[gn(t) for t in tickers])
        except Exception as e:
            news_results=[]
            errors.append(f"news_gather: {type(e).__name__}: {e}")
        try:
            sec_results = await asyncio.gather(*[sec(t) for t in tickers])
        except Exception as e:
            sec_results=[]
            errors.append(f"sec_gather: {type(e).__name__}: {e}")

    for t, items in news_results: news_cache[t]=items
    for t, items in sec_results: sec_cache[t]=items

    STATE["news_cache"]=news_cache
    STATE["sec_cache"]=sec_cache
    STATE["last_run"]=started
    STATE["debug"]={
        "started": started,
        "tickers": len(tickers),
        "news_total": sum(len(v) for v in news_cache.values()),
        "sec_total": sum(len(v) for v in sec_cache.values()),
        "errors": errors[:50],
        "pnl_debug": pnl_dbg,
        "instrument_ids": len(instrument_ids),
        "instrument_lookup": inst_dbg,
        "note": "If tickers=0 but instrument_ids>0, ETORO_INSTRUMENTS_URL is wrong or parsing keys differ.",
    }
    save_state(STATE)
    return STATE["debug"]

app = FastAPI(title=APP_NAME)

@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"

@app.get("/tasks/daily", response_class=JSONResponse)
async def tasks_daily() -> Dict[str, Any]:
    dbg = await run_daily_refresh()
    return {"status": "ok", "debug": dbg}

@app.get("/debug/tickers", response_class=PlainTextResponse)
def debug_tickers() -> str:
    return ",".join(STATE.get("tickers", []) or [])

@app.get("/api/news", response_class=JSONResponse)
def api_news() -> Dict[str, Any]:
    return {"tickers": STATE.get("tickers", []), "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {}), "news_cache": STATE.get("news_cache", {})}

@app.get("/api/sec", response_class=JSONResponse)
def api_sec() -> Dict[str, Any]:
    return {"tickers": STATE.get("tickers", []), "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {}), "sec_cache": STATE.get("sec_cache", {})}

def render_card(t: str) -> str:
    news = (STATE.get("news_cache", {}) or {}).get(t, [])[:NEWS_PER_TICKER]
    sec = (STATE.get("sec_cache", {}) or {}).get(t, [])[:SEC_PER_TICKER]
    def render_items(items, kind):
        if not items:
            return "<div class='muted'>Empty. Run <code>/tasks/daily</code>.</div>"
        html=[]
        for it in items:
            title=html_escape(safe_text(it.get("title")))
            link=html_escape(safe_text(it.get("link")))
            meta=html_escape(safe_text(it.get("source") or it.get("filed") or ""))
            resume=html_escape(safe_text(it.get("resume") or ""))
            badge=""
            if kind=="sec":
                badge=f"<span class='badge'>{html_escape(safe_text(it.get('form') or 'Filing'))}</span>"
            html.append(f"<div class='item'><div class='title'>{badge}<a href='{link}' target='_blank' rel='noopener noreferrer'>{title}</a></div><div class='meta'>{meta}</div><div class='resume'>{resume}</div></div>")
        return "\n".join(html)

    return f"""
    <div class="card">
      <div class="card-h"><div class="ticker">{html_escape(t)}</div><div class="small muted">News + SEC</div></div>
      <div class="grid">
        <div><div class="section-title">News</div>{render_items(news,"news")}</div>
        <div><div class="section-title">SEC</div>{render_items(sec,"sec")}</div>
      </div>
    </div>
    """

@app.get("/", response_class=HTMLResponse)
def dashboard(limit: int = Query(80, ge=1, le=500)) -> str:
    tickers = (STATE.get("tickers") or [])[:limit]
    last_run = STATE.get("last_run") or "never"
    dbg = STATE.get("debug", {}) or {}
    body = "<div class='card'><div class='muted'>No tickers yet. Set ETORO_INSTRUMENTS_URL then run <code>/tasks/daily</code>.</div></div>"
    if tickers:
        body = "\n".join(render_card(t) for t in tickers)
    return f"""
<!doctype html>
<html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html_escape(APP_NAME)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0b0f14;color:#e6edf3}}
a{{color:#7aa2ff;text-decoration:none}} a:hover{{text-decoration:underline}}
.wrap{{max-width:1200px;margin:0 auto;padding:18px}}
.top{{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:14px}}
h1{{font-size:18px;margin:0}} .meta{{font-size:12px;color:#9fb0c0}}
.pill{{display:inline-block;padding:4px 8px;border:1px solid #1f2a37;border-radius:999px;font-size:12px}}
.row{{display:flex;gap:10px;flex-wrap:wrap}}
.card{{background:#0f1621;border:1px solid #1f2a37;border-radius:16px;padding:14px;margin:12px 0;box-shadow:0 8px 24px rgba(0,0,0,.25)}}
.card-h{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}}
.ticker{{font-size:18px;font-weight:700}} .small{{font-size:12px}} .muted{{color:#9fb0c0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
.section-title{{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#9fb0c0;margin:4px 0 10px}}
.item{{padding:10px;border:1px solid #1f2a37;border-radius:12px;margin-bottom:10px;background:#0b111a}}
.title{{font-size:13px;line-height:1.25;margin-bottom:6px}} .resume{{font-size:12px;margin-top:6px;color:#c6d2dd}}
code{{background:#0b111a;border:1px solid #1f2a37;padding:2px 6px;border-radius:8px;color:#cfe2ff}}
.badge{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid #314055;color:#cfe2ff;margin-right:6px}}
details{{margin-top:10px}} pre{{white-space:pre-wrap;word-break:break-word;background:#0b111a;border:1px solid #1f2a37;padding:10px;border-radius:12px;color:#9fb0c0;font-size:12px}}
</style></head>
<body><div class="wrap">
<div class="top">
  <div><h1>{html_escape(APP_NAME)}</h1><div class="meta">Last refresh: {html_escape(last_run)} · Tickers: {len(tickers)}</div></div>
  <div class="row">
    <span class="pill"><a href="/tasks/daily" target="_blank" rel="noopener noreferrer">Run /tasks/daily</a></span>
    <span class="pill"><a href="/debug/tickers" target="_blank" rel="noopener noreferrer">/debug/tickers</a></span>
  </div>
</div>
<div class="meta">Debug: news_total={dbg.get("news_total","?")} · sec_total={dbg.get("sec_total","?")} · errors={len(dbg.get("errors",[]))}</div>
{body}
<details><summary class="meta">Debug details</summary><pre>{html_escape(json.dumps(dbg, ensure_ascii=False, indent=2))}</pre></details>
</div></body></html>
"""

# end
