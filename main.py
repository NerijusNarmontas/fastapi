# main.py
# FastAPI dashboard with:
# - News links per ticker (Google News RSS + optional Bing News RSS style fallback)
# - SEC filings links per ticker (EDGAR Atom feed; optional CIK-based feed if you provide mapping)
# - Short “resume” bullets (heuristic summaries; no full-text scraping)
#
# NOTES
# - Google News RSS can block generic bots: we send a real User-Agent.
# - SEC requires a descriptive User-Agent header.
# - This file is self-contained; you can later plug your eToro/stooq pipeline back in.
#
# Run locally:
#   uvicorn main:app --reload --port 8000
#
# Railway:
#   Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

import os
import re
import json
import math
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ----------------------------
# Config
# ----------------------------

APP_NAME = "Investing Dashboard"
STATE_PATH = os.getenv("STATE_PATH", "/tmp/investing_agent_state.json")

# If you already have a ticker universe (from eToro positions), you can pass it via env:
# TICKERS="EQT,SM,CCJ,LEU,SMR,OKLO,EPD"
TICKERS_ENV = os.getenv("TICKERS", "").strip()

# Exclude crypto symbols that collide with stocks (example from your note: W=Wormhole)
CRYPTO_EXCLUDE = set(
    s.strip().upper()
    for s in os.getenv("CRYPTO_EXCLUDE", "W").split(",")
    if s.strip()
)

# Limits
NEWS_PER_TICKER = int(os.getenv("NEWS_PER_TICKER", "6"))
SEC_PER_TICKER = int(os.getenv("SEC_PER_TICKER", "6"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))

# User-Agent headers (important)
DEFAULT_UA = os.getenv(
    "DEFAULT_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
)

# SEC asks for: Company Name + email/phone (or at least something descriptive)
SEC_UA = os.getenv(
    "SEC_UA",
    "NerijusNarmontas fastapi-investing-dashboard (contact: nerijus@example.com)",
)

# If you want better SEC results (CIK-based), you can provide a simple map:
# CIK_MAP='{"AAPL":"0000320193","MSFT":"0000789019"}'
CIK_MAP_JSON = os.getenv("CIK_MAP", "").strip()

# ----------------------------
# State
# ----------------------------

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

# Initialize expected keys
STATE.setdefault("tickers", [])
STATE.setdefault("news_cache", {})     # {ticker: [items]}
STATE.setdefault("sec_cache", {})      # {ticker: [items]}
STATE.setdefault("last_run", None)
STATE.setdefault("debug", {})

# ----------------------------
# Utilities
# ----------------------------

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    # Basic cleanup
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)
    return t

def dedupe_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        v = (it.get(key) or "").strip()
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(it)
    return out

def safe_text(x: Optional[str]) -> str:
    return (x or "").replace("\x00", "").strip()

def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )

def iso_to_localish(iso_str: str) -> str:
    # Keep it simple: show ISO date if parseable
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso_str

# ----------------------------
# RSS / Atom parsing
# ----------------------------

def parse_rss_items(xml_text: str) -> List[Dict[str, Any]]:
    """
    Works for RSS 2.0 and some Atom-like feeds.
    Returns list of dict: title, link, published, source
    """
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    # RSS: channel/item
    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item"):
            title = safe_text(item.findtext("title"))
            link = safe_text(item.findtext("link"))
            pub = safe_text(item.findtext("pubDate"))
            source = safe_text(item.findtext("source"))
            items.append({"title": title, "link": link, "published_raw": pub, "source": source})
        return items

    # Atom: entry
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = safe_text(entry.findtext("{http://www.w3.org/2005/Atom}title"))
        updated = safe_text(entry.findtext("{http://www.w3.org/2005/Atom}updated"))
        # link can be in <link href="..."/>
        link = ""
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        if link_el is not None and "href" in link_el.attrib:
            link = safe_text(link_el.attrib["href"])
        items.append({"title": title, "link": link, "published_raw": updated, "source": ""})
    return items

# ----------------------------
# News sources
# ----------------------------

@dataclass
class NewsItem:
    ticker: str
    title: str
    link: str
    source: str
    published: str
    resume: str

def resume_news(title: str) -> str:
    """
    Heuristic “resume” for key events.
    No full-text reading; just a short classification.
    """
    t = (title or "").lower()
    tags = []
    if any(k in t for k in ["earnings", "q1", "q2", "q3", "q4", "guidance", "revenue", "eps"]):
        tags.append("Earnings / guidance")
    if any(k in t for k in ["sec", "8-k", "10-q", "10-k", "s-1", "f-1", "prospectus"]):
        tags.append("Regulatory / filing")
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
    if any(k in t for k in ["production", "plant", "factory", "permit", "approval", "license"]):
        tags.append("Ops / approvals")

    if not tags:
        return "Headline update (read if position is large or volatility is high)."
    return " | ".join(tags)

async def fetch_google_news_rss(client: httpx.AsyncClient, ticker: str) -> List[NewsItem]:
    # Google News RSS query: ticker + stock
    q = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    try:
        r = await client.get(
            url,
            headers={
                "accept": "application/rss+xml, text/xml;q=0.9,*/*;q=0.8",
                "user-agent": DEFAULT_UA,
            },
        )
        if r.status_code != 200:
            return []
        raw_items = parse_rss_items(r.text)
    except Exception:
        return []

    out: List[NewsItem] = []
    for it in raw_items[: NEWS_PER_TICKER * 2]:
        title = safe_text(it.get("title"))
        link = safe_text(it.get("link"))
        if not title or not link:
            continue
        source = safe_text(it.get("source")) or "Google News"
        published = safe_text(it.get("published_raw")) or ""
        out.append(
            NewsItem(
                ticker=ticker,
                title=title,
                link=link,
                source=source,
                published=published,
                resume=resume_news(title),
            )
        )
    # Dedupe by link and cut
    deduped = dedupe_by_key([x.__dict__ for x in out], "link")
    return [NewsItem(**x) for x in deduped[:NEWS_PER_TICKER]]

# Optional fallback: a second RSS-like source is tricky without paid APIs.
# Keep it simple: Google News is usually enough if UA is correct.

# ----------------------------
# SEC filings (links + short resumes)
# ----------------------------

@dataclass
class SecItem:
    ticker: str
    form: str
    title: str
    link: str
    filed: str
    resume: str

def resume_sec(form: str, title: str) -> str:
    f = (form or "").upper().strip()
    # Fast, traditional “what matters” mapping
    if f == "8-K":
        return "Material event. Check for earnings release, financing, M&A, leadership changes, or material agreements."
    if f in ("10-Q",):
        return "Quarterly report. Look for margins, cash burn/FCF, guidance, and risk updates."
    if f in ("10-K",):
        return "Annual report. Business/risk overhaul; focus on liquidity, going-concern, segment performance."
    if f in ("S-1", "F-1"):
        return "IPO/registration. Dilution & selling pressure risk; read use-of-proceeds and underwriting."
    if f.startswith("S-") or f.startswith("F-"):
        return "Registration statement. Often relates to issuance/resales; dilution risk."
    if f in ("424B", "424B2", "424B3", "424B4", "424B5"):
        return "Prospectus/prospectus supplement. Usually financing details; check pricing and dilution."
    if f in ("SC 13D", "SC13D", "13D"):
        return "Activist/large holder filing. Can move the stock; watch intent and ownership changes."
    if f in ("SC 13G", "SC13G", "13G"):
        return "Passive large holder filing. Useful sentiment signal; usually less catalytic than 13D."
    if f in ("DEF 14A", "DEFA14A"):
        return "Proxy statement. Compensation, votes, governance; can hint at strategic shifts."
    return "Filing update. Open link and skim the cover + key sections."

async def fetch_sec_atom_by_ticker(client: httpx.AsyncClient, ticker: str) -> List[SecItem]:
    """
    SEC provides an Atom feed by 'CIK' reliably.
    Ticker-based atom feeds exist for some endpoints, but CIK is safer.
    If you don't provide CIK_MAP, we fall back to a broad EDGAR search link list (no atom parsing).
    """
    cik_map: Dict[str, str] = {}
    if CIK_MAP_JSON:
        try:
            cik_map = json.loads(CIK_MAP_JSON)
        except Exception:
            cik_map = {}

    cik = (cik_map.get(ticker) or "").strip()
    if cik:
        cik = cik.zfill(10)
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=40&output=atom"
        try:
            r = await client.get(
                url,
                headers={
                    "accept": "application/atom+xml, application/xml;q=0.9,*/*;q=0.8",
                    "user-agent": SEC_UA,
                },
            )
            if r.status_code != 200:
                return []
            raw_items = parse_rss_items(r.text)  # Atom entries
        except Exception:
            return []

        out: List[SecItem] = []
        for it in raw_items[: SEC_PER_TICKER * 2]:
            title = safe_text(it.get("title"))
            link = safe_text(it.get("link"))
            filed = safe_text(it.get("published_raw")) or ""
            # Atom entry title often contains form + company name; try to pull the form
            # Example patterns vary; keep it conservative.
            form = ""
            m = re.search(r"\b(8-K|10-Q|10-K|S-1|F-1|424B\d*|DEF\s*14A|SC\s*13D|SC\s*13G)\b", title, re.I)
            if m:
                form = m.group(1).replace(" ", "").upper()
                if form == "SC13D":
                    form = "SC 13D"
                if form == "SC13G":
                    form = "SC 13G"
                if form.startswith("DEF"):
                    form = "DEF 14A"
            out.append(
                SecItem(
                    ticker=ticker,
                    form=form or "Filing",
                    title=title,
                    link=link,
                    filed=filed,
                    resume=resume_sec(form or "", title),
                )
            )
        deduped = dedupe_by_key([x.__dict__ for x in out], "link")
        return [SecItem(**x) for x in deduped[:SEC_PER_TICKER]]

    # Fallback: provide EDGAR search link without feed parsing
    # (Still gives you clickable “SEC filings” even without CIK mapping.)
    search_link = f"https://www.sec.gov/edgar/search/#/q={quote_plus(ticker)}&category=custom&forms=8-K%252C10-Q%252C10-K%252CS-1%252CF-1&sort=desc"
    return [
        SecItem(
            ticker=ticker,
            form="EDGAR",
            title=f"EDGAR search for {ticker} (8-K/10-Q/10-K/S-1/F-1)",
            link=search_link,
            filed="",
            resume="Click to see latest filings. If you want richer per-form lists, provide CIK_MAP env var.",
        )
    ]

# ----------------------------
# Tickers (plug your eToro pipeline here later)
# ----------------------------

def get_tickers() -> List[str]:
    # Priority:
    # 1) Previously saved tickers in STATE (from /tasks/daily or your pipeline)
    # 2) TICKERS env
    base: List[str] = []
    if isinstance(STATE.get("tickers"), list) and STATE["tickers"]:
        base = [normalize_ticker(x) for x in STATE["tickers"]]
    elif TICKERS_ENV:
        base = [normalize_ticker(x) for x in TICKERS_ENV.split(",")]

    # Filter empties + crypto collisions
    out = []
    for t in base:
        if not t:
            continue
        if t in CRYPTO_EXCLUDE:
            continue
        out.append(t)
    # De-dupe stable
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

# ----------------------------
# Daily task
# ----------------------------

async def run_daily_refresh() -> Dict[str, Any]:
    tickers = get_tickers()
    started = now_utc_iso()

    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:

        async def guarded_news(t: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
            async with sem:
                try:
                    items = await fetch_google_news_rss(client, t)
                    return t, [x.__dict__ for x in items], None
                except Exception as e:
                    return t, [], f"news_error: {type(e).__name__}"

        async def guarded_sec(t: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
            async with sem:
                try:
                    items = await fetch_sec_atom_by_ticker(client, t)
                    return t, [x.__dict__ for x in items], None
                except Exception as e:
                    return t, [], f"sec_error: {type(e).__name__}"

        news_tasks = [guarded_news(t) for t in tickers]
        sec_tasks = [guarded_sec(t) for t in tickers]

        news_results = await asyncio.gather(*news_tasks)
        sec_results = await asyncio.gather(*sec_tasks)

    news_cache: Dict[str, Any] = {}
    sec_cache: Dict[str, Any] = {}
    errors: List[str] = []

    for t, items, err in news_results:
        news_cache[t] = items
        if err:
            errors.append(f"{t}: {err}")

    for t, items, err in sec_results:
        sec_cache[t] = items
        if err:
            errors.append(f"{t}: {err}")

    STATE["tickers"] = tickers
    STATE["news_cache"] = news_cache
    STATE["sec_cache"] = sec_cache
    STATE["last_run"] = started
    STATE["debug"] = {
        "started": started,
        "tickers": len(tickers),
        "news_total": sum(len(v) for v in news_cache.values()),
        "sec_total": sum(len(v) for v in sec_cache.values()),
        "errors": errors[:50],
    }
    save_state(STATE)
    return STATE["debug"]

# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title=APP_NAME)

@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"

@app.get("/tasks/daily", response_class=JSONResponse)
async def tasks_daily() -> Dict[str, Any]:
    """
    Trigger refresh (news + SEC).
    In production, call this from a scheduler (Railway cron / GitHub Actions / external cron).
    """
    dbg = await run_daily_refresh()
    return {"status": "ok", "debug": dbg}

@app.get("/api/news", response_class=JSONResponse)
def api_news(ticker: Optional[str] = None) -> Dict[str, Any]:
    tickers = get_tickers()
    cache = STATE.get("news_cache", {}) or {}
    if ticker:
        t = normalize_ticker(ticker)
        return {"ticker": t, "items": cache.get(t, []), "last_run": STATE.get("last_run")}
    return {"tickers": tickers, "news_cache": cache, "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {})}

@app.get("/api/sec", response_class=JSONResponse)
def api_sec(ticker: Optional[str] = None) -> Dict[str, Any]:
    tickers = get_tickers()
    cache = STATE.get("sec_cache", {}) or {}
    if ticker:
        t = normalize_ticker(ticker)
        return {"ticker": t, "items": cache.get(t, []), "last_run": STATE.get("last_run")}
    return {"tickers": tickers, "sec_cache": cache, "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {})}

# ----------------------------
# Dashboard HTML
# ----------------------------

def render_cards_for_ticker(t: str) -> str:
    news_items = (STATE.get("news_cache", {}) or {}).get(t, [])[:NEWS_PER_TICKER]
    sec_items = (STATE.get("sec_cache", {}) or {}).get(t, [])[:SEC_PER_TICKER]

    def news_html() -> str:
        if not news_items:
            return "<div class='muted'>No news cached. Run <code>/tasks/daily</code>.</div>"
        rows = []
        for it in news_items:
            title = html_escape(safe_text(it.get("title")))
            link = html_escape(safe_text(it.get("link")))
            source = html_escape(safe_text(it.get("source")) or "Source")
            pub = html_escape(safe_text(it.get("published_raw")))
            resume = html_escape(safe_text(it.get("resume")))
            rows.append(
                f"""
                <div class="item">
                  <div class="title"><a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></div>
                  <div class="meta">{source} · {pub}</div>
                  <div class="resume">{resume}</div>
                </div>
                """
            )
        return "\n".join(rows)

    def sec_html() -> str:
        if not sec_items:
            return "<div class='muted'>No SEC cached. Run <code>/tasks/daily</code>. For richer lists, set <code>CIK_MAP</code>.</div>"
        rows = []
        for it in sec_items:
            title = html_escape(safe_text(it.get("title")))
            link = html_escape(safe_text(it.get("link")))
            form = html_escape(safe_text(it.get("form")) or "Filing")
            filed = html_escape(safe_text(it.get("filed")) or "")
            resume = html_escape(safe_text(it.get("resume")) or "")
            rows.append(
                f"""
                <div class="item">
                  <div class="title"><span class="badge">{form}</span>
                    <a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>
                  </div>
                  <div class="meta">{filed}</div>
                  <div class="resume">{resume}</div>
                </div>
                """
            )
        return "\n".join(rows)

    return f"""
    <div class="card">
      <div class="card-h">
        <div class="ticker">{html_escape(t)}</div>
        <div class="small muted">News + SEC (resumes only)</div>
      </div>
      <div class="grid">
        <div>
          <div class="section-title">News (free)</div>
          {news_html()}
        </div>
        <div>
          <div class="section-title">SEC filings</div>
          {sec_html()}
        </div>
      </div>
    </div>
    """

@app.get("/", response_class=HTMLResponse)
def dashboard(limit: int = Query(40, ge=1, le=300)) -> str:
    tickers = get_tickers()[:limit]
    last_run = STATE.get("last_run") or "never"
    dbg = STATE.get("debug", {}) or {}

    cards = "\n".join(render_cards_for_ticker(t) for t in tickers) if tickers else (
        "<div class='muted'>No tickers found. Set <code>TICKERS</code> env or run your pipeline to populate STATE['tickers'].</div>"
    )

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html_escape(APP_NAME)}</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 0; background: #0b0f14; color: #e6edf3; }}
    a {{ color: #7aa2ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 18px; }}
    .top {{ display:flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    h1 {{ font-size: 18px; margin: 0; letter-spacing: 0.2px; }}
    .meta {{ font-size: 12px; color: #9fb0c0; }}
    .pill {{ display:inline-block; padding: 4px 8px; border: 1px solid #1f2a37; border-radius: 999px; font-size: 12px; color: #9fb0c0; }}
    .row {{ display:flex; gap: 10px; flex-wrap: wrap; }}
    .card {{ background: #0f1621; border: 1px solid #1f2a37; border-radius: 16px; padding: 14px; margin: 12px 0; box-shadow: 0 8px 24px rgba(0,0,0,.25); }}
    .card-h {{ display:flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }}
    .ticker {{ font-size: 18px; font-weight: 700; }}
    .small {{ font-size: 12px; }}
    .muted {{ color: #9fb0c0; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .section-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: .12em; color: #9fb0c0; margin: 4px 0 10px; }}
    .item {{ padding: 10px; border: 1px solid #1f2a37; border-radius: 12px; margin-bottom: 10px; background: #0b111a; }}
    .title {{ font-size: 13px; line-height: 1.25; margin-bottom: 6px; }}
    .meta {{ font-size: 12px; color: #9fb0c0; }}
    .resume {{ font-size: 12px; margin-top: 6px; color: #c6d2dd; }}
    code {{ background: #0b111a; border: 1px solid #1f2a37; padding: 2px 6px; border-radius: 8px; color: #cfe2ff; }}
    .badge {{ display:inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid #314055; color: #cfe2ff; margin-right: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <h1>{html_escape(APP_NAME)}</h1>
        <div class="meta">Last refresh: {html_escape(last_run)} · Tickers: {len(tickers)}</div>
      </div>
      <div class="row">
        <span class="pill"><a href="/tasks/daily" target="_blank" rel="noopener noreferrer">Run /tasks/daily</a></span>
        <span class="pill"><a href="/api/news" target="_blank" rel="noopener noreferrer">/api/news</a></span>
        <span class="pill"><a href="/api/sec" target="_blank" rel="noopener noreferrer">/api/sec</a></span>
      </div>
    </div>

    <div class="meta">
      Debug: news_total={dbg.get("news_total","?")} · sec_total={dbg.get("sec_total","?")} · errors={len(dbg.get("errors",[]))}
    </div>

    {cards}

    <div class="meta" style="margin-top:18px;">
      Tip: For richer SEC lists per ticker, set env <code>CIK_MAP</code> (ticker→CIK). Example:
      <code>{{"AAPL":"0000320193","MSFT":"0000789019"}}</code>
    </div>
  </div>
</body>
</html>
"""

# ----------------------------
# Small helpers for you later
# ----------------------------
# You can replace get_tickers() by your eToro positions ticker extraction,
# then set STATE["tickers"] = those tickers inside /tasks/daily before refresh.
