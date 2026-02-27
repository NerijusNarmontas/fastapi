# main.py
# Paste this as your ONLY main.py.
# Railway startCommand should be: hypercorn main:app --bind "0.0.0.0:$PORT"
#
# What you MUST set in Railway Variables:
#   TICKERS = EQT,CIVI,DVN,SM,NOG,CCJ,LEU,SMR,OKLO,EPD,WMB,... (your stocks)
# Optional:
#   CRYPTO_EXCLUDE = W (comma-separated tickers you want to ignore)
#   NEWS_PER_TICKER = 6
#   SEC_PER_TICKER = 6
#   SEC_UA = "YourName AppName (email@domain.com)"   # important for SEC
#   DEFAULT_UA = "...browser UA..."                  # helps Google News RSS
#   CIK_MAP = {"AAPL":"0000320193","MSFT":"0000789019"}  # optional for richer SEC per ticker
#
# After deploy:
#   1) open /tasks/daily once
#   2) open /  (dashboard)
#   3) open /api/news if you want raw JSON

import os
import re
import json
import asyncio
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

# --- Core inputs ---
TICKERS_ENV = os.getenv("TICKERS", "").strip()
CRYPTO_EXCLUDE = set(
    s.strip().upper() for s in os.getenv("CRYPTO_EXCLUDE", "W").split(",") if s.strip()
)

# --- Limits / perf ---
NEWS_PER_TICKER = int(os.getenv("NEWS_PER_TICKER", "6"))
SEC_PER_TICKER = int(os.getenv("SEC_PER_TICKER", "6"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))

# --- User-Agent headers (important) ---
DEFAULT_UA = os.getenv(
    "DEFAULT_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
)
SEC_UA = os.getenv(
    "SEC_UA",
    "NerijusNarmontas fastapi-investing-dashboard (contact: nerijus@example.com)",
)

CIK_MAP_JSON = os.getenv("CIK_MAP", "").strip()


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
STATE.setdefault("news_cache", {})  # {ticker: [items]}
STATE.setdefault("sec_cache", {})   # {ticker: [items]}
STATE.setdefault("last_run", None)
STATE.setdefault("debug", {})


def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)
    return t


def safe_text(x: Optional[str]) -> str:
    return (x or "").replace("\x00", "").strip()


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def dedupe_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        v = (it.get(key) or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(it)
    return out


def parse_rss_or_atom(xml_text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    # RSS2: channel/item
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
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = safe_text(entry.findtext("{http://www.w3.org/2005/Atom}title"))
        updated = safe_text(entry.findtext("{http://www.w3.org/2005/Atom}updated"))
        link = ""
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        if link_el is not None and "href" in link_el.attrib:
            link = safe_text(link_el.attrib["href"])
        items.append({"title": title, "link": link, "published_raw": updated, "source": ""})
    return items


def resume_news(title: str) -> str:
    t = (title or "").lower()
    tags = []
    if any(k in t for k in ["earnings", "q1", "q2", "q3", "q4", "guidance", "revenue", "eps"]):
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
    if any(k in t for k in ["permit", "approval", "license", "regulator"]):
        tags.append("Approvals / regulatory")
    return " | ".join(tags) if tags else "Headline update (skim if position is meaningful)."


def resume_sec(form: str, title: str) -> str:
    f = (form or "").upper().strip()
    if f == "8-K":
        return "Material event. Check for earnings release, financing, M&A, leadership changes, major contracts."
    if f == "10-Q":
        return "Quarterly report. Focus on margins, cash burn/FCF, guidance, balance sheet."
    if f == "10-K":
        return "Annual report. Risks/business changes; liquidity, segment performance."
    if f in ("S-1", "F-1"):
        return "Registration/IPO. Dilution risk; read use-of-proceeds and offering terms."
    if f.startswith("S-") or f.startswith("F-"):
        return "Registration statement. Often issuance/resale; dilution risk."
    if f.startswith("424B"):
        return "Prospectus supplement. Financing details; pricing and dilution."
    if f in ("SC 13D", "SC13D"):
        return "Activist/large holder filing. Can be catalytic; check intent and stake change."
    if f in ("SC 13G", "SC13G"):
        return "Passive holder filing. Useful sentiment signal."
    if f in ("DEF 14A", "DEFA14A"):
        return "Proxy. Governance/compensation; can signal strategic shifts."
    return "Filing update. Open link and skim cover + key sections."


@dataclass
class NewsItem:
    ticker: str
    title: str
    link: str
    source: str
    published_raw: str
    resume: str


@dataclass
class SecItem:
    ticker: str
    form: str
    title: str
    link: str
    filed: str
    resume: str


def get_tickers() -> List[str]:
    base: List[str] = []
    # Prefer STATE tickers if already populated
    if isinstance(STATE.get("tickers"), list) and STATE["tickers"]:
        base = [normalize_ticker(x) for x in STATE["tickers"]]
    elif TICKERS_ENV:
        base = [normalize_ticker(x) for x in TICKERS_ENV.split(",")]

    out: List[str] = []
    for t in base:
        if not t:
            continue
        if t in CRYPTO_EXCLUDE:
            continue
        out.append(t)

    # de-dupe stable
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


async def fetch_google_news_rss(client: httpx.AsyncClient, ticker: str) -> List[NewsItem]:
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
    raw = parse_rss_or_atom(r.text)
    out: List[NewsItem] = []
    for it in raw[: NEWS_PER_TICKER * 3]:
        title = safe_text(it.get("title"))
        link = safe_text(it.get("link"))
        if not title or not link:
            continue
        out.append(
            NewsItem(
                ticker=ticker,
                title=title,
                link=link,
                source=safe_text(it.get("source")) or "Google News",
                published_raw=safe_text(it.get("published_raw")) or "",
                resume=resume_news(title),
            )
        )
    dedup = dedupe_by_key([x.__dict__ for x in out], "link")
    return [NewsItem(**x) for x in dedup[:NEWS_PER_TICKER]]


async def fetch_sec_for_ticker(client: httpx.AsyncClient, ticker: str) -> List[SecItem]:
    cik_map: Dict[str, str] = {}
    if CIK_MAP_JSON:
        try:
            cik_map = json.loads(CIK_MAP_JSON)
        except Exception:
            cik_map = {}

    cik = (cik_map.get(ticker) or "").strip()
    if cik:
        cik = cik.zfill(10)
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}&owner=exclude&count=40&output=atom"
        )
        r = await client.get(
            url,
            headers={
                "accept": "application/atom+xml, application/xml;q=0.9,*/*;q=0.8",
                "user-agent": SEC_UA,
            },
        )
        if r.status_code != 200:
            return []
        raw = parse_rss_or_atom(r.text)

        out: List[SecItem] = []
        for it in raw[: SEC_PER_TICKER * 3]:
            title = safe_text(it.get("title"))
            link = safe_text(it.get("link"))
            filed = safe_text(it.get("published_raw")) or ""
            if not title or not link:
                continue

            form = ""
            m = re.search(r"\b(8-K|10-Q|10-K|S-1|F-1|424B\d*|DEF\s*14A|SC\s*13D|SC\s*13G)\b", title, re.I)
            if m:
                form = m.group(1).upper().replace("  ", " ").strip()
                if form.startswith("SC"):
                    form = form.replace("SC", "SC ").replace("  ", " ").strip()

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
        dedup = dedupe_by_key([x.__dict__ for x in out], "link")
        return [SecItem(**x) for x in dedup[:SEC_PER_TICKER]]

    # fallback: still give you a clickable EDGAR search (no parsing)
    search_link = f"https://www.sec.gov/edgar/search/#/q={quote_plus(ticker)}&sort=desc"
    return [
        SecItem(
            ticker=ticker,
            form="EDGAR",
            title=f"EDGAR search for {ticker}",
            link=search_link,
            filed="",
            resume="Click to see latest filings. For richer per-form lists, set CIK_MAP (ticker→CIK).",
        )
    ]


async def run_daily_refresh() -> Dict[str, Any]:
    tickers = get_tickers()
    started = now_utc_iso()
    sem = asyncio.Semaphore(CONCURRENCY)

    news_cache: Dict[str, Any] = {}
    sec_cache: Dict[str, Any] = {}
    errors: List[str] = []

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:

        async def gn(t: str) -> Tuple[str, List[Dict[str, Any]]]:
            async with sem:
                items = await fetch_google_news_rss(client, t)
                return t, [x.__dict__ for x in items]

        async def sec(t: str) -> Tuple[str, List[Dict[str, Any]]]:
            async with sem:
                items = await fetch_sec_for_ticker(client, t)
                return t, [x.__dict__ for x in items]

        # run
        tasks_news = [gn(t) for t in tickers]
        tasks_sec = [sec(t) for t in tickers]

        try:
            news_results = await asyncio.gather(*tasks_news)
        except Exception as e:
            news_results = []
            errors.append(f"news_gather: {type(e).__name__}")

        try:
            sec_results = await asyncio.gather(*tasks_sec)
        except Exception as e:
            sec_results = []
            errors.append(f"sec_gather: {type(e).__name__}")

    for t, items in news_results:
        news_cache[t] = items

    for t, items in sec_results:
        sec_cache[t] = items

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
        "note": "If tickers=0, set Railway env var TICKERS.",
    }
    save_state(STATE)
    return STATE["debug"]


app = FastAPI(title=APP_NAME)
import os
import httpx
from fastapi.responses import PlainTextResponse

ETORO_API_KEY = os.getenv("ETORO_API_KEY", "").strip()
ETORO_USER_KEY = os.getenv("ETORO_USER_KEY", "").strip()

# If you already had these in your old code, keep the same endpoints you used.
ETORO_POSITIONS_URL = os.getenv(
    "ETORO_POSITIONS_URL",
    "https://api.etorostatic.com/sapi/trading/positions",  # placeholder: use YOUR working endpoint
)

@app.get("/debug/tickers", response_class=PlainTextResponse)
async def debug_tickers() -> str:
    """
    Returns: comma-separated STOCK tickers (crypto excluded) ready for Railway TICKERS env var.
    This uses the same idea as before: fetch positions -> map to symbols -> filter crypto collisions.
    """
    if not ETORO_API_KEY or not ETORO_USER_KEY:
        return "Missing ETORO_API_KEY / ETORO_USER_KEY in env."

    headers = {
        "accept": "application/json",
        "user-agent": DEFAULT_UA,
        "authorization": f"Bearer {ETORO_API_KEY}",
        "x-etoro-user-key": ETORO_USER_KEY,
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(ETORO_POSITIONS_URL, headers=headers)
        if r.status_code != 200:
            return f"eToro positions fetch failed: {r.status_code}\n{r.text[:300]}"

        data = r.json()

    # ---- YOU MUST ADAPT THIS PART to your actual eToro JSON shape ----
    # In your earlier logs you had "instrumentID" and later you mapped to "ticker".
    # If your JSON already contains symbol/ticker, prefer that.
    positions = data if isinstance(data, list) else data.get("positions") or data.get("data") or []

    raw_symbols = []
    for p in positions:
        sym = p.get("ticker") or p.get("symbol")
        if sym:
            raw_symbols.append(normalize_ticker(sym))

    # If you don't have symbol/ticker in positions, you need the instrumentID->ticker map like before.
    # (That part depends on your existing mapping code / endpoint you already had.)

    # crypto collisions exclude (e.g. W)
    tickers = sorted({s for s in raw_symbols if s and s not in CRYPTO_EXCLUDE})

    return ",".join(tickers)


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/tasks/daily", response_class=JSONResponse)
async def tasks_daily() -> Dict[str, Any]:
    dbg = await run_daily_refresh()
    return {"status": "ok", "debug": dbg}


@app.get("/api/news", response_class=JSONResponse)
def api_news(ticker: Optional[str] = None) -> Dict[str, Any]:
    tickers = get_tickers()
    cache = STATE.get("news_cache", {}) or {}
    if ticker:
        t = normalize_ticker(ticker)
        return {"ticker": t, "items": cache.get(t, []), "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {})}
    return {"tickers": tickers, "news_cache": cache, "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {})}


@app.get("/api/sec", response_class=JSONResponse)
def api_sec(ticker: Optional[str] = None) -> Dict[str, Any]:
    tickers = get_tickers()
    cache = STATE.get("sec_cache", {}) or {}
    if ticker:
        t = normalize_ticker(ticker)
        return {"ticker": t, "items": cache.get(t, []), "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {})}
    return {"tickers": tickers, "sec_cache": cache, "last_run": STATE.get("last_run"), "debug": STATE.get("debug", {})}


def render_ticker_card(t: str) -> str:
    news_items = (STATE.get("news_cache", {}) or {}).get(t, [])[:NEWS_PER_TICKER]
    sec_items = (STATE.get("sec_cache", {}) or {}).get(t, [])[:SEC_PER_TICKER]

    def news_html() -> str:
        if not news_items:
            return "<div class='muted'>No news cached yet. Run <code>/tasks/daily</code>.</div>"
        rows = []
        for it in news_items:
            title = html_escape(safe_text(it.get("title")))
            link = html_escape(safe_text(it.get("link")))
            source = html_escape(safe_text(it.get("source")) or "Source")
            pub = html_escape(safe_text(it.get("published_raw")) or "")
            resume = html_escape(safe_text(it.get("resume")) or "")
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
            return "<div class='muted'>No SEC cached yet. Run <code>/tasks/daily</code>.</div>"
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
def dashboard(limit: int = Query(60, ge=1, le=500)) -> str:
    tickers = get_tickers()[:limit]
    last_run = STATE.get("last_run") or "never"
    dbg = STATE.get("debug", {}) or {}

    if not tickers:
        body = """
        <div class="card">
          <div class="muted">
            No tickers found.<br/>
            Set Railway Variable <code>TICKERS</code> like: <code>EQT,SM,CCJ,LEU</code><br/>
            Then run <code>/tasks/daily</code>.
          </div>
        </div>
        """
    else:
        body = "\n".join(render_ticker_card(t) for t in tickers)

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

    {body}

    <div class="meta" style="margin-top:18px;">
      If SEC looks too thin: set <code>CIK_MAP</code> (ticker→CIK). SEC also needs <code>SEC_UA</code>.
    </div>
  </div>
</body>
</html>
"""


# End of file.
