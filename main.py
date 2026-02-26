# main.py
# Minimal “it works” pipeline: fetch news (Google News RSS) + US filings (SEC EDGAR) + telemetry.
# No paid APIs required. Designed to prevent silent "news_fetched=0" / "filings_fetched=0".

from __future__ import annotations

import os
import time
import json
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
import feedparser


# ----------------------------
# Telemetry / utilities
# ----------------------------

@dataclass
class FetchResult:
    symbol: str
    provider: str
    kind: str                 # "news" or "filings"
    status: str               # ok|empty|auth_missing|forbidden|rate_limited|timeout|parse_error|not_supported|error
    http_code: Optional[int] = None
    elapsed_ms: Optional[int] = None
    items: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = int((time.perf_counter() - self.t0) * 1000)


def safe_err(e: Exception) -> str:
    s = "".join(traceback.format_exception_only(type(e), e)).strip()
    return s[:400]


# ----------------------------
# News provider: Google News RSS (no API key)
# ----------------------------

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def fetch_news_google_rss(
    symbol: str,
    company_name: Optional[str] = None,
    max_items: int = 5,
    timeout: int = 12,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """
    Returns (http_status, items)
    Each item: {published_at, source, title, url}
    """
    # Ticker-only queries can be noisy. If you have company names, include them.
    if company_name:
        q = f'("{symbol}" OR "{company_name}") (stock OR shares OR earnings OR guidance)'
    else:
        q = f'"{symbol}" stock'

    url = GOOGLE_NEWS_RSS.format(query=quote_plus(q))

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    status = r.status_code
    if status != 200:
        return status, []

    feed = feedparser.parse(r.text)
    out: List[Dict[str, Any]] = []

    for e in feed.entries[: max_items * 3]:
        title = getattr(e, "title", None)
        link = getattr(e, "link", None)
        published = getattr(e, "published", None) or getattr(e, "updated", None)

        source = None
        if hasattr(e, "source") and isinstance(e.source, dict):
            source = e.source.get("title")

        if not title or not link:
            continue

        out.append(
            {
                "published_at": published,
                "source": source or "Google News",
                "title": title,
                "url": link,
            }
        )
        if len(out) >= max_items:
            break

    return status, out


# ----------------------------
# Filings provider: SEC EDGAR (no API key; needs User-Agent)
# ----------------------------

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def sec_headers(user_agent: str) -> Dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def load_sec_cik_map(user_agent: str, timeout: int = 15) -> Tuple[Optional[int], Dict[str, str]]:
    """
    Loads SEC ticker->CIK mapping once per run.
    Returns (http_status, map[ticker]=10-digit CIK string).
    """
    r = requests.get(SEC_TICKER_CIK_URL, headers=sec_headers(user_agent), timeout=timeout)
    if r.status_code != 200:
        return r.status_code, {}

    data = r.json()
    out: Dict[str, str] = {}
    for _, v in data.items():
        t = str(v.get("ticker", "")).upper().strip()
        cik = str(v.get("cik_str", "")).strip().zfill(10)
        if t and cik:
            out[t] = cik
    return 200, out


def fetch_latest_filings_sec(
    symbol: str,
    cik_map: Dict[str, str],
    user_agent: str,
    max_items: int = 6,
    timeout: int = 15,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """
    Returns (http_status, items)
    Each item: {type, filed_at, accession, report_date, primary_doc, url}
    """
    t = symbol.upper().strip()
    cik = cik_map.get(t)
    if not cik:
        # Not in SEC map (non-US listing, ETF not in map, etc.)
        return None, []

    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    r = requests.get(url, headers=sec_headers(user_agent), timeout=timeout)
    status = r.status_code
    if status != 200:
        return status, []

    j = r.json()
    recent = (j.get("filings", {}) or {}).get("recent", {}) or {}

    forms = recent.get("form", []) or []
    filed = recent.get("filingDate", []) or []
    accession = recent.get("accessionNumber", []) or []
    report_date = recent.get("reportDate", []) or []
    primary_doc = recent.get("primaryDocument", []) or []

    keep = {"10-K", "10-Q", "8-K", "20-F", "6-K", "S-1", "424B", "DEF 14A"}
    out: List[Dict[str, Any]] = []

    n = min(len(forms), len(filed), len(accession), len(primary_doc))
    for i in range(n):
        form = str(forms[i])
        if form not in keep:
            continue

        acc_no_dashes = str(accession[i]).replace("-", "")
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dashes}/{primary_doc[i]}"

        out.append(
            {
                "type": form,
                "filed_at": filed[i],
                "accession": accession[i],
                "report_date": report_date[i] if i < len(report_date) else None,
                "primary_doc": primary_doc[i],
                "url": filing_url,
            }
        )
        if len(out) >= max_items:
            break

    return status, out


# ----------------------------
# Main pipeline
# ----------------------------

def run(
    symbols: List[str],
    company_names: Optional[Dict[str, str]] = None,
    max_news: int = 5,
    max_filings: int = 6,
) -> Dict[str, Any]:
    """
    Returns a status object similar to your JSON, plus telemetry.
    """
    company_names = company_names or {}

    sec_user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not sec_user_agent:
        # SEC will block generic/non-contact UAs. Set SEC_USER_AGENT in env.
        # We'll still run news; filings will likely be blocked or empty.
        sec_user_agent = "MyStockAgent (set SEC_USER_AGENT env var)"

    telemetry: List[Dict[str, Any]] = []

    # Load SEC CIK map once
    cik_status, cik_map = load_sec_cik_map(sec_user_agent)
    if cik_status != 200:
        # We'll proceed; SEC might be blocked. Telemetry will reveal it.
        cik_map = {}

    news_requested = len(symbols)
    filings_requested = len(symbols)
    news_fetched = 0
    filings_fetched = 0

    # Optional: store fetched items (you can remove if you only need counts)
    news_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    filings_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    for sym in symbols:
        # NEWS
        with Timer() as t:
            try:
                http_code, items = fetch_news_google_rss(
                    sym,
                    company_name=company_names.get(sym),
                    max_items=max_news,
                )
                status = "ok" if items else "empty"
                if items:
                    news_fetched += 1
                    news_by_symbol[sym] = items
                telemetry.append(
                    FetchResult(
                        symbol=sym,
                        provider="google_rss",
                        kind="news",
                        status=status,
                        http_code=http_code,
                        elapsed_ms=t.elapsed_ms,
                        items=len(items),
                    ).to_dict()
                )
            except Exception as e:
                telemetry.append(
                    FetchResult(
                        symbol=sym,
                        provider="google_rss",
                        kind="news",
                        status="error",
                        elapsed_ms=t.elapsed_ms,
                        error=safe_err(e),
                    ).to_dict()
                )

        # FILINGS
        with Timer() as t:
            try:
                http_code, items = fetch_latest_filings_sec(
                    sym,
                    cik_map=cik_map,
                    user_agent=sec_user_agent,
                    max_items=max_filings,
                )

                if http_code is None and not items:
                    status = "not_supported"
                else:
                    status = "ok" if items else "empty"

                if items:
                    filings_fetched += 1
                    filings_by_symbol[sym] = items

                telemetry.append(
                    FetchResult(
                        symbol=sym,
                        provider="sec_edgar",
                        kind="filings",
                        status=status,
                        http_code=http_code,
                        elapsed_ms=t.elapsed_ms,
                        items=len(items),
                    ).to_dict()
                )
            except Exception as e:
                telemetry.append(
                    FetchResult(
                        symbol=sym,
                        provider="sec_edgar",
                        kind="filings",
                        status="error",
                        elapsed_ms=t.elapsed_ms,
                        error=safe_err(e),
                    ).to_dict()
                )

    status_obj: Dict[str, Any] = {
        "status": "ok",
        "unique_instruments": len(set(symbols)),
        "mapped_symbols": len(symbols),
        "news_requested": news_requested,
        "news_fetched": news_fetched,
        "filings_requested": filings_requested,
        "filings_fetched": filings_fetched,
        "telemetry": telemetry,
        "news": news_by_symbol,         # remove if you don't want payload size
        "filings": filings_by_symbol,   # remove if you don't want payload size
    }

    # Guardrails: if you asked for news but got literally none, fail loudly
    if news_requested > 0 and news_fetched == 0:
        status_obj["status"] = "error"
        status_obj["error"] = "News fetch returned zero across all symbols. Check telemetry (http_code/status/error)."

    # Filings can be legitimately 0 if holdings are non-US, but warn anyway.
    if filings_requested > 0 and filings_fetched == 0:
        status_obj.setdefault("warnings", []).append(
            "Filings fetched zero across all symbols. Could be non-US symbols or SEC blocked. Check telemetry and SEC_USER_AGENT."
        )

    return status_obj


if __name__ == "__main__":
    # Replace this list with your mapped symbols from the portfolio.
    # Quick smoke-test:
    symbols = ["AAPL", "NVDA", "CCJ", "TSLA"]

    # Optional: improves Google News results
    company_names = {
        "AAPL": "Apple",
        "NVDA": "NVIDIA",
        "CCJ": "Cameco",
        "TSLA": "Tesla",
    }

    result = run(symbols, company_names=company_names)
    print(json.dumps(result, indent=2))
