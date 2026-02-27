# main.py
# Silent agent runner:
# - Loads portfolio symbols (hardcoded or via holdings.json)
# - Fetches: news (Google News RSS) + filings (SEC EDGAR, US only)
# - Computes: energy thesis assessment (rule-based scoring)
# - Outputs: one JSON file (no essays on screen). Prints nothing unless ERROR.

from __future__ import annotations

import os
import json
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
import feedparser


# ----------------------------
# Telemetry / helpers
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
    return s[:500]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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
    if company_name:
        q = f'("{symbol}" OR "{company_name}") (stock OR shares OR earnings OR guidance OR acquisition OR merger)'
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
# Filings provider: SEC EDGAR (no API key; needs real User-Agent)
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
    tkr = symbol.upper().strip()
    cik = cik_map.get(tkr)
    if not cik:
        return None, []  # not supported / non-US / not in SEC map

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
# Thesis scoring (rule-based; no screen text)
# ----------------------------

DEFAULT_THESIS_RULES = {
    "version": "energy_thesis_v1",
    "hard_excludes": {
        "industries": ["Oil & Gas Refining & Marketing", "Refining"],
        "tags": ["DOWNSTREAM_HEAVY", "REFINING"],
    },
    "weights": {
        "segment_fit": 25,         # upstream/midstream/nuclear alignment
        "gas_leverage": 20,
        "capital_discipline": 20,
        "balance_sheet": 15,
        "cash_return": 15,
        "geo_relevance": 5,
    },
    "thresholds": {
        "net_debt_to_ebitda_good": 1.5,
        "shareholder_yield_good": 0.08,
    },
}


def load_rules(path: str = "thesis_rules.json") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_THESIS_RULES


def score_energy_thesis(symbol: str, facts: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    facts = structured fields (best effort).
    Required-ish keys (if missing, scorer uses neutral defaults):
      - industry (str)
      - tags (list[str]) e.g. UPSTREAM, MIDSTREAM, URANIUM, NUCLEAR, REFINING
      - geo (str) e.g. US, CANADA, EU, OTHER
      - revenue_mix_gas_pct (0..1)
      - net_debt_to_ebitda (float)
      - shareholder_yield (float)  # dividend+buyback / market cap, 0..1
      - capex_growth_pct (float)   # <=0 good, high positive bad
    """
    industry = (facts.get("industry") or "").strip()
    tags = set([str(x).upper() for x in (facts.get("tags") or [])])
    geo = (facts.get("geo") or "OTHER").upper()

    exclude_reasons: List[str] = []
    for bad_ind in rules["hard_excludes"].get("industries", []):
        if bad_ind.lower() in industry.lower():
            exclude_reasons.append(f"industry:{bad_ind}")
    for bad_tag in rules["hard_excludes"].get("tags", []):
        if bad_tag.upper() in tags:
            exclude_reasons.append(f"tag:{bad_tag.upper()}")

    if exclude_reasons:
        return {
            "symbol": symbol,
            "thesis_score": 0,
            "pillar_scores": {},
            "flags": [],
            "exclude": True,
            "exclude_reasons": exclude_reasons,
        }

    w = rules["weights"]
    t = rules["thresholds"]
    pillar: Dict[str, int] = {}
    flags: List[str] = []

    # Segment fit: upstream/midstream/nuclear/uranium are "in-thesis"
    segment_fit = 0.0
    if "UPSTREAM" in tags:
        segment_fit = max(segment_fit, 1.0)
    if "MIDSTREAM" in tags:
        segment_fit = max(segment_fit, 0.85)
    if "NUCLEAR" in tags or "URANIUM" in tags:
        segment_fit = max(segment_fit, 1.0)
        flags.append("NUCLEAR_BUCKET")

    pillar["segment_fit"] = int(round(w["segment_fit"] * clamp(segment_fit, 0, 1)))

    # Gas leverage
    gas_pct = float(facts.get("revenue_mix_gas_pct") or 0.0)
    if gas_pct >= 0.5:
        flags.append("US_GAS_LEVERAGE" if geo == "US" else "GAS_LEVERAGE")
    pillar["gas_leverage"] = int(round(w["gas_leverage"] * clamp(gas_pct, 0, 1)))

    # Capital discipline proxy (capex growth)
    capex_growth = facts.get("capex_growth_pct")
    if capex_growth is None:
        capex_score = 0.5
    else:
        # <=0 good; 50%+ growth -> 0
        capex_score = 1.0 if capex_growth <= 0.0 else clamp(1.0 - (float(capex_growth) / 0.5), 0, 1)
    pillar["capital_discipline"] = int(round(w["capital_discipline"] * capex_score))

    # Balance sheet (net debt / EBITDA)
    nde = facts.get("net_debt_to_ebitda")
    if nde is None:
        bs_score = 0.5
    else:
        nde = float(nde)
        bs_score = 1.0 if nde <= t["net_debt_to_ebitda_good"] else clamp(t["net_debt_to_ebitda_good"] / nde, 0, 1)
    pillar["balance_sheet"] = int(round(w["balance_sheet"] * bs_score))

    # Cash return (shareholder yield)
    sh_yield = facts.get("shareholder_yield")
    if sh_yield is None:
        cr_score = 0.4
    else:
        sh_yield = float(sh_yield)
        cr_score = 1.0 if sh_yield >= t["shareholder_yield_good"] else clamp(sh_yield / t["shareholder_yield_good"], 0, 1)
    pillar["cash_return"] = int(round(w["cash_return"] * cr_score))

    # Geo relevance (simple; tweak later)
    geo_score = 1.0 if geo in {"US", "CANADA"} else 0.7
    pillar["geo_relevance"] = int(round(w["geo_relevance"] * geo_score))

    total = int(sum(pillar.values()))
    return {
        "symbol": symbol,
        "thesis_score": total,
        "pillar_scores": pillar,
        "flags": flags,
        "exclude": False,
        "exclude_reasons": [],
    }


# ----------------------------
# Facts layer (you can replace with real fundamentals later)
# ----------------------------

def load_facts(path: str = "facts.json") -> Dict[str, Dict[str, Any]]:
    """
    Optional file you maintain/update:
      {
        "EQT": {"industry":"Oil & Gas E&P", "tags":["UPSTREAM"], "geo":"US", "revenue_mix_gas_pct":0.9, ...},
        "EPD": {"industry":"Midstream", "tags":["MIDSTREAM"], "geo":"US", ...}
      }
    If missing, returns empty dict and scorer uses neutral defaults.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_symbols() -> List[str]:
    """
    Option A: holdings.json with {"symbols":["EQT","DVN",...]}
    Option B: set env SYMBOLS="EQT,DVN,EPD"
    Option C: fallback hardcoded sample list.
    """
    if os.path.exists("holdings.json"):
        with open("holdings.json", "r", encoding="utf-8") as f:
            j = json.load(f)
            syms = j.get("symbols") or []
            return [str(x).upper().strip() for x in syms if str(x).strip()]

    env_syms = os.getenv("SYMBOLS", "").strip()
    if env_syms:
        return [s.strip().upper() for s in env_syms.split(",") if s.strip()]

    # Fallback sample
    return ["EQT", "DVN", "EPD", "CCJ", "LEU", "SMR", "OKLO"]


# ----------------------------
# Runner (silent)
# ----------------------------

def run_agent() -> Dict[str, Any]:
    symbols = load_symbols()
    facts_db = load_facts()
    rules = load_rules("thesis_rules.json")

    # Optional: improves news queries (you can put this in facts.json too)
    company_names = {sym: (facts_db.get(sym, {}).get("company_name")) for sym in symbols}
    company_names = {k: v for k, v in company_names.items() if v}

    sec_user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not sec_user_agent:
        sec_user_agent = "MyStockAgent (set SEC_USER_AGENT env var)"  # may get blocked by SEC

    telemetry: List[Dict[str, Any]] = []

    cik_status, cik_map = load_sec_cik_map(sec_user_agent)
    if cik_status != 200:
        cik_map = {}

    news_requested = len(symbols)
    filings_requested = len(symbols)
    news_fetched = 0
    filings_fetched = 0

    results: Dict[str, Any] = {}

    for sym in symbols:
        results[sym] = {"symbol": sym}

        # --- Thesis scoring (always runs)
        facts = facts_db.get(sym, {})
        results[sym]["thesis"] = score_energy_thesis(sym, facts, rules)

        # --- News
        with Timer() as t:
            try:
                http_code, items = fetch_news_google_rss(sym, company_name=company_names.get(sym))
                status = "ok" if items else "empty"
                if items:
                    news_fetched += 1
                telemetry.append(
                    FetchResult(sym, "google_rss", "news", status, http_code, t.elapsed_ms, len(items)).to_dict()
                )
                results[sym]["news"] = items  # remove if you want ultra-compact output
            except Exception as e:
                telemetry.append(
                    FetchResult(sym, "google_rss", "news", "error", None, t.elapsed_ms, 0, safe_err(e)).to_dict()
                )
                results[sym]["news"] = []

        # --- Filings
        with Timer() as t:
            try:
                http_code, items = fetch_latest_filings_sec(sym, cik_map, sec_user_agent)
                if http_code is None and not items:
                    status = "not_supported"
                else:
                    status = "ok" if items else "empty"
                if items:
                    filings_fetched += 1
                telemetry.append(
                    FetchResult(sym, "sec_edgar", "filings", status, http_code, t.elapsed_ms, len(items)).to_dict()
                )
                results[sym]["filings"] = items  # remove if you want ultra-compact output
            except Exception as e:
                telemetry.append(
                    FetchResult(sym, "sec_edgar", "filings", "error", None, t.elapsed_ms, 0, safe_err(e)).to_dict()
                )
                results[sym]["filings"] = []

    status_obj: Dict[str, Any] = {
        "status": "ok",
        "unique_instruments": len(set(symbols)),
        "mapped_symbols": len(symbols),
        "news_requested": news_requested,
        "news_fetched": news_fetched,
        "filings_requested": filings_requested,
        "filings_fetched": filings_fetched,
        "telemetry": telemetry,
        "results": results,
    }

    # Fail loudly if you requested news and got literally zero
    if news_requested > 0 and news_fetched == 0:
        status_obj["status"] = "error"
        status_obj["error"] = "News fetch returned zero across all symbols. Check telemetry (http_code/status/error)."

    # Filings can be legitimately zero if most symbols are non-US; still warn
    if filings_requested > 0 and filings_fetched == 0:
        status_obj.setdefault("warnings", []).append(
            "Filings fetched zero across all symbols. Could be non-US holdings or SEC blocked. Check telemetry and SEC_USER_AGENT."
        )

    # Write output silently
    out_path = os.getenv("OUTPUT_PATH", "agent_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(status_obj, f, ensure_ascii=False, indent=2)

    return status_obj


if __name__ == "__main__":
    # Silent by default: no prints.
    # If something goes wrong, raise so your runner/CI sees it.
    obj = run_agent()
    if obj.get("status") == "error":
        raise SystemExit(obj.get("error", "Agent failed"))
