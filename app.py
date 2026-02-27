import os
import json
import traceback
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, HTMLResponse
from main import run_agent

app = FastAPI(title="Investing Agent", version="1.0")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/agent_output.json")
TASKS_API_KEY = os.getenv("TASKS_API_KEY", "").strip()

# If you want the report page to require a key, set this to "true" in Railway
REPORT_PROTECT = os.getenv("REPORT_PROTECT", "false").lower() in {"1", "true", "yes"}


def require_api_key(x_api_key: str | None) -> None:
    if not TASKS_API_KEY:
        raise HTTPException(status_code=500, detail="TASKS_API_KEY is not set on the server.")
    if not x_api_key or x_api_key.strip() != TASKS_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    trace = traceback.format_exc()
    print("UNHANDLED ERROR:", trace)
    return JSONResponse(status_code=500, content={"error": str(exc), "trace": trace[:4000]})


@app.get("/")
def health():
    return {"status": "ok", "service": "investing-agent"}


@app.post("/run")
def run(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    os.environ["OUTPUT_PATH"] = OUTPUT_PATH
    return run_agent()


@app.get("/tasks/daily")
def tasks_daily(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    os.environ["OUTPUT_PATH"] = OUTPUT_PATH
    return run_agent()


@app.get("/last")
def last():
    if not os.path.exists(OUTPUT_PATH):
        raise HTTPException(status_code=404, detail=f"{OUTPUT_PATH} not found yet. Run /tasks/daily first.")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def _read_latest() -> dict:
    if not os.path.exists(OUTPUT_PATH):
        return {}
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/report", response_class=HTMLResponse)
def report(key: str | None = None):
    # Optional protection: works in a browser via /report?key=...
    if REPORT_PROTECT:
        if not TASKS_API_KEY:
            return HTMLResponse("<h3>TASKS_API_KEY not set</h3>", status_code=500)
        if not key or key.strip() != TASKS_API_KEY:
            return HTMLResponse("<h3>Unauthorized</h3><p>Add ?key=YOUR_KEY</p>", status_code=401)

    data = _read_latest()
    if not data:
        return HTMLResponse(
            "<h3>No report yet</h3><p>Run <code>/tasks/daily</code> first (via /docs with X-API-Key).</p>",
            status_code=404,
        )

    # Metadata
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    warnings = data.get("warnings", []) or []
    openai_brief = data.get("openai_brief", "")

    # Build rows
    results = data.get("results", {}) or {}
    tickers = list(results.keys())

    # Simple table: ticker, thesis score, latest filing, top news
    rows_html = []
    for tkr in tickers:
        r = results.get(tkr, {}) or {}
        thesis = r.get("thesis", {}) or {}
        score = thesis.get("thesis_score", "")
        flags = ", ".join(thesis.get("flags", []) or [])

        filings = r.get("filings", []) or []
        latest_filing = ""
        if filings:
            f0 = filings[0]
            latest_filing = f"{f0.get('type','')} {f0.get('filed_at','')}"
        news = r.get("news", []) or []
        top_news = ""
        if news:
            n0 = news[0]
            title = _html_escape(str(n0.get("title", "")))
            url = _html_escape(str(n0.get("url", "")))
            source = _html_escape(str(n0.get("source", "")))
            top_news = f'<a href="{url}" target="_blank" rel="noreferrer">{title}</a><div class="muted">{source}</div>'

        rows_html.append(
            f"""
            <tr>
              <td class="mono">{_html_escape(tkr)}</td>
              <td>{_html_escape(str(score))}</td>
              <td class="muted">{_html_escape(flags)}</td>
              <td class="mono">{_html_escape(latest_filing)}</td>
              <td>{top_news}</td>
            </tr>
            """
        )

    warn_html = ""
    if warnings:
        warn_items = "".join(f"<li>{_html_escape(str(w))}</li>" for w in warnings)
        warn_html = f"<div class='card warn'><h3>Warnings</h3><ul>{warn_items}</ul></div>"

    brief_html = ""
    if openai_brief:
        brief_html = f"<div class='card'><h3>Daily Brief</h3><pre>{_html_escape(openai_brief)}</pre></div>"

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>My AI Investing Agent</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; background: #0b0f14; color: #e9eef5; }}
        a {{ color: #8ab4ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .muted {{ color: #9aa7b5; font-size: 12px; }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
        .top {{ display:flex; gap: 16px; flex-wrap: wrap; align-items: baseline; }}
        .title {{ font-size: 26px; font-weight: 700; }}
        .pill {{ display:inline-block; padding: 4px 10px; border: 1px solid #2a3543; border-radius: 999px; font-size: 12px; color:#cbd5e1; }}
        .card {{ background: #111827; border: 1px solid #243041; border-radius: 14px; padding: 14px 16px; margin-top: 14px; }}
        .warn {{ border-color: #5b3b00; background: #16110a; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border-bottom: 1px solid #243041; padding: 10px 8px; vertical-align: top; }}
        th {{ text-align: left; font-size: 12px; color: #9aa7b5; font-weight: 600; }}
        pre {{ white-space: pre-wrap; margin: 0; }}
      </style>
    </head>
    <body>
      <div class="top">
        <div class="title">My AI Investing Agent</div>
        <div class="pill">Last update: {now}</div>
        <div class="pill">Universe: {len(tickers)} tickers</div>
        <div class="pill">News fetched: {data.get("news_fetched",0)}/{data.get("news_requested",0)}</div>
        <div class="pill">Filings fetched: {data.get("filings_fetched",0)}/{data.get("filings_requested",0)}</div>
        <div class="pill"><a href="/docs" target="_blank" rel="noreferrer">Swagger</a></div>
        <div class="pill"><a href="/last" target="_blank" rel="noreferrer">Raw JSON</a></div>
      </div>

      {warn_html}
      {brief_html}

      <div class="card">
        <h3>Portfolio</h3>
        <div class="muted">Tip: Run <span class="mono">/tasks/daily</span> to refresh data.</div>
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Thesis Score</th>
              <th>Flags</th>
              <th>Latest Filing</th>
              <th>Top News</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)
