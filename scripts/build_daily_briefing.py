from __future__ import annotations

import argparse
import csv
import html
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PUBLIC_DAILY_DIR = ROOT / "public" / "daily"
PORTFOLIO_PATH = ROOT / "portfolio.csv"
KST = ZoneInfo("Asia/Seoul")

sys.path.insert(0, str(ROOT))

from scan_volume_ichimoku import (  # noqa: E402
    DEFAULT_SCREEN_COUNT,
    MARKET_CAP_KRW_THRESHOLD,
    filter_universe,
    get_top_active_symbols,
    get_usd_krw_rate,
    market_cap_text,
    number_or_none,
)


@dataclass(frozen=True)
class PortfolioPosition:
    symbol: str
    average_price: float
    quantity: float
    currency: str = "USD"


def e(value: object) -> str:
    return html.escape(str(value))


def fmt_number(value: object) -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:,.0f}"


def fmt_price(value: object) -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    if abs(number) >= 100:
        return f"{number:,.2f}"
    return f"{number:.4g}"


def fmt_pct(value: object) -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:+.2f}%"


def fmt_ratio(value: object) -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.1f}"


def fmt_date(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, list) and value:
        return fmt_date(value[0])
    if isinstance(value, datetime):
        return value.astimezone(KST).strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)


def source_links(symbol: str) -> dict[str, str]:
    encoded = urllib.parse.quote(symbol)
    query = urllib.parse.quote(f"{symbol} 증권사 리포트")
    stockplus_query = urllib.parse.quote(symbol)
    naver_query = urllib.parse.quote(f"{symbol} 주식")
    return {
        "yahoo": f"https://finance.yahoo.com/quote/{encoded}/",
        "yahoo_stats": f"https://finance.yahoo.com/quote/{encoded}/key-statistics/",
        "yahoo_analysis": f"https://finance.yahoo.com/quote/{encoded}/analysis/",
        "investing": f"https://www.investing.com/search/?q={encoded}",
        "naver": f"https://search.naver.com/search.naver?query={naver_query}",
        "stockplus": f"https://stockplus.com/search?q={stockplus_query}",
        "reports": f"https://www.google.com/search?q={query}",
    }


def read_portfolio(path: Path = PORTFOLIO_PATH) -> dict[str, PortfolioPosition]:
    if not path.exists():
        return {}

    positions: dict[str, PortfolioPosition] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = str(row.get("symbol") or "").strip().upper()
            average_price = number_or_none(row.get("average_price"))
            quantity = number_or_none(row.get("quantity"))
            if not symbol or average_price is None or quantity is None:
                continue
            positions[symbol] = PortfolioPosition(
                symbol=symbol,
                average_price=average_price,
                quantity=quantity,
                currency=str(row.get("currency") or "USD").strip().upper(),
            )
    return positions


def fetch_yahoo_extras(symbol: str) -> dict[str, object]:
    ticker = yf.Ticker(symbol)
    info: dict[str, object] = {}
    calendar: dict[str, object] = {}

    try:
        raw_info = ticker.get_info()
        if isinstance(raw_info, dict):
            info = raw_info
    except Exception:
        info = {}

    try:
        raw_calendar = ticker.calendar
        if isinstance(raw_calendar, dict):
            calendar = raw_calendar
    except Exception:
        calendar = {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    low = number_or_none(info.get("fiftyTwoWeekLow"))
    high = number_or_none(info.get("fiftyTwoWeekHigh"))
    position_52w = "n/a"
    price_number = number_or_none(price)
    if price_number is not None and low is not None and high is not None and high > low:
        position_52w = f"{((price_number - low) / (high - low)) * 100:.0f}%"

    return {
        "sector": info.get("sector") or "n/a",
        "industry": info.get("industry") or "n/a",
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "beta": info.get("beta"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "target_mean_price": info.get("targetMeanPrice"),
        "recommendation": info.get("recommendationKey") or "n/a",
        "next_earnings": calendar.get("Earnings Date"),
        "eps_average": calendar.get("Earnings Average"),
        "revenue_average": calendar.get("Revenue Average"),
        "ex_dividend": calendar.get("Ex-Dividend Date"),
        "position_52w": position_52w,
    }


def fetch_news(symbol: str, limit: int) -> list[dict[str, str]]:
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?"
        f"s={urllib.parse.quote(symbol)}&region=US&lang=en-US"
    )
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=12) as response:
            data = response.read()
        root = ET.fromstring(data)
    except Exception:
        return []

    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or source_links(symbol)["yahoo"]
        pub_date = item.findtext("pubDate") or ""
        if title:
            items.append({"title": title, "link": link, "published": pub_date})
        if len(items) >= limit:
            break
    return items


def load_universe(count: int, screen_count: int) -> tuple[list[dict[str, object]], int, dict[str, int], float, float]:
    usd_krw = get_usd_krw_rate()
    min_market_cap_usd = MARKET_CAP_KRW_THRESHOLD / usd_krw
    raw = get_top_active_symbols(screen_count)
    universe, excluded = filter_universe(
        raw,
        count=count,
        min_market_cap_usd=min_market_cap_usd,
        exclude_leveraged_etfs=True,
    )
    return universe, len(raw), excluded, usd_krw, min_market_cap_usd


def enrich_records(
    records: list[dict[str, object]],
    *,
    enrich_limit: int,
    news_limit: int,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for idx, record in enumerate(records):
        symbol = str(record["symbol"])
        item = dict(record)
        item["links"] = source_links(symbol)
        item["extras"] = fetch_yahoo_extras(symbol) if idx < enrich_limit else {}
        item["news"] = fetch_news(symbol, 2) if idx < news_limit else []
        enriched.append(item)
    return enriched


def portfolio_rows(records: list[dict[str, object]], portfolio: dict[str, PortfolioPosition]) -> str:
    rows: list[str] = []
    by_symbol = {str(record["symbol"]).upper(): record for record in records}
    for symbol, position in portfolio.items():
        record = by_symbol.get(symbol)
        price = number_or_none(record.get("price")) if record else None
        value = price * position.quantity if price is not None else None
        pnl_pct = ((price - position.average_price) / position.average_price) * 100 if price is not None else None
        links = source_links(symbol)
        rows.append(
            "<tr>"
            f"<td>{e(symbol)}</td>"
            f"<td>{fmt_price(position.average_price)}</td>"
            f"<td>{fmt_price(price)}</td>"
            f"<td>{fmt_pct(pnl_pct)}</td>"
            f"<td>{fmt_price(value)}</td>"
            f'<td><a href="{e(links["yahoo"])}">시세</a> '
            f'<a href="{e(links["reports"])}">리포트</a></td>'
            "</tr>"
        )
    if not rows:
        return (
            '<tr><td colspan="6">평단가를 보려면 저장소 루트에 '
            '<code>portfolio.csv</code>를 만들고 symbol, average_price, quantity, currency를 입력하세요.</td></tr>'
        )
    return "".join(rows)


def stats_rows(records: list[dict[str, object]]) -> str:
    rows: list[str] = []
    for record in records:
        symbol = str(record["symbol"])
        extras = dict(record.get("extras") or {})
        links = dict(record.get("links") or source_links(symbol))
        rows.append(
            "<tr>"
            f"<td>{e(symbol)}</td>"
            f"<td>{e(record.get('asset_class', record.get('quote_type', 'n/a')))}</td>"
            f"<td>{fmt_price(record.get('price'))}</td>"
            f"<td>{fmt_pct(record.get('change_pct'))}</td>"
            f"<td>{fmt_number(record.get('volume'))}</td>"
            f"<td>{market_cap_text(record.get('market_cap'))}</td>"
            f"<td>{fmt_date(extras.get('next_earnings'))}</td>"
            f"<td>{fmt_ratio(extras.get('eps_average'))}</td>"
            f"<td>{fmt_ratio(extras.get('trailing_pe'))}/{fmt_ratio(extras.get('forward_pe'))}</td>"
            f"<td>{e(extras.get('position_52w', 'n/a'))}</td>"
            f'<td><a href="{e(links["investing"])}">Investing</a> '
            f'<a href="{e(links["yahoo_stats"])}">Stats</a></td>'
            "</tr>"
        )
    return "".join(rows)


def news_rows(records: list[dict[str, object]]) -> str:
    rows: list[str] = []
    for record in records:
        symbol = str(record["symbol"])
        links = dict(record.get("links") or source_links(symbol))
        news = list(record.get("news") or [])
        if not news:
            rows.append(
                "<tr>"
                f"<td>{e(symbol)}</td>"
                f'<td><a href="{e(links["yahoo"])}">뉴스 보기</a></td>'
                "<td>자동 수집 없음</td>"
                "</tr>"
            )
            continue
        for item in news:
            rows.append(
                "<tr>"
                f"<td>{e(symbol)}</td>"
                f'<td><a href="{e(item["link"])}">{e(item["title"])}</a></td>'
                f"<td>{e(item.get('published', ''))}</td>"
                "</tr>"
            )
    return "".join(rows)


def link_rows(records: list[dict[str, object]]) -> str:
    rows: list[str] = []
    for record in records:
        symbol = str(record["symbol"])
        links = dict(record.get("links") or source_links(symbol))
        is_korean_code = symbol.isdigit() and len(symbol) == 6
        naver_flow = (
            f"https://finance.naver.com/item/frgn.naver?code={symbol}"
            if is_korean_code
            else links["naver"]
        )
        flow_label = "네이버 수급" if is_korean_code else "네이버 검색"
        rows.append(
            "<tr>"
            f"<td>{e(symbol)}</td>"
            f'<td><a href="{e(naver_flow)}">{flow_label}</a></td>'
            f'<td><a href="{e(links["stockplus"])}">증권플러스</a></td>'
            f'<td><a href="{e(links["reports"])}">리포트 검색</a></td>'
            f'<td><a href="{e(links["yahoo_analysis"])}">애널리스트</a></td>'
            "</tr>"
        )
    return "".join(rows)


def write_markdown(
    records: list[dict[str, object]],
    *,
    raw_count: int,
    excluded: dict[str, int],
    usd_krw: float,
    min_market_cap_usd: float,
    portfolio: dict[str, PortfolioPosition],
    stamp: str,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"daily_market_briefing_{stamp}.md"
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    lines = [
        "# Daily Market Briefing",
        "",
        f"- Update: `{now}`",
        "- Scope: chart analysis excluded",
        "- Universe source: Yahoo Finance `most_actives` + `most_actives_etfs`",
        f"- Raw symbols fetched: `{raw_count}`",
        f"- Filtered universe analyzed: `{len(records)}`",
        f"- Equity market cap filter: KRW `{MARKET_CAP_KRW_THRESHOLD:,.0f}`+ ~= USD `{min_market_cap_usd:,.0f}`+ at USD/KRW `{usd_krw:,.2f}`",
        "- ETF rule: normal ETFs included; leveraged/inverse ETFs excluded",
        f"- Excluded summary: `{excluded}`",
        "",
        "## Portfolio average price",
        "",
    ]
    if portfolio:
        lines.append("| Symbol | Avg | Price | P/L | Value |")
        lines.append("|---|---:|---:|---:|---:|")
        by_symbol = {str(record["symbol"]).upper(): record for record in records}
        for symbol, position in portfolio.items():
            record = by_symbol.get(symbol)
            price = number_or_none(record.get("price")) if record else None
            pnl_pct = ((price - position.average_price) / position.average_price) * 100 if price is not None else None
            value = price * position.quantity if price is not None else None
            lines.append(f"| {symbol} | {fmt_price(position.average_price)} | {fmt_price(price)} | {fmt_pct(pnl_pct)} | {fmt_price(value)} |")
    else:
        lines.append("No `portfolio.csv` found. Create one from `portfolio.example.csv` to show average price and P/L.")

    lines.extend(["", "## Earnings and key stats", "", "| Symbol | Type | Price | Chg | Vol | MCap | Earnings | EPS Avg | PE T/F | 52w Pos |", "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"])
    for record in records:
        symbol = str(record["symbol"])
        extras = dict(record.get("extras") or {})
        lines.append(
            f"| {symbol} | {record.get('asset_class', record.get('quote_type', 'n/a'))} | "
            f"{fmt_price(record.get('price'))} | {fmt_pct(record.get('change_pct'))} | {fmt_number(record.get('volume'))} | "
            f"{market_cap_text(record.get('market_cap'))} | {fmt_date(extras.get('next_earnings'))} | "
            f"{fmt_ratio(extras.get('eps_average'))} | {fmt_ratio(extras.get('trailing_pe'))}/{fmt_ratio(extras.get('forward_pe'))} | "
            f"{extras.get('position_52w', 'n/a')} |"
        )

    lines.extend(["", "## Source links", "", "| Symbol | Investing | News | Supply/Flow | Reports |", "|---|---|---|---|---|"])
    for record in records:
        symbol = str(record["symbol"])
        links = dict(record.get("links") or source_links(symbol))
        lines.append(
            f"| {symbol} | [Investing]({links['investing']}) | [News]({links['yahoo']}) | "
            f"[Naver/Flow]({links['naver']}) | [Reports]({links['reports']}) |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def write_html(
    records: list[dict[str, object]],
    *,
    report_path: Path,
    raw_count: int,
    excluded: dict[str, int],
    usd_krw: float,
    min_market_cap_usd: float,
    portfolio: dict[str, PortfolioPosition],
) -> None:
    PUBLIC_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    public_report_path = PUBLIC_DAILY_DIR / report_path.name
    public_report_path.write_bytes(report_path.read_bytes())
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    report_link = report_path.name
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daily Market Briefing</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #111827;
      --muted: #5b6472;
      --line: #d8dde6;
      --accent: #2563eb;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111318;
        --panel: #181b22;
        --text: #f3f4f6;
        --muted: #a5adba;
        --line: #303640;
        --accent: #7aa2ff;
      }}
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, system-ui, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 18px 44px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 26px 0 10px; font-size: 20px; }}
    p {{ color: var(--muted); margin: 6px 0 12px; }}
    a {{ color: var(--accent); font-weight: 700; text-decoration: none; }}
    code {{ font-family: Consolas, monospace; }}
    .note {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      margin: 12px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 16px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }}
    .metric b {{ display: block; font-size: 20px; margin-top: 4px; }}
    .scroll {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      white-space: nowrap;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    @media (max-width: 780px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>데일리 종목 브리핑</h1>
      <p>업데이트: {e(now)} / 차트 분석 제외 / <a href="../index.html">이치모쿠 스캔 보기</a> / <a href="{e(report_link)}">markdown</a></p>
      <p>실적 날짜, 기업 주요 통계, 핵심 뉴스, 평단가, 수급·리포트 링크를 한 화면에 모았습니다.</p>
    </header>

    <div class="grid">
      <div class="metric">분석 종목<b>{len(records)}</b></div>
      <div class="metric">원자료<b>{raw_count}</b></div>
      <div class="metric">USD/KRW<b>{usd_krw:,.2f}</b></div>
      <div class="metric">제외 ETF<b>{excluded.get('leveraged_or_inverse_etf', 0)}</b></div>
    </div>

    <div class="note">
      주식 시총 필터: KRW {MARKET_CAP_KRW_THRESHOLD:,.0f}+ / USD {min_market_cap_usd:,.0f}+.
      SOXX 같은 일반 ETF는 포함하고 2x·3x·Bull·Bear·Inverse 계열은 제외합니다.
      평단가는 GitHub에 올리지 않는 <code>portfolio.csv</code> 파일이 있을 때만 계산됩니다.
    </div>

    <section>
      <h2>평단가</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Symbol</th><th>Avg</th><th>Price</th><th>P/L</th><th>Value</th><th>Links</th></tr></thead>
          <tbody>{portfolio_rows(records, portfolio)}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>실적 날짜와 기업 주요 통계</h2>
      <div class="scroll">
        <table>
          <thead>
            <tr><th>Symbol</th><th>Type</th><th>Price</th><th>Chg</th><th>Volume</th><th>Market Cap</th><th>Earnings</th><th>EPS Avg</th><th>PE T/F</th><th>52w Pos</th><th>Links</th></tr>
          </thead>
          <tbody>{stats_rows(records)}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>핵심 뉴스</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Symbol</th><th>Headline</th><th>Published</th></tr></thead>
          <tbody>{news_rows(records)}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>수급과 리포트 바로가기</h2>
      <p>미국 종목은 네이버식 기관·외국인·개인 수급이 직접 제공되지 않아 검색 링크로 연결합니다. 한국 6자리 종목코드는 네이버 수급 페이지로 연결됩니다.</p>
      <div class="scroll">
        <table>
          <thead><tr><th>Symbol</th><th>Supply/Flow</th><th>News</th><th>Broker Reports</th><th>Analyst</th></tr></thead>
          <tbody>{link_rows(records)}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    (PUBLIC_DAILY_DIR / "index.html").write_text(html_text, encoding="utf-8")


def ensure_portfolio_example() -> None:
    example = ROOT / "portfolio.example.csv"
    if example.exists():
        return
    example.write_text(
        "symbol,average_price,quantity,currency\n"
        "AAPL,220.00,3,USD\n"
        "SOXX,250.00,1,USD\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chart-free daily market briefing page.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--screen-count", type=int, default=DEFAULT_SCREEN_COUNT)
    parser.add_argument("--enrich-limit", type=int, default=25)
    parser.add_argument("--news-limit", type=int, default=25)
    args = parser.parse_args()

    ensure_portfolio_example()
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M")
    universe, raw_count, excluded, usd_krw, min_market_cap_usd = load_universe(args.count, args.screen_count)
    records = enrich_records(universe, enrich_limit=args.enrich_limit, news_limit=args.news_limit)
    portfolio = read_portfolio()
    report_path = write_markdown(
        records,
        raw_count=raw_count,
        excluded=excluded,
        usd_krw=usd_krw,
        min_market_cap_usd=min_market_cap_usd,
        portfolio=portfolio,
        stamp=stamp,
    )
    write_html(
        records,
        report_path=report_path,
        raw_count=raw_count,
        excluded=excluded,
        usd_krw=usd_krw,
        min_market_cap_usd=min_market_cap_usd,
        portfolio=portfolio,
    )
    print(f"daily briefing: {PUBLIC_DAILY_DIR / 'index.html'}")
    print(f"report: {report_path.name}")


if __name__ == "__main__":
    main()
