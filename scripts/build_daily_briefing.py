from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import yfinance as yf

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - GitHub Actions installs this dependency.
    BeautifulSoup = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PUBLIC_ANALYSIS_DIR = ROOT / "public" / "analysis"
PUBLIC_DAILY_DIR = ROOT / "public" / "daily"
PORTFOLIO_PATH = ROOT / "portfolio.csv"
KST = ZoneInfo("Asia/Seoul")
NAVER_RESEARCH_URL = "https://finance.naver.com/research/"

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


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def fmt_fin_pct(value: object) -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    if abs(number) <= 2:
        number *= 100
    return f"{number:+.1f}%"


def fmt_ratio(value: object) -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.1f}"


def fmt_money(value: object, currency: str = "USD") -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000_000:
        text = f"{number / 1_000_000_000_000:.2f}T"
    elif number >= 1_000_000_000:
        text = f"{number / 1_000_000_000:.2f}B"
    elif number >= 1_000_000:
        text = f"{number / 1_000_000:.1f}M"
    else:
        text = f"{number:,.0f}"
    return f"{sign}{currency} {text}"


def fmt_date(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, list) and value:
        return fmt_date(value[0])
    if hasattr(value, "to_pydatetime"):
        return fmt_date(value.to_pydatetime())
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=KST)
        return value.astimezone(KST).strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return clean_text(value) or "n/a"


def fmt_datetime(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=KST)
        return value.astimezone(KST).strftime("%m-%d %H:%M")
    return clean_text(value) or "n/a"


def date_from_timestamp(value: object) -> str | None:
    number = number_or_none(value)
    if number is None:
        return None
    try:
        return datetime.fromtimestamp(number, tz=KST).strftime("%Y-%m-%d")
    except Exception:
        return None


def source_links(symbol: str) -> dict[str, str]:
    encoded = urllib.parse.quote(symbol)
    report_query = urllib.parse.quote(f"{symbol} 증권사 리포트")
    news_query = urllib.parse.quote(f"{symbol} 핵심 뉴스")
    stockplus_query = urllib.parse.quote(symbol)
    naver_query = urllib.parse.quote(f"{symbol} 주식")
    return {
        "yahoo": f"https://finance.yahoo.com/quote/{encoded}/",
        "yahoo_stats": f"https://finance.yahoo.com/quote/{encoded}/key-statistics/",
        "yahoo_analysis": f"https://finance.yahoo.com/quote/{encoded}/analysis/",
        "investing": f"https://www.investing.com/search/?q={encoded}",
        "naver": f"https://search.naver.com/search.naver?query={naver_query}",
        "stockplus": f"https://stockplus.com/search?q={stockplus_query}",
        "news": f"https://www.google.com/search?q={news_query}",
        "reports": f"https://www.google.com/search?q={report_query}",
        "naver_research": NAVER_RESEARCH_URL,
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


def first_metric(statement: object, names: list[str]) -> float | None:
    if statement is None or getattr(statement, "empty", True):
        return None
    try:
        frame = statement
        labels = list(frame.index)
        columns = list(frame.columns)
    except Exception:
        return None

    label_lookup = {clean_text(label).lower(): label for label in labels}
    for name in names:
        label = label_lookup.get(name.lower())
        if label is None:
            for candidate in labels:
                if name.lower() in clean_text(candidate).lower():
                    label = candidate
                    break
        if label is None:
            continue
        for column in columns:
            try:
                value = frame.loc[label, column]
            except Exception:
                continue
            number = number_or_none(value)
            if number is not None:
                return number
    return None


def latest_statement_period(statement: object) -> str:
    if statement is None or getattr(statement, "empty", True):
        return "n/a"
    try:
        columns = list(statement.columns)
    except Exception:
        return "n/a"
    if not columns:
        return "n/a"
    return fmt_date(columns[0])


def price_target_summary(targets: object) -> str:
    if not isinstance(targets, dict):
        return ""
    current = fmt_price(targets.get("current"))
    mean = fmt_price(targets.get("mean"))
    median = fmt_price(targets.get("median"))
    high = fmt_price(targets.get("high"))
    low = fmt_price(targets.get("low"))
    if all(value == "n/a" for value in (mean, median, high, low)):
        return ""
    return f"현재 {current} / 평균 {mean} / 중앙 {median} / 고 {high} / 저 {low}"


def recommendation_mix(summary: object) -> str:
    if summary is None or getattr(summary, "empty", True):
        return ""
    try:
        row = summary.iloc[0]
    except Exception:
        return ""
    parts = [
        f"강매수 {fmt_number(row.get('strongBuy'))}",
        f"매수 {fmt_number(row.get('buy'))}",
        f"보유 {fmt_number(row.get('hold'))}",
        f"매도 {fmt_number(row.get('sell'))}",
        f"강매도 {fmt_number(row.get('strongSell'))}",
    ]
    return " / ".join(parts)


def analyst_action_ko(action: object, price_target_action: object) -> str:
    target_text = clean_text(price_target_action).lower()
    action_text = clean_text(action).lower()
    if "raise" in target_text:
        return "목표가 상향"
    if "lower" in target_text or "drop" in target_text:
        return "목표가 하향"
    if "maintain" in target_text:
        return "목표가 유지"
    if "announce" in target_text:
        return "목표가 제시"
    if action_text in {"init", "initiates"}:
        return "신규 커버리지"
    if action_text in {"up", "upgrade"}:
        return "투자의견 상향"
    if action_text in {"down", "downgrade"}:
        return "투자의견 하향"
    if action_text in {"main", "reit"}:
        return "투자의견 유지"
    return "애널리스트 업데이트"


def foreign_report_summary(symbol: str, report: dict[str, str]) -> str:
    firm = report.get("firm") or "해외 증권사"
    action = report.get("action_ko") or "업데이트"
    grade = report.get("grade_change") or "등급 정보 없음"
    target = report.get("target_change") or "목표가 정보 없음"
    return f"{firm}가 {symbol}에 대해 {action} 의견을 냈습니다. {grade}, {target}."


def foreign_report_analysis(report: dict[str, str]) -> str:
    action = report.get("action_ko", "")
    grade = f"{report.get('to_grade', '')} {report.get('from_grade', '')}".lower()
    if "하향" in action:
        return "해외 톤은 부담 쪽입니다. 반등보다 목표가·등급 하향의 지속 여부를 봅니다."
    if "상향" in action or any(word in grade for word in ("buy", "outperform", "overweight")):
        return "해외 톤은 우호적입니다. 단기 가격보다 컨센서스가 이어지는지 확인합니다."
    if any(word in grade for word in ("sell", "underperform", "underweight")):
        return "해외 톤은 부담 쪽입니다. 반등보다 목표가·등급 하향의 지속 여부를 봅니다."
    return "해외 톤은 유지·점검 성격입니다. 기존 추세를 바꿀 정도의 변화인지 확인합니다."


def fetch_foreign_analyst_bundle(symbol: str, ticker: yf.Ticker, limit: int) -> dict[str, object]:
    bundle: dict[str, object] = {"reports": [], "price_targets": "", "recommendation_mix": ""}
    if limit <= 0:
        return bundle

    try:
        targets = ticker.analyst_price_targets
        bundle["price_targets"] = price_target_summary(targets)
    except Exception:
        pass

    try:
        summary = ticker.recommendations_summary
        bundle["recommendation_mix"] = recommendation_mix(summary)
    except Exception:
        pass

    try:
        actions = ticker.upgrades_downgrades
    except Exception:
        actions = None

    if actions is None or getattr(actions, "empty", True):
        return bundle

    reports: list[dict[str, str]] = []
    try:
        rows = actions.head(limit).iterrows()
    except Exception:
        return bundle

    for index, row in rows:
        from_grade = clean_text(row.get("FromGrade"))
        to_grade = clean_text(row.get("ToGrade"))
        current_target = fmt_price(row.get("currentPriceTarget"))
        prior_target = fmt_price(row.get("priorPriceTarget"))
        grade_change = f"{from_grade or 'n/a'} -> {to_grade or 'n/a'}"
        target_change = f"{prior_target} -> {current_target}" if current_target != "n/a" else ""
        report = {
            "symbol": symbol,
            "date": fmt_datetime(index),
            "firm": clean_text(row.get("Firm")) or "n/a",
            "action_ko": analyst_action_ko(row.get("Action"), row.get("priceTargetAction")),
            "from_grade": from_grade,
            "to_grade": to_grade,
            "grade_change": grade_change,
            "target_change": target_change,
            "price_targets": str(bundle.get("price_targets") or ""),
            "recommendation_mix": str(bundle.get("recommendation_mix") or ""),
            "link": source_links(symbol)["yahoo_analysis"],
        }
        report["summary"] = foreign_report_summary(symbol, report)
        report["analysis"] = foreign_report_analysis(report)
        reports.append(report)
    bundle["reports"] = reports
    return bundle


def fetch_yahoo_extras(symbol: str, *, foreign_report_limit: int = 0) -> dict[str, object]:
    ticker = yf.Ticker(symbol)
    info: dict[str, object] = {}
    calendar: dict[str, object] = {}
    statement: object | None = None

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

    try:
        statement = ticker.quarterly_income_stmt
    except Exception:
        statement = None

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    low = number_or_none(info.get("fiftyTwoWeekLow"))
    high = number_or_none(info.get("fiftyTwoWeekHigh"))
    price_number = number_or_none(price)
    position_52w = "n/a"
    if price_number is not None and low is not None and high is not None and high > low:
        position_52w = f"{((price_number - low) / (high - low)) * 100:.0f}%"

    next_earnings = (
        calendar.get("Earnings Date")
        or date_from_timestamp(info.get("earningsTimestampStart"))
        or date_from_timestamp(info.get("earningsTimestamp"))
    )
    foreign_bundle = fetch_foreign_analyst_bundle(symbol, ticker, foreign_report_limit)

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
        "next_earnings": next_earnings,
        "eps_average": calendar.get("Earnings Average"),
        "revenue_average": calendar.get("Revenue Average"),
        "ex_dividend": calendar.get("Ex-Dividend Date"),
        "position_52w": position_52w,
        "latest_quarter": latest_statement_period(statement),
        "latest_revenue": first_metric(statement, ["Total Revenue", "Operating Revenue"]),
        "latest_net_income": first_metric(
            statement,
            [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
            ],
        ),
        "latest_eps": first_metric(statement, ["Diluted EPS", "Basic EPS"]),
        "analyst_price_target_summary": foreign_bundle.get("price_targets", ""),
        "analyst_recommendation_mix": foreign_bundle.get("recommendation_mix", ""),
        "foreign_reports": foreign_bundle.get("reports", []),
    }


def parse_news_datetime(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def news_theme(title: str) -> str:
    text = title.lower()
    if any(word in text for word in ("earnings", "revenue", "profit", "guidance", "sales")):
        return "실적/가이던스"
    if any(word in text for word in ("analyst", "rating", "target", "upgrade", "downgrade")):
        return "애널리스트"
    if any(word in text for word in ("ai", "chip", "semiconductor", "data center", "gpu")):
        return "AI·반도체"
    if any(word in text for word in ("ev", "battery", "energy", "oil", "gas", "solar")):
        return "전기차·에너지"
    if any(word in text for word in ("bitcoin", "crypto", "rate", "fed", "yield", "inflation")):
        return "매크로·크립토"
    if any(word in text for word in ("lawsuit", "probe", "regulator", "tariff", "ban")):
        return "규제·소송"
    return "기업 이슈"


def news_summary(symbol: str, title: str) -> str:
    theme = news_theme(title)
    if theme == "실적/가이던스":
        return f"{symbol}의 실적 기대치나 가이던스 변화가 주가 반응의 중심입니다."
    if theme == "애널리스트":
        return f"{symbol}에 대한 목표가·투자의견 변화가 단기 수급을 흔들 수 있습니다."
    if theme == "AI·반도체":
        return f"{symbol} 관련 AI·반도체 수요와 밸류체인 흐름을 같이 봐야 합니다."
    if theme == "전기차·에너지":
        return f"{symbol}의 수요, 원가, 정책 변수가 같이 움직이는 뉴스입니다."
    if theme == "매크로·크립토":
        return f"{symbol} 자체 이슈보다 금리·유동성·위험선호 변화와 연결해 봅니다."
    if theme == "규제·소송":
        return f"{symbol}의 밸류에이션보다 이벤트 리스크 확인이 먼저입니다."
    return f"{symbol}의 개별 재료가 기존 추세를 강화하는지 확인할 뉴스입니다."


def korean_news_title(symbol: str, title: str) -> str:
    theme = news_theme(title)
    text = title.lower()
    if "why" in text and "stock" in text:
        return f"{symbol} 주가 변동 배경"
    if "earnings" in text or "guidance" in text:
        return f"{symbol} 실적·가이던스 이슈"
    if "analyst" in text or "rating" in text or "target" in text:
        return f"{symbol} 애널리스트 의견 변화"
    if "upgrade" in text:
        return f"{symbol} 투자의견 상향 관련 뉴스"
    if "downgrade" in text:
        return f"{symbol} 투자의견 하향 관련 뉴스"
    if "ai" in text or "chip" in text or "semiconductor" in text:
        return f"{symbol} AI·반도체 수요 관련 뉴스"
    if "stock" in text and ("rise" in text or "jump" in text or "gain" in text):
        return f"{symbol} 주가 강세 관련 뉴스"
    if "stock" in text and ("fall" in text or "drop" in text or "sink" in text):
        return f"{symbol} 주가 약세 관련 뉴스"
    return f"{symbol} {theme} 핵심 뉴스"


def news_angle(title: str) -> str:
    theme = news_theme(title)
    if theme in {"실적/가이던스", "애널리스트"}:
        return "발표 직후 갭과 거래량이 유지되는지 확인"
    if theme in {"AI·반도체", "전기차·에너지"}:
        return "같은 섹터 동반 강도와 대장주 흐름 확인"
    if theme == "매크로·크립토":
        return "지수·금리·달러 움직임과 같이 비교"
    if theme == "규제·소송":
        return "첫 반응보다 추가 보도와 변동성 확대 주의"
    return "뉴스 이후 전고·전저 돌파 여부 확인"


def fetch_news(symbol: str, limit: int) -> list[dict[str, object]]:
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

    items: list[dict[str, object]] = []
    for item in root.findall("./channel/item"):
        title = clean_text(item.findtext("title") or "")
        link = item.findtext("link") or source_links(symbol)["yahoo"]
        published = item.findtext("pubDate") or ""
        published_dt = parse_news_datetime(published)
        if title:
            items.append(
                {
                    "symbol": symbol,
                    "title": title,
                    "link": link,
                    "published": published,
                    "published_dt": published_dt,
                    "theme": news_theme(title),
                    "ko_title": korean_news_title(symbol, title),
                    "summary": news_summary(symbol, title),
                    "angle": news_angle(title),
                }
            )
        if len(items) >= limit:
            break
    return items


def http_soup(url: str, *, encoding: str = "utf-8", timeout: int = 15):
    if BeautifulSoup is None:
        return None
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return BeautifulSoup(data, "html.parser", from_encoding=encoding)


def report_theme(category: str, title: str) -> str:
    text = f"{category} {title}".lower()
    if re.search(r"\b[1-4]q\b|실적|review|preview|earnings|컨센서스", text):
        return "실적"
    if any(word in text for word in ("반도체", "ai", "hbm", "메모리", "chip")):
        return "AI·반도체"
    if any(word in text for word in ("자동차", "배터리", "전기차", "2차전지", "에너지")):
        return "자동차·에너지"
    if any(word in text for word in ("은행", "증권", "보험", "금융")):
        return "금융"
    if any(word in text for word in ("주간", "weekly", "전략", "시황")):
        return "시장전략"
    if any(word in text for word in ("산업", "업종", "sector")):
        return "업종"
    return category or "리서치"


def report_summary(report: dict[str, str]) -> str:
    theme = report_theme(report.get("category", ""), report.get("title", ""))
    opinion = report.get("opinion") or ""
    target_price = report.get("target_price") or ""
    parts: list[str] = []
    if theme == "실적":
        parts.append("실적과 컨센서스 차이를 확인하는 리포트입니다.")
    elif theme == "AI·반도체":
        parts.append("AI·반도체 수요와 밸류체인 영향을 보는 리포트입니다.")
    elif theme == "자동차·에너지":
        parts.append("수요, 원가, 정책 변수가 함께 작용하는 리포트입니다.")
    elif theme == "금융":
        parts.append("금리, 마진, 건전성 변수를 중심으로 보는 리포트입니다.")
    elif theme == "시장전략":
        parts.append("지수와 업종 흐름을 함께 정리하는 전략 리포트입니다.")
    elif theme == "업종":
        parts.append("개별 종목보다 업종 사이클을 먼저 보는 리포트입니다.")
    else:
        parts.append("제목 기준 핵심 이슈를 점검하는 리포트입니다.")
    if opinion:
        parts.append(f"투자의견은 {opinion}입니다.")
    if target_price:
        parts.append(f"목표가는 {target_price}입니다.")
    return " ".join(parts)


def report_analysis(report: dict[str, str]) -> str:
    text = f"{report.get('title', '')} {report.get('opinion', '')}".lower()
    positive = ("호실적", "상회", "개선", "성장", "모멘텀", "회복", "레벨업", "buy", "매수")
    negative = ("부진", "하회", "둔화", "적자", "우려", "리스크", "하향", "중립", "sell", "hold")
    if any(word in text for word in positive):
        return "톤은 긍정 쪽입니다. 차트 신호와 거래량이 같은 방향으로 붙는지 확인합니다."
    if any(word in text for word in negative):
        return "톤은 방어적입니다. 반등보다 매물 부담과 실적 리스크를 먼저 봅니다."
    return "톤은 중립 점검에 가깝습니다. 같은 업종 흐름과 수급 동반 여부를 확인합니다."


def extract_report_detail(url: str) -> dict[str, str]:
    try:
        soup = http_soup(url, timeout=10)
    except Exception:
        return {}
    if soup is None:
        return {}

    tokens = [clean_text(token) for token in soup.get_text("|", strip=True).split("|")]
    detail: dict[str, str] = {}
    for idx, token in enumerate(tokens):
        if token == "목표가" and idx + 1 < len(tokens):
            detail["target_price"] = tokens[idx + 1]
        elif token == "투자의견" and idx + 1 < len(tokens):
            detail["opinion"] = tokens[idx + 1]
    return detail


def first_report_link(row, needle: str) -> str:
    for anchor in row.find_all("a", href=True):
        href = str(anchor["href"])
        if needle in href:
            return urljoin(NAVER_RESEARCH_URL, href)
    return ""


def first_pdf_link(row) -> str:
    for anchor in row.find_all("a", href=True):
        href = str(anchor["href"])
        if ".pdf" in href.lower() or "stock-research" in href.lower():
            return urljoin(NAVER_RESEARCH_URL, href)
    return ""


def fetch_naver_research_reports(limit: int = 36, detail_limit: int = 12) -> list[dict[str, str]]:
    try:
        soup = http_soup(NAVER_RESEARCH_URL)
    except Exception:
        return []
    if soup is None:
        return []

    reports: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    category_fallback = ["종목분석", "산업분석", "시황정보", "투자정보"]
    tables = soup.find_all("table", class_="type_3")
    detail_count = 0

    for table_index, table in enumerate(tables):
        category = category_fallback[min(table_index, len(category_fallback) - 1)]
        heading = table.find_previous(["h3", "h4", "strong"])
        heading_text = clean_text(heading.get_text(" ", strip=True)) if heading else ""
        for known in category_fallback:
            if known in heading_text:
                category = known
                break

        for row in table.find_all("tr"):
            cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
            if len(cells) < 3 or not any(cells):
                continue
            if any(cell in {"제목", "종목명", "날짜"} for cell in cells):
                continue

            if category in {"종목분석", "산업분석"} and len(cells) >= 5:
                subject = cells[0]
                title = cells[1]
                broker = cells[2]
                report_date = cells[-1]
            else:
                subject = category
                title = cells[0]
                broker = cells[1] if len(cells) > 1 else ""
                report_date = cells[-1]

            if not title or not report_date:
                continue

            read_url = first_report_link(row, "_read.naver")
            pdf_url = first_pdf_link(row)
            key = (title, report_date)
            if key in seen:
                continue
            seen.add(key)

            report = {
                "category": category,
                "subject": subject,
                "title": title,
                "broker": broker,
                "date": report_date,
                "read_url": read_url,
                "pdf_url": pdf_url,
                "theme": report_theme(category, title),
            }
            if read_url and detail_count < detail_limit:
                report.update(extract_report_detail(read_url))
                detail_count += 1
            report["summary"] = report_summary(report)
            report["analysis"] = report_analysis(report)
            reports.append(report)
            if len(reports) >= limit:
                return reports
    return reports


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
    foreign_symbol_limit: int,
    foreign_report_per_symbol: int,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for idx, record in enumerate(records):
        symbol = str(record["symbol"])
        item = dict(record)
        item["links"] = source_links(symbol)
        foreign_limit = foreign_report_per_symbol if idx < foreign_symbol_limit else 0
        item["extras"] = fetch_yahoo_extras(symbol, foreign_report_limit=foreign_limit) if idx < enrich_limit else {}
        item["news"] = fetch_news(symbol, 2) if idx < news_limit else []
        enriched.append(item)
    return enriched


def portfolio_lookup_data(records: list[dict[str, object]], portfolio: dict[str, PortfolioPosition]) -> dict[str, dict[str, object]]:
    by_symbol = {str(record["symbol"]).upper(): record for record in records}
    data: dict[str, dict[str, object]] = {}
    for symbol, position in portfolio.items():
        record = by_symbol.get(symbol)
        price = number_or_none(record.get("price")) if record else None
        pnl_pct = ((price - position.average_price) / position.average_price) * 100 if price is not None else None
        value = price * position.quantity if price is not None else None
        data[symbol] = {
            "symbol": symbol,
            "average_price": fmt_price(position.average_price),
            "quantity": fmt_number(position.quantity),
            "currency": position.currency,
            "price": fmt_price(price),
            "pnl_pct": fmt_pct(pnl_pct),
            "value": fmt_price(value),
        }
    return data


def scanned_lookup_data(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = {}
    for record in records:
        symbol = str(record["symbol"]).upper()
        extras = dict(record.get("extras") or {})
        data[symbol] = {
            "symbol": symbol,
            "name": clean_text(record.get("name") or record.get("long_name") or symbol),
            "type": clean_text(record.get("asset_class") or record.get("quote_type") or "n/a"),
            "price": fmt_price(record.get("price")),
            "change_pct": fmt_pct(record.get("change_pct")),
            "volume": fmt_number(record.get("volume")),
            "market_cap": market_cap_text(record.get("market_cap")),
            "next_earnings": fmt_date(extras.get("next_earnings")),
            "latest_quarter": fmt_date(extras.get("latest_quarter")),
            "latest_revenue": fmt_money(extras.get("latest_revenue")),
            "latest_net_income": fmt_money(extras.get("latest_net_income")),
            "latest_eps": fmt_ratio(extras.get("latest_eps")),
        }
    return data


def portfolio_rows(records: list[dict[str, object]], portfolio: dict[str, PortfolioPosition]) -> str:
    rows: list[str] = []
    by_symbol = {str(record["symbol"]).upper(): record for record in records}
    for symbol, position in portfolio.items():
        record = by_symbol.get(symbol)
        price = number_or_none(record.get("price")) if record else None
        value = price * position.quantity if price is not None else None
        pnl_pct = ((price - position.average_price) / position.average_price) * 100 if price is not None else None
        rows.append(
            "<tr>"
            f"<td>{e(symbol)}</td>"
            f"<td>{fmt_price(position.average_price)}</td>"
            f"<td>{fmt_number(position.quantity)}</td>"
            f"<td>{fmt_price(price)}</td>"
            f"<td>{fmt_pct(pnl_pct)}</td>"
            f"<td>{fmt_price(value)}</td>"
            "</tr>"
        )
    if not rows:
        return (
            '<tr><td colspan="6">평단가를 보려면 저장소 루트에 '
            '<code>portfolio.csv</code>를 만들고 symbol, average_price, quantity, currency를 입력하세요. '
            "공개 GitHub Pages에는 개인 평단가가 자동으로 올라가지 않습니다.</td></tr>"
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
            f"<td>{fmt_date(extras.get('latest_quarter'))}</td>"
            f"<td>{fmt_money(extras.get('latest_revenue'))}</td>"
            f"<td>{fmt_money(extras.get('latest_net_income'))}</td>"
            f"<td>{fmt_ratio(extras.get('latest_eps'))}</td>"
            f"<td>{fmt_fin_pct(extras.get('revenue_growth'))}</td>"
            f"<td>{fmt_fin_pct(extras.get('profit_margin'))}</td>"
            f"<td>{fmt_ratio(extras.get('trailing_pe'))}/{fmt_ratio(extras.get('forward_pe'))}</td>"
            f"<td>{e(extras.get('position_52w', 'n/a'))}</td>"
            f'<td><a href="{e(links["investing"])}">실적</a> '
            f'<a href="{e(links["yahoo_stats"])}">통계</a></td>'
            "</tr>"
        )
    return "".join(rows)


def collect_news(records: list[dict[str, object]], limit: int = 24) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for record in records:
        for item in list(record.get("news") or []):
            items.append(dict(item))
    items.sort(key=lambda item: item.get("published_dt") or datetime.min.replace(tzinfo=KST), reverse=True)
    return items[:limit]


def collect_foreign_reports(records: list[dict[str, object]], limit: int = 24) -> list[dict[str, str]]:
    reports: list[dict[str, str]] = []
    for record in records:
        extras = dict(record.get("extras") or {})
        for report in list(extras.get("foreign_reports") or []):
            reports.append(dict(report))
            if len(reports) >= limit:
                return reports
    return reports


def news_rows(records: list[dict[str, object]]) -> str:
    rows: list[str] = []
    for item in collect_news(records):
        rows.append(
            "<tr>"
            f"<td>{e(item.get('symbol', ''))}</td>"
            f"<td>{e(item.get('theme', '기업 이슈'))}</td>"
            f'<td><a href="{e(item.get("link", ""))}">{e(item.get("ko_title", ""))}</a></td>'
            f"<td>{e(item.get('summary', ''))}</td>"
            f"<td>{e(item.get('angle', ''))}</td>"
            f"<td>{e(item.get('title', ''))}</td>"
            f"<td>{fmt_datetime(item.get('published_dt'))}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="7">자동 수집된 핵심 뉴스가 없습니다. 종목별 뉴스 링크에서 직접 확인하세요.</td></tr>'
    return "".join(rows)


def report_rows(reports: list[dict[str, str]]) -> str:
    rows: list[str] = []
    for report in reports:
        title = e(report.get("title", ""))
        read_url = report.get("read_url") or ""
        pdf_url = report.get("pdf_url") or ""
        title_cell = f'<a href="{e(read_url)}">{title}</a>' if read_url else title
        pdf_cell = f'<a href="{e(pdf_url)}">PDF</a>' if pdf_url else ""
        rows.append(
            "<tr>"
            f"<td>{e(report.get('date', ''))}</td>"
            f"<td>{e(report.get('category', ''))}</td>"
            f"<td>{e(report.get('subject', ''))}</td>"
            f"<td>{title_cell}</td>"
            f"<td>{e(report.get('broker', ''))}</td>"
            f"<td>{e(report.get('theme', ''))}</td>"
            f"<td>{e(report.get('summary', ''))}</td>"
            f"<td>{e(report.get('analysis', ''))}</td>"
            f"<td>{pdf_cell}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="9">오늘 수집된 네이버 리서치 목록이 없습니다.</td></tr>'
    return "".join(rows)


def foreign_report_rows(reports: list[dict[str, str]]) -> str:
    rows: list[str] = []
    for report in reports:
        rows.append(
            "<tr>"
            f"<td>{e(report.get('symbol', ''))}</td>"
            f"<td>{e(report.get('date', ''))}</td>"
            f"<td>{e(report.get('firm', ''))}</td>"
            f"<td>{e(report.get('action_ko', ''))}</td>"
            f"<td>{e(report.get('grade_change', ''))}</td>"
            f"<td>{e(report.get('target_change', ''))}</td>"
            f"<td>{e(report.get('price_targets', ''))}</td>"
            f"<td>{e(report.get('recommendation_mix', ''))}</td>"
            f"<td>{e(report.get('summary', ''))}</td>"
            f"<td>{e(report.get('analysis', ''))}</td>"
            f'<td><a href="{e(report.get("link", ""))}" target="_blank" rel="noopener">Yahoo</a></td>'
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="11">자동 수집된 해외 애널리스트 업데이트가 없습니다.</td></tr>'
    return "".join(rows)


def source_launcher(records: list[dict[str, object]]) -> str:
    buttons = []
    for record in records[:18]:
        symbol = str(record["symbol"])
        buttons.append(f'<button type="button" data-symbol="{e(symbol)}">{e(symbol)}</button>')
    return "".join(buttons)


def write_markdown(
    records: list[dict[str, object]],
    *,
    raw_count: int,
    excluded: dict[str, int],
    usd_krw: float,
    min_market_cap_usd: float,
    portfolio: dict[str, PortfolioPosition],
    research_reports: list[dict[str, str]],
    stamp: str,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"daily_market_briefing_{stamp}.md"
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    lines = [
        "# Stock Analysis Briefing",
        "",
        f"- Update: `{now}`",
        "- Scope: chart analysis separated from source briefing",
        "- Universe source: Yahoo Finance `most_actives` + `most_actives_etfs`",
        f"- Raw symbols fetched: `{raw_count}`",
        f"- Filtered universe analyzed: `{len(records)}`",
        f"- Equity market cap filter: KRW `{MARKET_CAP_KRW_THRESHOLD:,.0f}`+ ~= USD `{min_market_cap_usd:,.0f}`+ at USD/KRW `{usd_krw:,.2f}`",
        "- ETF rule: normal ETFs included; leveraged/inverse ETFs excluded",
        f"- Excluded summary: `{excluded}`",
        "",
        "## Average price lookup",
        "",
    ]
    if portfolio:
        lines.append("| Symbol | Avg | Qty | Price | P/L | Value |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        by_symbol = {str(record["symbol"]).upper(): record for record in records}
        for symbol, position in portfolio.items():
            record = by_symbol.get(symbol)
            price = number_or_none(record.get("price")) if record else None
            pnl_pct = ((price - position.average_price) / position.average_price) * 100 if price is not None else None
            value = price * position.quantity if price is not None else None
            lines.append(
                f"| {symbol} | {fmt_price(position.average_price)} | {fmt_number(position.quantity)} | "
                f"{fmt_price(price)} | {fmt_pct(pnl_pct)} | {fmt_price(value)} |"
            )
    else:
        lines.append("No `portfolio.csv` found. Create one from `portfolio.example.csv` to show average price and P/L.")

    lines.extend(
        [
            "",
            "## Earnings dates, results, and key stats",
            "",
            "| Symbol | Type | Price | Chg | Vol | MCap | Next Earnings | Latest Q | Revenue | Net Income | EPS | Revenue Growth | Margin | PE T/F | 52w Pos |",
            "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        symbol = str(record["symbol"])
        extras = dict(record.get("extras") or {})
        lines.append(
            f"| {symbol} | {record.get('asset_class', record.get('quote_type', 'n/a'))} | "
            f"{fmt_price(record.get('price'))} | {fmt_pct(record.get('change_pct'))} | {fmt_number(record.get('volume'))} | "
            f"{market_cap_text(record.get('market_cap'))} | {fmt_date(extras.get('next_earnings'))} | "
            f"{fmt_date(extras.get('latest_quarter'))} | {fmt_money(extras.get('latest_revenue'))} | "
            f"{fmt_money(extras.get('latest_net_income'))} | {fmt_ratio(extras.get('latest_eps'))} | "
            f"{fmt_fin_pct(extras.get('revenue_growth'))} | {fmt_fin_pct(extras.get('profit_margin'))} | "
            f"{fmt_ratio(extras.get('trailing_pe'))}/{fmt_ratio(extras.get('forward_pe'))} | {extras.get('position_52w', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## Daily key news",
            "",
            "| Symbol | Theme | Korean Brief | Summary | Checkpoint | Original | Published |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in collect_news(records, limit=24):
        lines.append(
            f"| {item.get('symbol', '')} | {item.get('theme', '')} | [{item.get('ko_title', '')}]({item.get('link', '')}) | "
            f"{item.get('summary', '')} | {item.get('angle', '')} | {item.get('title', '')} | {fmt_datetime(item.get('published_dt'))} |"
        )

    lines.extend(
        [
            "",
            "## Overseas analyst updates",
            "",
            "| Symbol | Date | Firm | Action | Grade | Target | Consensus Target | Rating Mix | Summary | Analysis |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for report in collect_foreign_reports(records, limit=24):
        lines.append(
            f"| {report.get('symbol', '')} | {report.get('date', '')} | {report.get('firm', '')} | "
            f"{report.get('action_ko', '')} | {report.get('grade_change', '')} | {report.get('target_change', '')} | "
            f"{report.get('price_targets', '')} | {report.get('recommendation_mix', '')} | "
            f"{report.get('summary', '')} | {report.get('analysis', '')} |"
        )

    lines.extend(
        [
            "",
            "## Broker reports",
            "",
            "| Date | Category | Subject | Title | Broker | Theme | Summary | Analysis |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for report in research_reports:
        title = report.get("title", "")
        title_cell = f"[{title}]({report['read_url']})" if report.get("read_url") else title
        lines.append(
            f"| {report.get('date', '')} | {report.get('category', '')} | {report.get('subject', '')} | {title_cell} | "
            f"{report.get('broker', '')} | {report.get('theme', '')} | {report.get('summary', '')} | {report.get('analysis', '')} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def write_daily_redirect() -> None:
    PUBLIC_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    html_text = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=../analysis/">
  <title>Moved</title>
</head>
<body>
  <p><a href="../analysis/">종목 분석 창으로 이동</a></p>
</body>
</html>
"""
    (PUBLIC_DAILY_DIR / "index.html").write_text(html_text, encoding="utf-8")


def write_html(
    records: list[dict[str, object]],
    *,
    report_path: Path,
    raw_count: int,
    excluded: dict[str, int],
    usd_krw: float,
    min_market_cap_usd: float,
    portfolio: dict[str, PortfolioPosition],
    research_reports: list[dict[str, str]],
) -> None:
    PUBLIC_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    public_report_path = PUBLIC_ANALYSIS_DIR / report_path.name
    public_report_path.write_bytes(report_path.read_bytes())
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    report_link = report_path.name
    portfolio_json = json.dumps(portfolio_lookup_data(records, portfolio), ensure_ascii=False)
    scan_json = json.dumps(scanned_lookup_data(records), ensure_ascii=False)
    naver_research = source_links("NVDA")["naver_research"]
    foreign_reports = collect_foreign_reports(records, limit=24)
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stock Analysis Window</title>
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
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, system-ui, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1260px;
      margin: 0 auto;
      padding: 28px 18px 44px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      margin: 6px 0 12px;
    }}
    a {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    code {{
      font-family: Consolas, monospace;
    }}
    button {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      padding: 8px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    input {{
      width: min(260px, 100%);
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
      text-transform: uppercase;
    }}
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
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 16px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-width: 0;
    }}
    .metric b {{
      display: block;
      font-size: 20px;
      margin-top: 4px;
      word-break: break-word;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 16px 0;
    }}
    .lookup {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
      gap: 12px;
      margin: 12px 0 18px;
    }}
    .lookup-panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      min-width: 0;
    }}
    .lookup-panel h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .lookup-panel dl {{
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 6px 10px;
      margin: 0;
    }}
    .lookup-panel dt {{
      color: var(--muted);
    }}
    .lookup-panel dd {{
      margin: 0;
      word-break: break-word;
    }}
    .source-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0 18px;
    }}
    .source {{
      display: block;
      min-height: 104px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      color: var(--text);
    }}
    .source span {{
      display: block;
      color: var(--muted);
      font-weight: 400;
      margin-top: 6px;
      white-space: normal;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .section-tabs {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 10px 0;
      background: var(--bg);
      border-bottom: 1px solid var(--line);
    }}
    .section-tab {{
      white-space: nowrap;
    }}
    .section-tab.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }}
    .data-section.hidden {{
      display: none;
    }}
    .scroll {{
      overflow-x: auto;
    }}
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
    td:nth-child(4),
    td:nth-child(7),
    td:nth-child(8) {{
      white-space: normal;
      min-width: 220px;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    @media (max-width: 900px) {{
      .grid {{
        grid-template-columns: 1fr 1fr;
      }}
      .lookup,
      .source-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>종목 분석 창</h1>
      <p>업데이트: {e(now)} / <a href="../index.html" target="_blank" rel="noopener">이치모쿠 창 열기</a> / <a href="{e(report_link)}">markdown</a></p>
      <p>차트 판단과 분리해서 실적, 뉴스, 평단가·수급, 증권사 리포트를 한 번에 확인하는 학습용 화면입니다.</p>
    </header>

    <div class="grid">
      <div class="metric">분석 종목<b>{len(records)}</b></div>
      <div class="metric">원자료<b>{raw_count}</b></div>
      <div class="metric">USD/KRW<b>{usd_krw:,.2f}</b></div>
      <div class="metric">국내 리포트<b>{len(research_reports)}</b></div>
      <div class="metric">해외 업데이트<b>{len(foreign_reports)}</b></div>
    </div>

    <div class="note">
      주식 시총 필터: KRW {MARKET_CAP_KRW_THRESHOLD:,.0f}+ / USD {min_market_cap_usd:,.0f}+.
      SOXX 같은 일반 ETF는 포함하고 2x·3x·Bull·Bear·Inverse 계열은 제외합니다.
      리포트 요약은 공개 목록과 제목·메타데이터 기준의 학습용 요약이며, 원문 판단은 링크에서 확인합니다.
    </div>

    <nav class="section-tabs" aria-label="분석 자료 선택">
      <button type="button" class="section-tab active" data-section="stats">실적/통계</button>
      <button type="button" class="section-tab" data-section="news">핵심뉴스</button>
      <button type="button" class="section-tab" data-section="lookup">평단/수급</button>
      <button type="button" class="section-tab" data-section="domesticReports">국내 리포트</button>
      <button type="button" class="section-tab" data-section="foreignReports">해외 리포트</button>
    </nav>

    <section id="section-stats" class="data-section">
      <h2>실적 날짜·결과·기업 주요 통계</h2>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Type</th><th>Price</th><th>Chg</th><th>Volume</th><th>Market Cap</th>
              <th>Next Earnings</th><th>Latest Q</th><th>Revenue</th><th>Net Income</th><th>EPS</th>
              <th>Revenue Growth</th><th>Margin</th><th>PE T/F</th><th>52w Pos</th><th>Links</th>
            </tr>
          </thead>
          <tbody>{stats_rows(records)}</tbody>
        </table>
      </div>
    </section>

    <section id="section-news" class="data-section hidden">
      <h2>오늘 핵심뉴스</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Symbol</th><th>Theme</th><th>한국어 브리핑</th><th>요약</th><th>확인 포인트</th><th>원문 제목</th><th>Published</th></tr></thead>
          <tbody>{news_rows(records)}</tbody>
        </table>
      </div>
    </section>

    <section id="section-lookup" class="data-section hidden">
      <h2>평단가·수급 티커 조회</h2>
      <div class="toolbar">
        <input id="symbolInput" value="NVDA" aria-label="symbol">
        <button type="button" id="applySymbol">종목 적용</button>
        <button type="button" id="openAll">분석 출처 한번에 열기</button>
      </div>
      <div class="chips">{source_launcher(records)}</div>
      <div class="lookup">
        <div class="lookup-panel">
          <h3 id="lookupTitle">종목 조회</h3>
          <dl id="lookupStats"></dl>
        </div>
        <div class="lookup-panel">
          <h3>평단가</h3>
          <dl id="portfolioLookup"></dl>
        </div>
      </div>
      <div class="source-grid">
        <a class="source" data-link="investing" target="_blank" rel="noopener">실적·기업 통계<span>Investing.com 검색으로 실적 캘린더와 주요 지표 확인</span></a>
        <a class="source" data-link="stockplus" target="_blank" rel="noopener">핵심뉴스<span>증권플러스 종목 뉴스와 이슈 흐름 확인</span></a>
        <a class="source" data-link="naver" target="_blank" rel="noopener">기관·외국인·개인 수급<span>한국 6자리 종목코드는 네이버 수급 페이지로 연결</span></a>
        <a class="source" data-link="reports" target="_blank" rel="noopener">증권사 리포트<span>종목별 리포트 검색과 원문 확인</span></a>
      </div>
      <div class="scroll">
        <table>
          <thead><tr><th>Symbol</th><th>Avg</th><th>Qty</th><th>Price</th><th>P/L</th><th>Value</th></tr></thead>
          <tbody>{portfolio_rows(records, portfolio)}</tbody>
        </table>
      </div>
    </section>

    <section id="section-domesticReports" class="data-section hidden">
      <h2>증권사 리포트</h2>
      <p><a href="{e(naver_research)}" target="_blank" rel="noopener">네이버 리서치 원문 목록</a> 기준으로 최신 리포트를 모읍니다.</p>
      <div class="scroll">
        <table>
          <thead>
            <tr><th>Date</th><th>Category</th><th>Subject</th><th>Title</th><th>Broker</th><th>Theme</th><th>요약</th><th>분석</th><th>PDF</th></tr>
          </thead>
          <tbody>{report_rows(research_reports)}</tbody>
        </table>
      </div>
    </section>

    <section id="section-foreignReports" class="data-section hidden">
      <h2>해외 증권사 리포트</h2>
      <p>Yahoo Finance의 해외 애널리스트 등급·목표가 업데이트를 모읍니다. 전문 리포트가 공개되지 않는 경우가 많아, 공개 메타데이터와 원문 링크 중심으로 정리합니다.</p>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Date</th><th>Firm</th><th>Action</th><th>Grade</th><th>Target</th>
              <th>Consensus Target</th><th>Rating Mix</th><th>요약</th><th>분석</th><th>Link</th>
            </tr>
          </thead>
          <tbody>{foreign_report_rows(foreign_reports)}</tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const input = document.getElementById("symbolInput");
    const portfolioData = {portfolio_json};
    const scannedData = {scan_json};
    const sources = {{
      investing: (symbol) => `https://www.investing.com/search/?q=${{encodeURIComponent(symbol)}}`,
      stockplus: (symbol) => `https://stockplus.com/search?q=${{encodeURIComponent(symbol)}}`,
      naver: (symbol) => /^\\d{{6}}$/.test(symbol)
        ? `https://finance.naver.com/item/frgn.naver?code=${{symbol}}`
        : `https://search.naver.com/search.naver?query=${{encodeURIComponent(symbol + " 주식 기관 외국인 개인")}}`,
      reports: (symbol) => `https://www.google.com/search?q=${{encodeURIComponent(symbol + " 증권사 리포트")}}`,
    }};

    function currentSymbol() {{
      return (input.value || "NVDA").trim().toUpperCase();
    }}

    function rowsFromObject(data) {{
      return Object.entries(data)
        .map(([key, value]) => `<dt>${{key}}</dt><dd>${{value}}</dd>`)
        .join("");
    }}

    function applySymbol() {{
      const symbol = currentSymbol();
      input.value = symbol;
      document.querySelectorAll("[data-link]").forEach((link) => {{
        link.href = sources[link.dataset.link](symbol);
      }});

      const scanned = scannedData[symbol];
      const stats = scanned
        ? {{
            이름: scanned.name,
            구분: scanned.type,
            현재가: scanned.price,
            등락률: scanned.change_pct,
            거래량: scanned.volume,
            시가총액: scanned.market_cap,
            다음실적: scanned.next_earnings,
            최근분기: scanned.latest_quarter,
            최근매출: scanned.latest_revenue,
            순이익: scanned.latest_net_income,
            EPS: scanned.latest_eps,
          }}
        : {{안내: "현재 거래량 상위 스캔 원자료에는 없는 티커입니다. 아래 출처 링크로 직접 확인하세요."}};
      document.getElementById("lookupTitle").textContent = `${{symbol}} 조회`;
      document.getElementById("lookupStats").innerHTML = rowsFromObject(stats);

      const position = portfolioData[symbol];
      const portfolio = position
        ? {{
            평단: position.average_price,
            수량: position.quantity,
            현재가: position.price,
            손익률: position.pnl_pct,
            평가금액: position.value,
          }}
        : {{안내: "portfolio.csv에 이 티커가 없거나 공개 Pages 빌드에는 평단가가 포함되지 않았습니다."}};
      document.getElementById("portfolioLookup").innerHTML = rowsFromObject(portfolio);
    }}

    const sectionButtons = document.querySelectorAll("[data-section]");
    const sections = {{
      stats: document.getElementById("section-stats"),
      news: document.getElementById("section-news"),
      lookup: document.getElementById("section-lookup"),
      domesticReports: document.getElementById("section-domesticReports"),
      foreignReports: document.getElementById("section-foreignReports"),
    }};

    function showSection(name) {{
      Object.entries(sections).forEach(([key, section]) => {{
        section.classList.toggle("hidden", key !== name);
      }});
      sectionButtons.forEach((button) => {{
        button.classList.toggle("active", button.dataset.section === name);
      }});
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }}

    sectionButtons.forEach((button) => {{
      button.addEventListener("click", () => showSection(button.dataset.section));
    }});
    document.getElementById("applySymbol").addEventListener("click", applySymbol);
    input.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") applySymbol();
    }});
    document.querySelectorAll("[data-symbol]").forEach((button) => {{
      button.addEventListener("click", () => {{
        input.value = button.dataset.symbol;
        applySymbol();
      }});
    }});
    document.getElementById("openAll").addEventListener("click", () => {{
      applySymbol();
      ["investing", "stockplus", "naver", "reports"].forEach((key, index) => {{
        window.setTimeout(() => window.open(sources[key](currentSymbol()), "_blank", "noopener"), index * 100);
      }});
    }});
    applySymbol();
  </script>
</body>
</html>
"""
    (PUBLIC_ANALYSIS_DIR / "index.html").write_text(html_text, encoding="utf-8")
    write_daily_redirect()


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
    parser.add_argument("--report-limit", type=int, default=36)
    parser.add_argument("--foreign-symbol-limit", type=int, default=14)
    parser.add_argument("--foreign-report-per-symbol", type=int, default=2)
    args = parser.parse_args()

    ensure_portfolio_example()
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M")
    universe, raw_count, excluded, usd_krw, min_market_cap_usd = load_universe(args.count, args.screen_count)
    records = enrich_records(
        universe,
        enrich_limit=args.enrich_limit,
        news_limit=args.news_limit,
        foreign_symbol_limit=args.foreign_symbol_limit,
        foreign_report_per_symbol=args.foreign_report_per_symbol,
    )
    portfolio = read_portfolio()
    research_reports = fetch_naver_research_reports(limit=args.report_limit)
    report_path = write_markdown(
        records,
        raw_count=raw_count,
        excluded=excluded,
        usd_krw=usd_krw,
        min_market_cap_usd=min_market_cap_usd,
        portfolio=portfolio,
        research_reports=research_reports,
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
        research_reports=research_reports,
    )
    print(f"analysis briefing: {PUBLIC_ANALYSIS_DIR / 'index.html'}")
    print(f"report: {report_path.name}")


if __name__ == "__main__":
    main()
