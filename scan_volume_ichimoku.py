from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from ichimoku_lab import OUTPUT_DIR, add_ichimoku, current_snapshot, normalize_ohlcv
from multi_timeframe_quiz import TimeframeSpec, plot_timeframe_panel, write_zoom_viewer


@dataclass(frozen=True)
class ScanTimeframe:
    label: str
    interval: str
    period: str
    role: str
    window_before: int
    window_after: int


TIMEFRAMES = [
    ScanTimeframe("1d", "1d", "1y", "daily filter", 180, 26),
    ScanTimeframe("1h", "60m", "60d", "higher trend", 150, 52),
    ScanTimeframe("15m", "15m", "60d", "setup", 180, 96),
    ScanTimeframe("5m", "5m", "60d", "entry timing", 220, 156),
]

REQUIRED_COLS = ["Tenkan", "Kijun", "MA20", "SpanA", "SpanB", "SpanA_raw", "SpanB_raw", "Close_26_Ago"]
KST = ZoneInfo("Asia/Seoul")
MARKET_CAP_KRW_THRESHOLD = 2_000_000_000_000
USD_KRW_FALLBACK = 1_350.0
DEFAULT_SCREEN_COUNT = 250
ACTIVE_SCREENERS = ("most_actives", "most_actives_etfs")
LEVERAGED_ETF_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b[1-9](?:\.\d+)?x\b",
        r"\bultrapro\b",
        r"\bultra\b",
        r"\bleveraged?\b",
        r"\binverse\b",
        r"\bbear\b",
        r"\bbull\b",
        r"\bdaily target\b",
    )
]


@dataclass(frozen=True)
class ScanRun:
    report_path: Path
    chart_paths: list[Path]
    results: list[dict[str, object]]
    selected: list[dict[str, object]]
    excluded_counts: dict[str, int]
    raw_count: int
    universe_count: int
    usd_krw: float
    min_market_cap_usd: float
    prepost: bool


def number_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def get_usd_krw_rate() -> float:
    try:
        raw = yf.download(
            "KRW=X",
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0].dropna()
        else:
            close = raw["Close"].dropna()
        rate = float(close.iloc[-1])
        if 900 <= rate <= 2_000:
            return rate
    except Exception:
        pass
    return USD_KRW_FALLBACK


def get_top_active_symbols(screen_count: int) -> list[dict[str, object]]:
    records_by_symbol: dict[str, dict[str, object]] = {}
    for screener in ACTIVE_SCREENERS:
        response = yf.screen(screener, count=screen_count)
        quotes = response.get("quotes", [])
        for quote in quotes:
            symbol = quote.get("symbol")
            if not symbol:
                continue
            symbol = str(symbol)
            record = {
                "symbol": symbol,
                "name": quote.get("shortName") or quote.get("longName") or symbol,
                "long_name": quote.get("longName") or quote.get("shortName") or symbol,
                "volume": quote.get("regularMarketVolume"),
                "price": quote.get("regularMarketPrice"),
                "change_pct": quote.get("regularMarketChangePercent"),
                "market_time": quote.get("regularMarketTime"),
                "market_cap": quote.get("marketCap"),
                "quote_type": quote.get("quoteType") or "UNKNOWN",
                "source": screener,
            }
            existing = records_by_symbol.get(symbol)
            if existing is None or (number_or_none(record["volume"]) or 0) > (number_or_none(existing.get("volume")) or 0):
                records_by_symbol[symbol] = record

    records = list(records_by_symbol.values())
    records.sort(key=lambda record: -(number_or_none(record.get("volume")) or 0))
    return records


def is_etf(record: dict[str, object]) -> bool:
    quote_type = str(record.get("quote_type") or "").upper()
    if quote_type == "ETF":
        return True
    text = " ".join(str(record.get(key) or "") for key in ("name", "long_name"))
    return bool(re.search(r"\bETF\b", text, re.IGNORECASE))


def is_leveraged_or_inverse_etf(record: dict[str, object]) -> bool:
    if not is_etf(record):
        return False
    text = " ".join(str(record.get(key) or "") for key in ("symbol", "name", "long_name")).lower()
    return any(pattern.search(text) for pattern in LEVERAGED_ETF_PATTERNS)


def market_cap_text(value: object) -> str:
    number = number_or_none(value)
    if number is None:
        return "n/a"
    if number >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.2f}T"
    if number >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if number >= 1_000_000:
        return f"${number / 1_000_000:.1f}M"
    return f"${number:,.0f}"


def ascii_category(value: object) -> str:
    mapping = {
        "관망": "watch",
        "롱 정렬": "long aligned",
        "눌림/돌파 롱 대기": "long pullback/breakout wait",
        "숏 정렬": "short aligned",
        "반등 숏 대기": "short rebound wait",
        "롱 구조-추격주의": "long extended",
        "숏 구조-추격주의": "short extended",
        "상위봉 구름 안-확인 대기": "higher TF in cloud",
    }
    return mapping.get(str(value), str(value))


def filter_universe(
    records: list[dict[str, object]],
    *,
    count: int,
    min_market_cap_usd: float,
    exclude_leveraged_etfs: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    filtered: list[dict[str, object]] = []
    excluded: Counter[str] = Counter()

    for record in records:
        if exclude_leveraged_etfs and is_leveraged_or_inverse_etf(record):
            excluded["leveraged_or_inverse_etf"] += 1
            continue

        if is_etf(record):
            record["asset_class"] = "ETF"
            filtered.append(record)
        else:
            record["asset_class"] = "Stock"
            market_cap = number_or_none(record.get("market_cap"))
            if market_cap is None:
                excluded["missing_market_cap"] += 1
                continue
            if market_cap < min_market_cap_usd:
                excluded["market_cap_below_threshold"] += 1
                continue
            filtered.append(record)

        if len(filtered) >= count:
            break

    return filtered, dict(excluded)


def extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        tickers = set(str(item) for item in raw.columns.get_level_values(0))
        if symbol not in tickers:
            return None
        frame = raw[symbol]
    else:
        frame = raw

    frame = frame.dropna(how="all")
    if frame.empty:
        return None

    try:
        return add_ichimoku(normalize_ohlcv(frame))
    except Exception:
        return None


def download_frames(symbols: list[str], *, prepost: bool) -> dict[str, dict[str, pd.DataFrame]]:
    frames: dict[str, dict[str, pd.DataFrame]] = {symbol: {} for symbol in symbols}
    for spec in TIMEFRAMES:
        print(f"downloading {spec.label} {spec.period} for {len(symbols)} symbols")
        raw = yf.download(
            symbols,
            period=spec.period,
            interval=spec.interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=prepost,
            group_by="ticker",
        )
        for symbol in symbols:
            frame = extract_symbol_frame(raw, symbol)
            if frame is not None:
                frames[symbol][spec.label] = frame
    return frames


def latest_valid_position(df: pd.DataFrame) -> int | None:
    valid = df[REQUIRED_COLS].notna().all(axis=1) & df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    positions = np.flatnonzero(valid.to_numpy())
    return int(positions[-1]) if len(positions) else None


def signal_counts(snapshot: dict[str, object]) -> tuple[int, int]:
    bull = 0
    bear = 0

    bull += 2 if snapshot["cloud_state"] == "above_cloud" else 0
    bear += 2 if snapshot["cloud_state"] == "below_cloud" else 0
    bull += snapshot["tk_state"] == "tenkan_above_kijun"
    bear += snapshot["tk_state"] == "tenkan_below_kijun"
    bull += snapshot["future_cloud"] == "bullish_future_cloud"
    bear += snapshot["future_cloud"] == "bearish_future_cloud"
    bull += snapshot["chikou_state"] == "chikou_above_price_26_ago"
    bear += snapshot["chikou_state"] == "chikou_below_price_26_ago"
    bull += float(snapshot["kijun_distance_pct"]) > 0
    bear += float(snapshot["kijun_distance_pct"]) < 0
    bull += float(snapshot["ma20_distance_pct"]) > 0
    bear += float(snapshot["ma20_distance_pct"]) < 0

    return int(bull), int(bear)


def tf_signal(snapshot: dict[str, object]) -> str:
    bull, bear = signal_counts(snapshot)
    cloud_state = snapshot["cloud_state"]
    if cloud_state == "inside_cloud":
        return "range"
    if cloud_state == "above_cloud" and bull >= 5 and bull > bear:
        return "bull"
    if cloud_state == "below_cloud" and bear >= 5 and bear > bull:
        return "bear"
    if bull > bear:
        return "mixed_bull"
    if bear > bull:
        return "mixed_bear"
    return "mixed"


def analyze_symbol(record: dict[str, object], frames: dict[str, pd.DataFrame]) -> dict[str, object] | None:
    snapshots: dict[str, dict[str, object]] = {}
    positions: dict[str, int] = {}
    signals: dict[str, str] = {}

    for spec in TIMEFRAMES:
        df = frames.get(spec.label)
        if df is None:
            return None
        pos = latest_valid_position(df)
        if pos is None:
            return None
        positions[spec.label] = pos
        snapshot = current_snapshot(df, pos)
        snapshots[spec.label] = snapshot
        signals[spec.label] = tf_signal(snapshot)

    weights = {"1d": 3.0, "1h": 2.0, "15m": 1.25, "5m": 0.75}
    long_score = 0.0
    short_score = 0.0
    for label, snapshot in snapshots.items():
        bull, bear = signal_counts(snapshot)
        weight = weights[label]
        long_score += weight * bull / 7.0
        short_score += weight * bear / 7.0

    extension_risk = max(
        abs(float(snapshots["1h"]["kijun_distance_pct"])),
        abs(float(snapshots["15m"]["kijun_distance_pct"])),
        abs(float(snapshots["1h"]["cloud_distance_pct"])),
        abs(float(snapshots["15m"]["cloud_distance_pct"])),
    )

    category = "관망"
    if signals["1d"] in {"bull", "mixed_bull"} and signals["1h"] in {"bull", "mixed_bull"}:
        if signals["15m"] == "bull" and signals["5m"] in {"bull", "mixed_bull"}:
            category = "롱 정렬" if extension_risk <= 4.0 else "롱 구조-추격주의"
        elif signals["15m"] in {"range", "mixed_bull"} or signals["5m"] in {"range", "mixed_bear"}:
            category = "눌림/돌파 롱 대기"
    elif signals["1d"] in {"bear", "mixed_bear"} and signals["1h"] in {"bear", "mixed_bear"}:
        if signals["15m"] == "bear" and signals["5m"] in {"bear", "mixed_bear"}:
            category = "숏 정렬" if extension_risk <= 4.0 else "숏 구조-추격주의"
        elif signals["15m"] in {"range", "mixed_bear"} or signals["5m"] in {"range", "mixed_bull"}:
            category = "반등 숏 대기"
    elif signals["1d"] == "range" or signals["1h"] == "range":
        category = "상위봉 구름 안-확인 대기"

    return {
        **record,
        "snapshots": snapshots,
        "positions": positions,
        "signals": signals,
        "category": category,
        "long_score": long_score,
        "short_score": short_score,
        "extension_risk": extension_risk,
    }


def extend_with_future_cloud(df: pd.DataFrame, periods: int = 26) -> pd.DataFrame:
    if df.empty:
        return df
    if len(df.index) >= 2:
        step = df.index[-1] - df.index[-2]
    else:
        step = pd.Timedelta(days=1)
    if not isinstance(step, pd.Timedelta) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=1)

    future_index = [df.index[-1] + step * i for i in range(1, periods + 1)]
    future = pd.DataFrame(index=pd.DatetimeIndex(future_index))
    extended = pd.concat([df, future], axis=0)

    for offset in range(1, periods + 1):
        source_pos = len(df) + offset - 1 - periods
        target_pos = len(df) + offset - 1
        if 0 <= source_pos < len(df):
            extended.iloc[target_pos, extended.columns.get_loc("SpanA")] = df["SpanA_raw"].iloc[source_pos]
            extended.iloc[target_pos, extended.columns.get_loc("SpanB")] = df["SpanB_raw"].iloc[source_pos]
    return extended


def plot_scan_chart(result: dict[str, object], frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    symbol = str(result["symbol"])
    name = str(result["name"])
    fig, axes = plt.subplots(len(TIMEFRAMES), 1, figsize=(16, 14))

    for ax, spec in zip(axes, TIMEFRAMES):
        original = frames[spec.label]
        pos = int(result["positions"][spec.label])
        extended = extend_with_future_cloud(original)
        panel_spec = TimeframeSpec(spec.label, spec.interval, spec.role, spec.window_before, spec.window_after)
        plot_timeframe_panel(ax, extended, panel_spec, pos, reveal_future=False)
        ax.set_title(f"{spec.label} ({spec.role}) | {result['signals'][spec.label]}", loc="left", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=6, fontsize=8, frameon=True)
    volume = result.get("volume")
    volume_text = f"{int(volume):,}" if isinstance(volume, (int, float)) and not math.isnan(float(volume)) else "n/a"
    title = f"{symbol} - {name}"
    subtitle = (
        f"{ascii_category(result['category'])} | volume {volume_text} | "
        f"long {result['long_score']:.2f} / short {result['short_score']:.2f}"
    )
    fig.suptitle(f"{title}\n{subtitle}", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fmt_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:+.2f}%"


def fmt_price(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:.2f}"


def write_report(
    results: list[dict[str, object]],
    selected: list[dict[str, object]],
    report_path: Path,
    *,
    prepost: bool,
    raw_count: int,
    universe_count: int,
    excluded_counts: dict[str, int],
    usd_krw: float,
    min_market_cap_krw: float,
    min_market_cap_usd: float,
    exclude_leveraged_etfs: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    lines = [
        "# Volume Top 50 Ichimoku Multi-Timeframe Scan",
        "",
        f"- Scan time: `{now}`",
        "- Universe source: Yahoo Finance `most_actives` + `most_actives_etfs`",
        f"- Raw symbols fetched: `{raw_count}`",
        f"- Filtered universe analyzed: `{universe_count}`",
        f"- Equity market cap filter: KRW `{min_market_cap_krw:,.0f}`+ ~= USD `{min_market_cap_usd:,.0f}`+ at USD/KRW `{usd_krw:,.2f}`",
        f"- ETF rule: normal ETFs included; leveraged/inverse ETFs excluded: `{exclude_leveraged_etfs}`",
        f"- Extended-hours intraday data: `{prepost}`",
        "- Timeframes: `1d`, `1h`, `15m`, `5m`",
        "- Indicators: Ichimoku `9/26/52` + MA20",
        f"- Excluded summary: `{excluded_counts}`",
        "",
        "## Selected candidates",
        "",
        "| Symbol | Type | Name | Category | Price | Change | Volume | Market cap | 1d | 1h | 15m | 5m | Note |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]

    for result in selected:
        volume = result.get("volume")
        volume_text = f"{int(volume):,}" if isinstance(volume, (int, float)) and not math.isnan(float(volume)) else "n/a"
        signals = result["signals"]
        note = f"L {result['long_score']:.2f} / S {result['short_score']:.2f}, ext {result['extension_risk']:.1f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["symbol"]),
                    str(result.get("asset_class") or result.get("quote_type") or "n/a"),
                    str(result["name"]).replace("|", "/"),
                    str(result["category"]),
                    fmt_price(result.get("price")),
                    fmt_pct(result.get("change_pct")),
                    volume_text,
                    market_cap_text(result.get("market_cap")),
                    signals["1d"],
                    signals["1h"],
                    signals["15m"],
                    signals["5m"],
                    note,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Full top 50 classification",
            "",
            "| Rank | Symbol | Type | Category | Volume | Market cap | 1d | 1h | 15m | 5m |",
            "|---:|---|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for rank, result in enumerate(results, 1):
        volume = result.get("volume")
        volume_text = f"{int(volume):,}" if isinstance(volume, (int, float)) and not math.isnan(float(volume)) else "n/a"
        signals = result["signals"]
        lines.append(
            f"| {rank} | {result['symbol']} | {result.get('asset_class') or result.get('quote_type') or 'n/a'} | "
            f"{result['category']} | {volume_text} | {market_cap_text(result.get('market_cap'))} | "
            f"{signals['1d']} | {signals['1h']} | {signals['15m']} | {signals['5m']} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def select_candidates(results: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    quotas = {
        "롱 정렬": 2,
        "눌림/돌파 롱 대기": 2,
        "숏 정렬": 2,
        "반등 숏 대기": 2,
        "롱 구조-추격주의": 1,
        "숏 구조-추격주의": 1,
        "상위봉 구름 안-확인 대기": 1,
    }

    def volume_value(result: dict[str, object]) -> float:
        volume = result.get("volume")
        return float(volume) if isinstance(volume, (int, float)) and not math.isnan(float(volume)) else 0.0

    def direction_score(result: dict[str, object]) -> float:
        category = str(result["category"])
        if "숏" in category:
            return float(result["short_score"])
        if "롱" in category:
            return float(result["long_score"])
        return max(float(result["long_score"]), float(result["short_score"]))

    selected: list[dict[str, object]] = []
    for category, quota in quotas.items():
        bucket = [result for result in results if result["category"] == category]
        bucket.sort(key=lambda result: (-direction_score(result), -volume_value(result)))
        for result in bucket[:quota]:
            if len(selected) < limit and result not in selected:
                selected.append(result)

    if len(selected) < limit:
        remaining = [
            result
            for result in results
            if result["category"] != "관망" and result not in selected
        ]
        remaining.sort(key=lambda result: (-direction_score(result), -volume_value(result)))
        selected.extend(remaining[: limit - len(selected)])

    return selected


def scan(
    count: int,
    chart_limit: int,
    *,
    prepost: bool,
    screen_count: int = DEFAULT_SCREEN_COUNT,
    min_market_cap_krw: float = MARKET_CAP_KRW_THRESHOLD,
    exclude_leveraged_etfs: bool = True,
) -> ScanRun:
    usd_krw = get_usd_krw_rate()
    min_market_cap_usd = min_market_cap_krw / usd_krw
    raw_records = get_top_active_symbols(screen_count)
    records, excluded_counts = filter_universe(
        raw_records,
        count=count,
        min_market_cap_usd=min_market_cap_usd,
        exclude_leveraged_etfs=exclude_leveraged_etfs,
    )
    symbols = [str(record["symbol"]) for record in records]
    frames = download_frames(symbols, prepost=prepost)

    results: list[dict[str, object]] = []
    by_symbol = {str(record["symbol"]): record for record in records}
    for symbol in symbols:
        result = analyze_symbol(by_symbol[symbol], frames.get(symbol, {}))
        if result is not None:
            results.append(result)

    selected = select_candidates(results, chart_limit)
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M")
    session_tag = "prepost" if prepost else "regular"
    report_path = OUTPUT_DIR / f"volume_ichimoku_scan_{stamp}_{session_tag}.md"
    write_report(
        results,
        selected,
        report_path,
        prepost=prepost,
        raw_count=len(raw_records),
        universe_count=len(records),
        excluded_counts=excluded_counts,
        usd_krw=usd_krw,
        min_market_cap_krw=min_market_cap_krw,
        min_market_cap_usd=min_market_cap_usd,
        exclude_leveraged_etfs=exclude_leveraged_etfs,
    )

    chart_paths: list[Path] = []
    for result in selected:
        symbol = str(result["symbol"]).replace("/", "-")
        chart_path = OUTPUT_DIR / f"volume_scan_{stamp}_{session_tag}_{symbol}.png"
        plot_scan_chart(result, frames[str(result["symbol"])], chart_path)
        write_zoom_viewer(chart_path, chart_path.with_suffix(".html"), f"{symbol} Ichimoku scan")
        chart_paths.append(chart_path)
        print(f"chart {result['symbol']}: {chart_path.name}")

    print(f"report: {report_path.name}")
    return ScanRun(
        report_path=report_path,
        chart_paths=chart_paths,
        results=results,
        selected=selected,
        excluded_counts=excluded_counts,
        raw_count=len(raw_records),
        universe_count=len(records),
        usd_krw=usd_krw,
        min_market_cap_usd=min_market_cap_usd,
        prepost=prepost,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan Yahoo most active stocks with Ichimoku MTF filters")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--screen-count", type=int, default=DEFAULT_SCREEN_COUNT)
    parser.add_argument("--chart-limit", type=int, default=8)
    parser.add_argument("--min-market-cap-krw", type=float, default=MARKET_CAP_KRW_THRESHOLD)
    parser.add_argument(
        "--include-leveraged-etfs",
        action="store_true",
        help="Include leveraged/inverse ETFs. The default excludes them.",
    )
    parser.add_argument("--prepost", action="store_true", help="Include premarket/after-hours intraday bars")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scan(
        args.count,
        args.chart_limit,
        prepost=args.prepost,
        screen_count=args.screen_count,
        min_market_cap_krw=args.min_market_cap_krw,
        exclude_leveraged_etfs=not args.include_leveraged_etfs,
    )


if __name__ == "__main__":
    main()
