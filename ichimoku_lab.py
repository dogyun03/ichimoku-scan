from __future__ import annotations

import argparse
import re
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ICHIMOKU_SHIFT = 26

KR_TICKERS = [
    ("KR", "005930", "Samsung Electronics"),
    ("KR", "000660", "SK Hynix"),
    ("KR", "005380", "Hyundai Motor"),
    ("KR", "035420", "NAVER"),
    ("KR", "035720", "Kakao"),
    ("KR", "068270", "Celltrion"),
    ("KR", "051910", "LG Chem"),
    ("KR", "006400", "Samsung SDI"),
    ("KR", "207940", "Samsung Biologics"),
    ("KR", "373220", "LG Energy Solution"),
]

US_TICKERS = [
    ("US", "AAPL", "Apple"),
    ("US", "MSFT", "Microsoft"),
    ("US", "NVDA", "NVIDIA"),
    ("US", "AMZN", "Amazon"),
    ("US", "GOOGL", "Alphabet"),
    ("US", "META", "Meta"),
    ("US", "TSLA", "Tesla"),
    ("US", "AMD", "AMD"),
    ("US", "SPY", "SPDR S&P 500 ETF"),
    ("US", "QQQ", "Invesco QQQ ETF"),
]


@dataclass(frozen=True)
class TickerInfo:
    market: str
    symbol: str
    name: str


@dataclass(frozen=True)
class QuizCase:
    case_id: str
    ticker: TickerInfo
    cutoff_pos: int
    cutoff_date: pd.Timestamp


def load_prices(ticker: TickerInfo, start: str, end: str) -> pd.DataFrame:
    if ticker.market == "KR":
        import FinanceDataReader as fdr

        df = fdr.DataReader(ticker.symbol, start, end)
    else:
        import yfinance as yf

        df = yf.download(
            ticker.symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
            multi_level_index=False,
        )

    return normalize_ohlcv(df)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("empty price data")

    df = df.copy()
    df.columns = [str(col).strip().title() for col in df.columns]

    rename_map = {
        "Adj Close": "Adj Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")

    df = df[required].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA20"] = out["Close"].rolling(20).mean()

    high9 = out["High"].rolling(9).max()
    low9 = out["Low"].rolling(9).min()
    out["Tenkan"] = (high9 + low9) / 2

    high26 = out["High"].rolling(26).max()
    low26 = out["Low"].rolling(26).min()
    out["Kijun"] = (high26 + low26) / 2

    out["SpanA_raw"] = (out["Tenkan"] + out["Kijun"]) / 2

    high52 = out["High"].rolling(52).max()
    low52 = out["Low"].rolling(52).min()
    out["SpanB_raw"] = (high52 + low52) / 2

    out["SpanA"] = out["SpanA_raw"].shift(ICHIMOKU_SHIFT)
    out["SpanB"] = out["SpanB_raw"].shift(ICHIMOKU_SHIFT)
    out["Chikou"] = out["Close"].shift(-ICHIMOKU_SHIFT)
    out["Close_26_Ago"] = out["Close"].shift(ICHIMOKU_SHIFT)
    out["Kijun_Slope_5"] = out["Kijun"] - out["Kijun"].shift(5)
    return out


def current_snapshot(df: pd.DataFrame, pos: int) -> dict[str, object]:
    row = df.iloc[pos]
    close = float(row["Close"])
    tenkan = float(row["Tenkan"])
    kijun = float(row["Kijun"])
    ma20 = float(row["MA20"])
    span_a = float(row["SpanA"])
    span_b = float(row["SpanB"])
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    if close > cloud_top:
        cloud_state = "above_cloud"
        cloud_distance_pct = (close - cloud_top) / close * 100
    elif close < cloud_bottom:
        cloud_state = "below_cloud"
        cloud_distance_pct = (close - cloud_bottom) / close * 100
    else:
        cloud_state = "inside_cloud"
        cloud_distance_pct = 0.0

    if tenkan > kijun:
        tk_state = "tenkan_above_kijun"
    elif tenkan < kijun:
        tk_state = "tenkan_below_kijun"
    else:
        tk_state = "tenkan_equals_kijun"

    future_cloud = (
        "bullish_future_cloud"
        if row["SpanA_raw"] > row["SpanB_raw"]
        else "bearish_future_cloud"
        if row["SpanA_raw"] < row["SpanB_raw"]
        else "flat_future_cloud"
    )

    if pd.isna(row["Close_26_Ago"]):
        chikou_state = "not_enough_history"
    elif close > float(row["Close_26_Ago"]):
        chikou_state = "chikou_above_price_26_ago"
    elif close < float(row["Close_26_Ago"]):
        chikou_state = "chikou_below_price_26_ago"
    else:
        chikou_state = "chikou_equals_price_26_ago"

    cloud_thickness_pct = (cloud_top - cloud_bottom) / close * 100
    kijun_distance_pct = (close - kijun) / kijun * 100 if kijun else np.nan
    ma20_distance_pct = (close - ma20) / ma20 * 100 if ma20 else np.nan

    return {
        "date": df.index[pos].date().isoformat(),
        "close": close,
        "cloud_state": cloud_state,
        "tk_state": tk_state,
        "future_cloud": future_cloud,
        "chikou_state": chikou_state,
        "cloud_thickness_pct": cloud_thickness_pct,
        "cloud_distance_pct": cloud_distance_pct,
        "kijun_distance_pct": kijun_distance_pct,
        "ma20_distance_pct": ma20_distance_pct,
        "kijun_slope_5": float(row["Kijun_Slope_5"]),
        "heuristic": heuristic_label(
            cloud_state=cloud_state,
            tk_state=tk_state,
            future_cloud=future_cloud,
            chikou_state=chikou_state,
            kijun_slope_5=float(row["Kijun_Slope_5"]),
        ),
        "inbum_overlay": inbum_overlay_label(
            cloud_state=cloud_state,
            tk_state=tk_state,
            future_cloud=future_cloud,
            chikou_state=chikou_state,
            kijun_distance_pct=kijun_distance_pct,
            cloud_distance_pct=cloud_distance_pct,
            ma20_distance_pct=ma20_distance_pct,
        ),
    }


def heuristic_label(
    cloud_state: str,
    tk_state: str,
    future_cloud: str,
    chikou_state: str,
    kijun_slope_5: float,
) -> str:
    bull = 0
    bear = 0

    bull += cloud_state == "above_cloud"
    bear += cloud_state == "below_cloud"
    bull += tk_state == "tenkan_above_kijun"
    bear += tk_state == "tenkan_below_kijun"
    bull += future_cloud == "bullish_future_cloud"
    bear += future_cloud == "bearish_future_cloud"
    bull += chikou_state == "chikou_above_price_26_ago"
    bear += chikou_state == "chikou_below_price_26_ago"
    bull += kijun_slope_5 > 0
    bear += kijun_slope_5 < 0

    if cloud_state == "inside_cloud":
        return "range_or_no_trade"
    if bull >= 4 and bull > bear:
        return "bullish_continuation_candidate"
    if bear >= 4 and bear > bull:
        return "bearish_continuation_candidate"
    return "mixed_or_whipsaw_risk"


def inbum_overlay_label(
    *,
    cloud_state: str,
    tk_state: str,
    future_cloud: str,
    chikou_state: str,
    kijun_distance_pct: float,
    cloud_distance_pct: float,
    ma20_distance_pct: float,
) -> str:
    if cloud_state == "inside_cloud":
        return "wait_inside_cloud_balance_zone"

    if cloud_state == "above_cloud" and future_cloud == "bullish_future_cloud":
        if kijun_distance_pct > 6 or cloud_distance_pct > 8:
            return "do_not_chase_wait_for_time_or_price_correction"
        if tk_state == "tenkan_above_kijun" and chikou_state == "chikou_above_price_26_ago":
            if abs(ma20_distance_pct) <= 3 or abs(kijun_distance_pct) <= 4:
                return "bullish_support_test_candidate"
            return "bullish_but_wait_for_pullback_to_ma20_kijun_or_cloud"

    if cloud_state == "below_cloud" and future_cloud == "bearish_future_cloud":
        if kijun_distance_pct < -6:
            return "bearish_but_avoid_late_short_watch_oversold_rebound"
        return "bearish_rebound_to_cloud_can_fail"

    return "mixed_wait_for_cloud_or_kijun_confirmation"


def forward_stats(df: pd.DataFrame, pos: int, horizons: Iterable[int] = (5, 10, 20, 60)) -> dict[str, float]:
    base_close = float(df["Close"].iloc[pos])
    stats: dict[str, float] = {}
    for horizon in horizons:
        future_pos = pos + horizon
        if future_pos < len(df):
            stats[f"return_{horizon}d_pct"] = (float(df["Close"].iloc[future_pos]) / base_close - 1) * 100
        else:
            stats[f"return_{horizon}d_pct"] = np.nan

    max_horizon = max(horizons)
    future = df.iloc[pos + 1 : pos + max_horizon + 1]
    if len(future):
        stats[f"max_high_{max_horizon}d_pct"] = (float(future["High"].max()) / base_close - 1) * 100
        stats[f"max_low_{max_horizon}d_pct"] = (float(future["Low"].min()) / base_close - 1) * 100
    else:
        stats[f"max_high_{max_horizon}d_pct"] = np.nan
        stats[f"max_low_{max_horizon}d_pct"] = np.nan

    return stats


def draw_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    dates = mdates.date2num(df.index.to_pydatetime())
    if len(dates) > 1:
        width = min(0.8, np.median(np.diff(dates)) * 0.65)
    else:
        width = 0.6

    for x, row in zip(dates, df.itertuples()):
        open_price = float(row.Open)
        high = float(row.High)
        low = float(row.Low)
        close = float(row.Close)
        color = "#d62728" if close >= open_price else "#1f77b4"
        ax.plot([x, x], [low, high], color=color, linewidth=0.8, alpha=0.9)
        lower = min(open_price, close)
        height = abs(close - open_price)
        if height == 0:
            height = max(close * 0.0005, 0.01)
        ax.add_patch(
            Rectangle(
                (x - width / 2, lower),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                alpha=0.75,
            )
        )


def plot_ichimoku_case(
    df: pd.DataFrame,
    case: QuizCase,
    output_path: Path,
    *,
    reveal_future: bool,
    window_before: int = 160,
    window_after: int = 60,
    future_cloud_bars: int = ICHIMOKU_SHIFT,
) -> None:
    cutoff_pos = case.cutoff_pos
    start_pos = max(0, cutoff_pos - window_before)
    visible_after = window_after if reveal_future else future_cloud_bars
    end_pos = min(len(df), cutoff_pos + visible_after + 1)
    view = df.iloc[start_pos:end_pos].copy()
    view_positions = np.arange(start_pos, end_pos)
    cutoff_date = df.index[cutoff_pos]
    known_last_pos = end_pos - 1 if reveal_future else cutoff_pos

    fig, ax = plt.subplots(figsize=(14, 7))
    price_view = view.iloc[view_positions <= known_last_pos]
    draw_candles(ax, price_view)

    visible_indicators = view_positions <= known_last_pos
    tenkan = view["Tenkan"].where(visible_indicators)
    kijun = view["Kijun"].where(visible_indicators)
    ma20 = view["MA20"].where(visible_indicators)

    ax.plot(view.index, tenkan, label="Tenkan 9", color="#ff7f0e", linewidth=1.1)
    ax.plot(view.index, kijun, label="Kijun 26", color="#9467bd", linewidth=1.1)
    ax.plot(view.index, ma20, label="MA20", color="#17becf", linewidth=1.05, linestyle="--")

    span_a = view["SpanA"].copy()
    span_b = view["SpanB"].copy()
    if not reveal_future:
        known_cloud = (view_positions - ICHIMOKU_SHIFT) <= cutoff_pos
        span_a = span_a.where(known_cloud)
        span_b = span_b.where(known_cloud)

    ax.plot(view.index, span_a, label="Senkou A", color="#2ca02c", linewidth=0.9)
    ax.plot(view.index, span_b, label="Senkou B", color="#8c564b", linewidth=0.9)
    chikou = view["Chikou"].copy()
    chikou.iloc[view_positions + ICHIMOKU_SHIFT > known_last_pos] = np.nan
    ax.plot(view.index, chikou, label="Chikou", color="#7f7f7f", linewidth=0.9, alpha=0.75)

    compare_pos = cutoff_pos - ICHIMOKU_SHIFT
    if start_pos <= compare_pos < end_pos:
        compare_date = df.index[compare_pos]
        ax.axvline(compare_date, color="#777777", linewidth=0.8, linestyle=":", alpha=0.55)

    cloud_view = pd.DataFrame({"SpanA": span_a, "SpanB": span_b}, index=view.index)
    valid_cloud = cloud_view[["SpanA", "SpanB"]].dropna()
    if not valid_cloud.empty:
        cloud = cloud_view.loc[valid_cloud.index]
        ax.fill_between(
            mdates.date2num(cloud.index.to_pydatetime()),
            cloud["SpanA"].to_numpy(dtype=float),
            cloud["SpanB"].to_numpy(dtype=float),
            where=(cloud["SpanA"] >= cloud["SpanB"]).to_numpy(),
            color="#2ca02c",
            alpha=0.12,
            interpolate=True,
        )
        ax.fill_between(
            mdates.date2num(cloud.index.to_pydatetime()),
            cloud["SpanA"].to_numpy(dtype=float),
            cloud["SpanB"].to_numpy(dtype=float),
            where=(cloud["SpanA"] < cloud["SpanB"]).to_numpy(),
            color="#d62728",
            alpha=0.10,
            interpolate=True,
        )

    ax.axvline(cutoff_date, color="#111111", linewidth=1.0, linestyle="--", alpha=0.75)
    if reveal_future:
        ax.text(cutoff_date, ax.get_ylim()[1], " cutoff", va="top", ha="left", fontsize=9)
    else:
        ax.axvspan(cutoff_date, view.index[-1], color="#999999", alpha=0.035)
        ax.text(
            cutoff_date,
            ax.get_ylim()[1],
            " known future cloud only",
            va="top",
            ha="left",
            fontsize=9,
            color="#444444",
        )

    title = "Ichimoku Quiz - Hidden Future" if not reveal_future else f"Ichimoku Review - {case.ticker.market}:{case.ticker.symbol}"
    subtitle = f"Case {case.case_id} | cutoff {cutoff_date.date().isoformat()}"
    if not reveal_future:
        subtitle += " | ticker hidden | future price hidden, 26-bar cloud shown"

    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13)
    ax.set_ylabel("Price")
    ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlim(view.index[0], view.index[-1])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_review(case: QuizCase, snapshot: dict[str, object], stats: dict[str, float]) -> str:
    def fmt_pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:+.2f}%"

    lines = [
        f"# Ichimoku Quiz Review: {case.case_id}",
        "",
        f"- Market/Symbol: `{case.ticker.market}:{case.ticker.symbol}` ({case.ticker.name})",
        f"- Cutoff date: `{snapshot['date']}`",
        f"- Close at cutoff: `{snapshot['close']:.2f}`",
        "",
        "## Snapshot",
        "",
        f"- Price vs cloud: `{snapshot['cloud_state']}`",
        f"- Tenkan/Kijun: `{snapshot['tk_state']}`",
        f"- Forward cloud structure: `{snapshot['future_cloud']}`",
        f"- Chikou confirmation: `{snapshot['chikou_state']}`",
        f"- Cloud thickness: `{snapshot['cloud_thickness_pct']:.2f}%` of price",
        f"- Distance from cloud edge: `{snapshot['cloud_distance_pct']:+.2f}%`",
        f"- Distance from Kijun: `{snapshot['kijun_distance_pct']:+.2f}%`",
        f"- Distance from MA20: `{snapshot['ma20_distance_pct']:+.2f}%`",
        f"- Heuristic bucket: `{snapshot['heuristic']}`",
        f"- Video-method overlay: `{snapshot['inbum_overlay']}`",
        "",
        "## What Happened Next",
        "",
        f"- 5 trading days: `{fmt_pct(stats['return_5d_pct'])}`",
        f"- 10 trading days: `{fmt_pct(stats['return_10d_pct'])}`",
        f"- 20 trading days: `{fmt_pct(stats['return_20d_pct'])}`",
        f"- 60 trading days: `{fmt_pct(stats['return_60d_pct'])}`",
        f"- Best high within 60 days: `{fmt_pct(stats['max_high_60d_pct'])}`",
        f"- Worst low within 60 days: `{fmt_pct(stats['max_low_60d_pct'])}`",
        "",
        "## 복기 질문",
        "",
        "1. 가격이 구름 위/안/아래 중 어디였는가?",
        "2. 전환선과 기준선은 같은 방향을 말했는가?",
        "3. 후행스팬 확인은 추세를 지지했는가, 방해했는가?",
        "4. 기준선과의 이격이 추격 리스크를 키웠는가?",
        "5. 실제 결과가 내 예측과 달랐다면, 어떤 조건을 과대평가했는가?",
        "",
    ]
    return "\n".join(lines)


def valid_case_positions(df: pd.DataFrame, min_history: int = 220, future_bars: int = 60) -> list[int]:
    required_cols = ["Tenkan", "Kijun", "SpanA", "SpanB", "SpanA_raw", "SpanB_raw", "Close_26_Ago"]
    valid = df[required_cols].notna().all(axis=1)
    positions = np.flatnonzero(valid.to_numpy())
    return [int(pos) for pos in positions if pos >= min_history and pos + future_bars < len(df)]


def next_case_number(output_dir: Path) -> int:
    pattern = re.compile(r"case-(\d{3})_")
    highest = 0
    for path in output_dir.glob("case-*_*.png"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def generate_quizzes(cases: int, seed: int, start: str, end: str) -> list[Path]:
    rng = random.Random(seed)
    tickers = [TickerInfo(*item) for item in KR_TICKERS + US_TICKERS]
    created: list[Path] = []
    attempts = 0
    first_case_number = next_case_number(OUTPUT_DIR)

    while len(created) // 3 < cases and attempts < cases * 12:
        attempts += 1
        ticker = rng.choice(tickers)

        try:
            df = add_ichimoku(load_prices(ticker, start, end))
            positions = valid_case_positions(df)
            if not positions:
                continue

            cutoff_pos = rng.choice(positions)
            cutoff_date = df.index[cutoff_pos]
            case_id = f"case-{first_case_number + len(created) // 3:03d}"
            case = QuizCase(case_id, ticker, cutoff_pos, cutoff_date)

            question_path = OUTPUT_DIR / f"{case_id}_question.png"
            answer_path = OUTPUT_DIR / f"{case_id}_answer.png"
            review_path = OUTPUT_DIR / f"{case_id}_review.md"

            plot_ichimoku_case(df, case, question_path, reveal_future=False)
            plot_ichimoku_case(df, case, answer_path, reveal_future=True)
            snapshot = current_snapshot(df, cutoff_pos)
            stats = forward_stats(df, cutoff_pos)
            review_path.write_text(make_review(case, snapshot, stats), encoding="utf-8-sig")

            created.extend([question_path, answer_path, review_path])
            print(f"created {case_id}: {question_path.name}, {answer_path.name}, {review_path.name}")
        except Exception as exc:
            print(f"skipped {ticker.market}:{ticker.symbol} - {exc}")

    if len(created) < cases * 3:
        raise RuntimeError(f"created only {len(created) // 3} cases after {attempts} attempts")

    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ichimoku chart quiz generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quiz = subparsers.add_parser("quiz", help="Generate hidden-future Ichimoku quiz cases")
    quiz.add_argument("--cases", type=int, default=3, help="Number of quiz cases")
    quiz.add_argument("--seed", type=int, default=7, help="Random seed for reproducible cases")
    quiz.add_argument("--start", default="2016-01-01", help="Historical data start date")
    quiz.add_argument(
        "--end",
        default=(date.today() + timedelta(days=1)).isoformat(),
        help="Historical data end date; yfinance treats this as exclusive",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "quiz":
        generate_quizzes(cases=args.cases, seed=args.seed, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
