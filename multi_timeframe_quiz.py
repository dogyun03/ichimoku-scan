from __future__ import annotations

import argparse
import html
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.patches import Rectangle

from ichimoku_lab import (
    ICHIMOKU_SHIFT,
    OUTPUT_DIR,
    US_TICKERS,
    TickerInfo,
    add_ichimoku,
    current_snapshot,
    forward_stats,
    next_case_number,
    normalize_ohlcv,
    valid_case_positions,
)


@dataclass(frozen=True)
class TimeframeSpec:
    label: str
    interval: str
    role: str
    window_before: int
    window_after: int


TIMEFRAMES = [
    TimeframeSpec("1h", "60m", "higher trend", 150, 52),
    TimeframeSpec("15m", "15m", "setup", 180, 96),
    TimeframeSpec("5m", "5m", "entry timing", 220, 156),
]


def download_timeframes(ticker: TickerInfo, period: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for spec in TIMEFRAMES:
        raw = yf.download(
            ticker.symbol,
            period=period,
            interval=spec.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
            multi_level_index=False,
        )
        frames[spec.label] = add_ichimoku(normalize_ohlcv(raw))
    return frames


def align_pos_at_or_before(df: pd.DataFrame, timestamp: pd.Timestamp) -> int:
    pos = int(df.index.searchsorted(timestamp, side="right") - 1)
    if pos < 0:
        raise ValueError(f"no bar at or before {timestamp}")
    return pos


def is_valid_aligned_case(frames: dict[str, pd.DataFrame], positions: dict[str, int]) -> bool:
    required_cols = ["Tenkan", "Kijun", "SpanA", "SpanB", "SpanA_raw", "SpanB_raw", "Close_26_Ago"]
    for spec in TIMEFRAMES:
        df = frames[spec.label]
        pos = positions[spec.label]
        if pos < spec.window_before or pos + spec.window_after >= len(df):
            return False
        if df[required_cols].iloc[pos].isna().any():
            return False
    return True


def pick_aligned_case(
    frames: dict[str, pd.DataFrame],
    rng: random.Random,
) -> tuple[pd.Timestamp, dict[str, int]]:
    anchor_spec = TIMEFRAMES[0]
    anchor_df = frames[anchor_spec.label]
    anchor_positions = valid_case_positions(
        anchor_df,
        min_history=anchor_spec.window_before,
        future_bars=anchor_spec.window_after,
    )
    if not anchor_positions:
        raise ValueError("no valid 1h anchor positions")

    candidates = anchor_positions[-180:] if len(anchor_positions) > 180 else anchor_positions
    rng.shuffle(candidates)

    for anchor_pos in candidates:
        cutoff_ts = anchor_df.index[anchor_pos]
        positions = {anchor_spec.label: anchor_pos}
        try:
            for spec in TIMEFRAMES[1:]:
                positions[spec.label] = align_pos_at_or_before(frames[spec.label], cutoff_ts)
        except ValueError:
            continue
        if is_valid_aligned_case(frames, positions):
            return pd.Timestamp(cutoff_ts), positions

    raise ValueError("no aligned 1h/15m/5m case")


def draw_candles_index(ax: plt.Axes, view: pd.DataFrame, xs: np.ndarray, known_mask: np.ndarray) -> None:
    width = 0.62
    for x, (_, row), known in zip(xs, view.iterrows(), known_mask):
        if not known:
            continue

        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        color = "#d62728" if close >= open_price else "#1f77b4"
        ax.vlines(x, low, high, color=color, linewidth=0.75, alpha=0.85)

        lower = min(open_price, close)
        height = abs(close - open_price)
        if height == 0:
            height = max(close * 0.00025, 0.01)

        ax.add_patch(
            Rectangle(
                (x - width / 2, lower),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.45,
                alpha=0.78,
            )
        )


def fmt_timestamp(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%m-%d\n%H:%M")


def plot_timeframe_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    spec: TimeframeSpec,
    cutoff_pos: int,
    *,
    reveal_future: bool,
) -> None:
    start_pos = max(0, cutoff_pos - spec.window_before)
    visible_after = spec.window_after if reveal_future else ICHIMOKU_SHIFT
    end_pos = min(len(df), cutoff_pos + visible_after + 1)
    view = df.iloc[start_pos:end_pos].copy()
    global_pos = np.arange(start_pos, end_pos)
    xs = np.arange(len(view))
    known_last_pos = end_pos - 1 if reveal_future else cutoff_pos
    known_mask = global_pos <= known_last_pos

    draw_candles_index(ax, view, xs, known_mask)

    ax.plot(xs, view["Tenkan"].where(known_mask), label="Tenkan 9", color="#ff7f0e", linewidth=1.0)
    ax.plot(xs, view["Kijun"].where(known_mask), label="Kijun 26", color="#9467bd", linewidth=1.0)
    ax.plot(xs, view["MA20"].where(known_mask), label="MA20", color="#17becf", linewidth=0.95, linestyle="--")

    span_a = view["SpanA"].copy()
    span_b = view["SpanB"].copy()
    if not reveal_future:
        known_cloud = (global_pos - ICHIMOKU_SHIFT) <= cutoff_pos
        span_a = span_a.where(known_cloud)
        span_b = span_b.where(known_cloud)

    ax.plot(xs, span_a, label="Senkou A", color="#2ca02c", linewidth=0.85)
    ax.plot(xs, span_b, label="Senkou B", color="#8c564b", linewidth=0.85)

    chikou = view["Chikou"].copy()
    chikou.iloc[global_pos + ICHIMOKU_SHIFT > known_last_pos] = np.nan
    ax.plot(xs, chikou, label="Chikou", color="#7f7f7f", linewidth=0.85, alpha=0.72)

    valid = span_a.notna() & span_b.notna()
    if valid.any():
        ax.fill_between(
            xs,
            span_a.astype(float),
            span_b.astype(float),
            where=(span_a >= span_b).to_numpy() & valid.to_numpy(),
            color="#2ca02c",
            alpha=0.12,
            interpolate=True,
        )
        ax.fill_between(
            xs,
            span_a.astype(float),
            span_b.astype(float),
            where=(span_a < span_b).to_numpy() & valid.to_numpy(),
            color="#d62728",
            alpha=0.10,
            interpolate=True,
        )

    cutoff_x = cutoff_pos - start_pos
    compare_x = cutoff_pos - ICHIMOKU_SHIFT - start_pos
    ax.axvline(cutoff_x, color="#111111", linewidth=1.0, linestyle="--", alpha=0.75)
    if 0 <= compare_x < len(view):
        ax.axvline(compare_x, color="#777777", linewidth=0.8, linestyle=":", alpha=0.55)

    if not reveal_future:
        ax.axvspan(cutoff_x, len(view) - 1, color="#999999", alpha=0.035)

    tick_count = min(8, len(view))
    tick_idx = np.linspace(0, len(view) - 1, tick_count, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([fmt_timestamp(view.index[i]) for i in tick_idx], fontsize=8)
    ax.set_xlim(0, len(view) - 1)
    ax.set_ylabel("Price")
    ax.set_title(f"{spec.label} ({spec.role})", loc="left", fontsize=11)
    ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.6)


def plot_multi_timeframe_case(
    frames: dict[str, pd.DataFrame],
    ticker: TickerInfo,
    case_id: str,
    cutoff_ts: pd.Timestamp,
    positions: dict[str, int],
    output_path: Path,
    *,
    reveal_future: bool,
) -> None:
    fig, axes = plt.subplots(len(TIMEFRAMES), 1, figsize=(16, 12))
    for ax, spec in zip(axes, TIMEFRAMES):
        plot_timeframe_panel(
            ax,
            frames[spec.label],
            spec,
            positions[spec.label],
            reveal_future=reveal_future,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=6, fontsize=8, frameon=True)

    title = "Ichimoku Multi-Timeframe Quiz - Hidden Future"
    subtitle = f"{case_id} | 1h + 15m + 5m | cutoff {cutoff_ts.strftime('%Y-%m-%d %H:%M')}"
    if reveal_future:
        title = f"Ichimoku Multi-Timeframe Review - {ticker.market}:{ticker.symbol}"
    else:
        subtitle += " | ticker hidden | future price hidden, 26-bar cloud shown"

    fig.suptitle(f"{title}\n{subtitle}", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_zoom_viewer(image_path: Path, html_path: Path, title: str) -> None:
    image_name = html.escape(image_path.name, quote=True)
    page_title = html.escape(title, quote=True)
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Arial, sans-serif;
    }}
    body {{
      margin: 0;
      overflow: hidden;
      background: #101214;
      color: #f1f5f9;
    }}
    .toolbar {{
      position: fixed;
      z-index: 2;
      top: 12px;
      left: 12px;
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      background: rgba(16, 18, 20, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 8px;
      backdrop-filter: blur(8px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24);
    }}
    button, a {{
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      background: #253040;
      color: #f8fafc;
      padding: 7px 10px;
      font-size: 13px;
      text-decoration: none;
      cursor: pointer;
    }}
    button:hover, a:hover {{
      background: #344155;
    }}
    .hint {{
      color: #cbd5e1;
      font-size: 12px;
      white-space: nowrap;
    }}
    .viewport {{
      width: 100vw;
      height: 100vh;
      overflow: hidden;
      cursor: grab;
      user-select: none;
      touch-action: none;
    }}
    .viewport.dragging {{
      cursor: grabbing;
    }}
    img {{
      transform-origin: 0 0;
      will-change: transform;
      max-width: none;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="fit" type="button">Fit</button>
    <button id="reset" type="button">100%</button>
    <a href="{image_name}" target="_blank" rel="noopener">PNG</a>
    <span id="zoom" class="hint">100%</span>
    <span class="hint">Wheel: zoom, drag: move</span>
  </div>
  <div id="viewport" class="viewport">
    <img id="chart" src="{image_name}" alt="{page_title}">
  </div>
  <script>
    const viewport = document.getElementById("viewport");
    const chart = document.getElementById("chart");
    const zoomLabel = document.getElementById("zoom");
    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function applyTransform() {{
      chart.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px) scale(${{scale}})`;
      zoomLabel.textContent = `${{Math.round(scale * 100)}}%`;
    }}

    function fitToWindow() {{
      const rect = viewport.getBoundingClientRect();
      const imageW = chart.naturalWidth || chart.width;
      const imageH = chart.naturalHeight || chart.height;
      scale = Math.min(rect.width / imageW, rect.height / imageH);
      offsetX = (rect.width - imageW * scale) / 2;
      offsetY = (rect.height - imageH * scale) / 2;
      applyTransform();
    }}

    function resetToOne() {{
      scale = 1;
      offsetX = 0;
      offsetY = 0;
      applyTransform();
    }}

    viewport.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      const beforeX = (mouseX - offsetX) / scale;
      const beforeY = (mouseY - offsetY) / scale;
      const factor = Math.exp(-event.deltaY * 0.0015);
      scale = Math.min(8, Math.max(0.15, scale * factor));
      offsetX = mouseX - beforeX * scale;
      offsetY = mouseY - beforeY * scale;
      applyTransform();
    }}, {{ passive: false }});

    viewport.addEventListener("pointerdown", (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      viewport.classList.add("dragging");
      viewport.setPointerCapture(event.pointerId);
    }});

    viewport.addEventListener("pointermove", (event) => {{
      if (!dragging) return;
      offsetX += event.clientX - lastX;
      offsetY += event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      applyTransform();
    }});

    viewport.addEventListener("pointerup", (event) => {{
      dragging = false;
      viewport.classList.remove("dragging");
      viewport.releasePointerCapture(event.pointerId);
    }});

    document.getElementById("fit").addEventListener("click", fitToWindow);
    document.getElementById("reset").addEventListener("click", resetToOne);
    window.addEventListener("resize", fitToWindow);
    chart.addEventListener("load", fitToWindow);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_multi_timeframe_review(
    frames: dict[str, pd.DataFrame],
    ticker: TickerInfo,
    case_id: str,
    cutoff_ts: pd.Timestamp,
    positions: dict[str, int],
    review_path: Path,
) -> None:
    lines = [
        f"# Ichimoku Multi-Timeframe Quiz Review: {case_id}",
        "",
        f"- Market/Symbol: `{ticker.market}:{ticker.symbol}` ({ticker.name})",
        f"- Cutoff datetime: `{cutoff_ts}`",
        "- Timeframes: `1h`, `15m`, `5m`",
        "",
    ]

    for spec in TIMEFRAMES:
        df = frames[spec.label]
        pos = positions[spec.label]
        snapshot = current_snapshot(df, pos)
        stats = forward_stats(df, pos)
        lines.extend(
            [
                f"## {spec.label} snapshot",
                "",
                f"- Close at cutoff: `{snapshot['close']:.2f}`",
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
                "### What happened next",
                "",
                f"- 5 bars: `{stats['return_5d_pct']:+.2f}%`",
                f"- 10 bars: `{stats['return_10d_pct']:+.2f}%`",
                f"- 20 bars: `{stats['return_20d_pct']:+.2f}%`",
                f"- 60 bars: `{stats['return_60d_pct']:+.2f}%`",
                f"- Best high within 60 bars: `{stats['max_high_60d_pct']:+.2f}%`",
                f"- Worst low within 60 bars: `{stats['max_low_60d_pct']:+.2f}%`",
                "",
            ]
        )

    lines.extend(
        [
            "## Review questions",
            "",
            "1. Does the 1h chart support long, short, or no-trade?",
            "2. Does the 15m chart confirm that view or warn about range/whipsaw?",
            "3. Does the 5m chart offer an entry, or should entry wait?",
            "4. Is Chikou above or below the price from 26 bars ago on each timeframe?",
            "5. Where would the stop go if this were a real trade plan?",
            "",
        ]
    )
    review_path.write_text("\n".join(lines), encoding="utf-8-sig")


def generate_multi_timeframe_quizzes(cases: int, seed: int, period: str) -> list[Path]:
    rng = random.Random(seed)
    tickers = [TickerInfo(*item) for item in US_TICKERS]
    created: list[Path] = []
    attempts = 0
    first_case_number = next_case_number(OUTPUT_DIR)

    while len(created) // 3 < cases and attempts < cases * 40:
        attempts += 1
        ticker = rng.choice(tickers)
        try:
            frames = download_timeframes(ticker, period)
            cutoff_ts, positions = pick_aligned_case(frames, rng)

            case_id = f"case-{first_case_number + len(created) // 3:03d}"
            question_path = OUTPUT_DIR / f"{case_id}_question.png"
            answer_path = OUTPUT_DIR / f"{case_id}_answer.png"
            question_html_path = OUTPUT_DIR / f"{case_id}_question.html"
            answer_html_path = OUTPUT_DIR / f"{case_id}_answer.html"
            review_path = OUTPUT_DIR / f"{case_id}_review.md"

            plot_multi_timeframe_case(
                frames,
                ticker,
                case_id,
                cutoff_ts,
                positions,
                question_path,
                reveal_future=False,
            )
            plot_multi_timeframe_case(
                frames,
                ticker,
                case_id,
                cutoff_ts,
                positions,
                answer_path,
                reveal_future=True,
            )
            write_multi_timeframe_review(frames, ticker, case_id, cutoff_ts, positions, review_path)
            write_zoom_viewer(question_path, question_html_path, f"{case_id} question")
            write_zoom_viewer(answer_path, answer_html_path, f"{case_id} answer")

            created.extend([question_path, answer_path, review_path])
            print(f"created {case_id}: {ticker.symbol} {cutoff_ts}")
        except Exception as exc:
            print(f"skipped {ticker.market}:{ticker.symbol} - {exc}")

    if len(created) < cases * 3:
        raise RuntimeError(f"created only {len(created) // 3} cases after {attempts} attempts")

    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 1h/15m/5m Ichimoku quiz cases")
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=date.today().toordinal())
    parser.add_argument("--period", default="60d")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_multi_timeframe_quizzes(args.cases, args.seed, args.period)


if __name__ == "__main__":
    main()
