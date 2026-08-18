from __future__ import annotations

import html
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "public"
KST = ZoneInfo("Asia/Seoul")
sys.path.insert(0, str(ROOT))

from scan_volume_ichimoku import MARKET_CAP_KRW_THRESHOLD, ScanRun, scan


def reset_public_dir() -> None:
    public_path = PUBLIC_DIR.resolve()
    root_path = ROOT.resolve()
    if root_path not in public_path.parents:
        raise RuntimeError(f"Refusing to replace unexpected public path: {public_path}")
    public_path.mkdir(parents=True, exist_ok=True)
    for child in public_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def copy_run_artifacts(run: ScanRun, name: str) -> dict[str, str]:
    target_dir = PUBLIC_DIR / name
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(run.report_path, target_dir / run.report_path.name)
    chart_links: dict[str, str] = {}
    for chart_path in run.chart_paths:
        html_path = chart_path.with_suffix(".html")
        shutil.copy2(chart_path, target_dir / chart_path.name)
        if html_path.exists():
            shutil.copy2(html_path, target_dir / html_path.name)

        parts = chart_path.stem.split("_")
        symbol = parts[-1] if parts else chart_path.stem
        chart_links[symbol] = f"{name}/{html_path.name}"

    return chart_links


def e(value: object) -> str:
    return html.escape(str(value))


def volume_text(value: object) -> str:
    try:
        return f"{int(float(value)):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def run_summary(run: ScanRun, name: str, chart_links: dict[str, str]) -> str:
    report_link = f"{name}/{run.report_path.name}"
    rows = []
    for result in run.selected:
        symbol = str(result["symbol"])
        signals = result["signals"]
        chart = chart_links.get(symbol.replace("/", "-"))
        chart_cell = f'<a href="{e(chart)}">chart</a>' if chart else ""
        rows.append(
            "<tr>"
            f"<td>{e(symbol)}</td>"
            f"<td>{e(result.get('asset_class', 'n/a'))}</td>"
            f"<td>{e(result.get('category', 'n/a'))}</td>"
            f"<td>{volume_text(result.get('volume'))}</td>"
            f"<td>{e(signals['1d'])}</td>"
            f"<td>{e(signals['1h'])}</td>"
            f"<td>{e(signals['15m'])}</td>"
            f"<td>{e(signals['5m'])}</td>"
            f"<td>{chart_cell}</td>"
            "</tr>"
        )

    return f"""
      <section>
        <h2>{'프리/애프터 포함' if run.prepost else '정규장 기준'}</h2>
        <p>
          <a href="{e(report_link)}">전체 리포트</a>
          <span>분석 종목 {run.universe_count}개, 원자료 {run.raw_count}개, USD/KRW {run.usd_krw:,.2f}</span>
        </p>
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Type</th><th>Category</th><th>Volume</th>
              <th>1d</th><th>1h</th><th>15m</th><th>5m</th><th>Chart</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def diff_summary(regular: ScanRun, prepost: ScanRun) -> str:
    regular_by_symbol = {str(result["symbol"]): result for result in regular.results}
    prepost_by_symbol = {str(result["symbol"]): result for result in prepost.results}
    common_symbols = sorted(
        regular_by_symbol.keys() & prepost_by_symbol.keys(),
        key=lambda symbol: -(float(regular_by_symbol[symbol].get("volume") or 0)),
    )

    rows = []
    for symbol in common_symbols:
        reg = regular_by_symbol[symbol]
        pre = prepost_by_symbol[symbol]
        if reg["category"] == pre["category"] and reg["signals"] == pre["signals"]:
            continue
        reg_signals = reg["signals"]
        pre_signals = pre["signals"]
        rows.append(
            "<tr>"
            f"<td>{e(symbol)}</td>"
            f"<td>{e(reg.get('category'))}</td>"
            f"<td>{e(pre.get('category'))}</td>"
            f"<td>{e(reg_signals['1d'])}/{e(reg_signals['1h'])}/{e(reg_signals['15m'])}/{e(reg_signals['5m'])}</td>"
            f"<td>{e(pre_signals['1d'])}/{e(pre_signals['1h'])}/{e(pre_signals['15m'])}/{e(pre_signals['5m'])}</td>"
            "</tr>"
        )
        if len(rows) >= 20:
            break

    if not rows:
        rows.append('<tr><td colspan="5">정규장 기준과 프리/애프터 포함 기준의 큰 충돌이 없습니다.</td></tr>')

    return f"""
      <section>
        <h2>정규장 vs 프리/애프터 충돌</h2>
        <table>
          <thead>
            <tr><th>Symbol</th><th>Regular</th><th>Pre/Post</th><th>Regular TF</th><th>Pre/Post TF</th></tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def build_page(regular: ScanRun, prepost: ScanRun, regular_links: dict[str, str], prepost_links: dict[str, str]) -> None:
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ichimoku Volume Scan</title>
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
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 44px;
    }}
    header {{
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 20px;
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
    .scroll {{
      overflow-x: auto;
    }}
    .note {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>거래량 상위 50 이치모쿠 스캔</h1>
      <p>업데이트: {e(now)} / 주식은 시가총액 2조 원 이상, 일반 ETF 포함, 레버리지·인버스 ETF 제외</p>
      <p><a href="daily/index.html">차트 제외 데일리 브리핑</a></p>
      <p>학습용 보조지표 스캔이며 매수·매도 지시가 아닙니다. 프리/애프터 포함 신호는 휩쏘 가능성이 더 큽니다.</p>
    </header>
    <div class="note">
      주식 시총 필터: KRW {MARKET_CAP_KRW_THRESHOLD:,.0f}+ / 정규장 환산 USD {regular.min_market_cap_usd:,.0f}+.
      ETF는 SOXX 같은 일반 ETF를 포함하고, 2x·3x·Bull·Bear·Inverse 계열은 제외합니다.
    </div>
    <div class="scroll">{diff_summary(regular, prepost)}</div>
    <div class="scroll">{run_summary(regular, 'regular', regular_links)}</div>
    <div class="scroll">{run_summary(prepost, 'prepost', prepost_links)}</div>
  </main>
</body>
</html>
"""
    (PUBLIC_DIR / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    reset_public_dir()
    regular = scan(50, 10, prepost=False)
    prepost = scan(50, 10, prepost=True)
    regular_links = copy_run_artifacts(regular, "regular")
    prepost_links = copy_run_artifacts(prepost, "prepost")
    build_page(regular, prepost, regular_links, prepost_links)
    print(f"public page: {PUBLIC_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
