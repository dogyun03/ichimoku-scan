# Ichimoku Training Lab

한국+미국 주식의 과거 일봉 차트로 이치모쿠 판단력을 훈련하는 작은 랩입니다.

이 프로젝트의 원칙:

- TradingView는 수동 CSV 보조용으로만 사용합니다.
- 자동 퀴즈와 통계 복기는 `FinanceDataReader`, `yfinance`, `pandas`로 재현 가능하게 만듭니다.
- 처음에는 이치모쿠 단독 판단만 훈련합니다.
- 분봉 단타와 엔벨롭은 일봉 이치모쿠 숙련 뒤 확장합니다.

## 빠른 실행

```powershell
python ichimoku_lab.py quiz --cases 3 --seed 7
```

생성물은 `outputs/`에 저장됩니다.

- `*_question.png`: 미래 구간을 가린 퀴즈 차트
- `*_answer.png`: 실제 이후 흐름을 공개한 정답 차트
- `*_review.md`: 이치모쿠 조건, 이후 수익률, 복기 질문

## 거래량 상위 50 자동 스캔

로컬에서 실행:

```powershell
python scan_volume_ichimoku.py --count 50 --chart-limit 10
python scan_volume_ichimoku.py --count 50 --chart-limit 10 --prepost
```

현재 스캔 조건:

- Yahoo Finance `most_actives`와 `most_actives_etfs`를 합쳐 원자료를 가져옵니다.
- 주식은 시가총액 2조 원 이상만 통과시킵니다.
- SOXX 같은 일반 ETF는 포함합니다.
- 2x, 3x, Bull, Bear, Inverse 계열 레버리지/인버스 ETF는 제외합니다.
- 1일봉, 1시간봉, 15분봉, 5분봉 이치모쿠 상태를 함께 봅니다.

PC가 꺼져도 1시간마다 보려면 이 저장소를 GitHub에 올리고, GitHub 저장소의
`Settings > Pages`에서 `GitHub Actions` 배포를 켭니다. 그러면
`.github/workflows/ichimoku-scan.yml`이 매시간 `public/index.html`을 새로 만들어
GitHub Pages URL로 배포합니다.

## 통합 대시보드와 종목 분석 창

GitHub Pages의 루트 화면에서 이치모쿠 스캔과 종목 분석 탭을 함께 봅니다.

- `public/index.html`: 이치모쿠 스캔과 종목 분석을 함께 보는 통합 화면
- `public/analysis/index.html`: 종목 분석 탭에 들어가는 독립 화면
- `public/daily/index.html`: 기존 링크 호환용 리다이렉트

이치모쿠 스캔 화면과 분리해서 TradingView, Investing.com, 증권플러스,
네이버증권, 리포트 검색을 따로 열어보려면:

```powershell
python scripts/build_daily_briefing.py --count 50
```

분석 리포트 생성물:

- `outputs/daily_market_briefing_*.md`: 종목 분석 내용을 정리한 markdown 리포트

평단가와 보유 수량은 개인 정보라 git에 올리지 않습니다. `portfolio.example.csv`를 참고해서
루트 폴더에 `portfolio.csv`를 만들면 종목 분석 창의 평단가 영역이 자동 계산됩니다.

현재 데일리 브리핑은 Yahoo Finance 기반으로 실적/통계/뉴스를 자동 수집하고,
Investing.com, 네이버증권, 증권플러스, 리포트 검색으로 바로 가는 링크를 함께 보여줍니다.
미국 종목은 네이버식 기관·외국인·개인 수급이 직접 제공되지 않으므로 검색 링크로 연결합니다.

영상 분석에서 추가한 복기 항목:

- 구름과 현재 가격의 이격
- 기준선과 현재 가격의 이격
- 20일선과 현재 가격의 이격
- 구름/기준선/20일선이 따라올 때까지 기다리는지 여부

## 퀴즈에서 답할 것

각 `question.png`를 보고 아래 4개 중 하나를 고릅니다.

- `상승 지속`
- `하락 지속/전환`
- `횡보/관망`
- `휩쏘 가능`

그 다음 `review.md`와 `answer.png`를 열어 실제 결과와 비교합니다.

## 이치모쿠 기본값

- 20일 이동평균선: 종가 기준 20일 평균, 추세 속도와 단기 과열 확인 보조
- 전환선: 9봉 고가/저가 중간값
- 기준선: 26봉 고가/저가 중간값
- 선행스팬 A: 전환선과 기준선의 중간값을 26봉 앞으로 표시
- 선행스팬 B: 52봉 고가/저가 중간값을 26봉 앞으로 표시
- 후행스팬: 현재 종가를 26봉 뒤로 표시

이 랩은 투자 조언이 아니라 학습과 검증용입니다.
