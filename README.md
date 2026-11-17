# kospi-backtest

**질문:** 보험업 현업 지표와 언더라이팅식 "리스크 선별" 사고방식이 종목 선택에 도움이 되는가?

- 보험주 H1: 합산비율 개선 보험사 → 시장 초과수익? (FISIS 데이터, 02 노트북)
- 보험주 H2: 저PBR·고ROE 보험사 → 보험주 전체 대비 초과수익?
- 비보험주 H1: 저PER·고ROE → KOSPI 대비 초과수익?
- 비보험주 H2: 고변동성·적자 종목 제외 → 낙폭(MDD) 감소?

## 노트북
- `01_first_backtest` — 기본 전략: 재무지표(PER·PBR·ROE)로 종목 선택, 15일 리밸런싱
- `02_trend_following` — 내 매매 방식: 월봉·주봉 추세 + MACD + 거래량·OBV + 볼린저, 리스크 규칙(ATR 손절·비중조절·낙폭 방어·이벤트 차단)을 층층이 추가하며 비교
- `03_kospi_strategy_risk` — **KOSPI 시총 상위 100 전략 백테스트 + 리스크관리팀 보고서형 리스크 측정**: VaR 3종(역사적·분산공분산·몬테카를로)·ES, VaR 사후검증(Kupiec·바젤 신호등), 리스크 기여도, 스트레스 테스트(코로나·계엄 등 재현 + 가상 충격), 유동성 리스크, 한도 대시보드, 자동 요약
- `src/backtest.py`, `src/risk.py` — 백테스트 엔진·리스크 계산 함수 (노트북에서 import)
- `data_events/events.csv` — 선거·정책·글로벌 이벤트 목록 (미래 일정 추가하면 경고에 반영)

이후 단계: 모의투자 → 실전 소액(보험 2·비보험 2종목) → 백테스트 대비 괴리 분석

## 실행
```bash
pip install -r requirements.txt
cp .env.example .env   # KRX 아이디/비번 입력
jupyter notebook
```
