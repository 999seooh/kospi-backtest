# kospi-backtest

**질문:** 보험업 현업 지표와 언더라이팅식 "리스크 선별" 사고방식이 종목 선택에 도움이 되는가?

- 보험주 H1: 합산비율 개선 보험사 → 시장 초과수익? (FISIS 데이터, 02 노트북)
- 보험주 H2: 저PBR·고ROE 보험사 → 보험주 전체 대비 초과수익?
- 비보험주 H1: 저PER·고ROE → KOSPI 대비 초과수익?
- 비보험주 H2: 고변동성·적자 종목 제외 → 낙폭(MDD) 감소?

이후 단계: 모의투자 → 실전 소액(보험 2·비보험 2종목) → 백테스트 대비 괴리 분석

## 실행
```bash
pip install -r requirements.txt
cp .env.example .env   # KRX 아이디/비번 입력
jupyter notebook
```
