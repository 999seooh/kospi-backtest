"""리스크 측정 모듈 — 증권사 리스크관리팀이 보는 지표들

모든 VaR/ES 는 '손실'을 양수(%)로 표시한다.  예) VaR99 = 0.025 → 하루 2.5% 넘게 잃을 확률 1%
"""
import math
import numpy as np
import pandas as pd

Z = {0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def _pdf(z):
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# 1. VaR · ES 시계열 (전략 수익률 자체로 계산 → 사후검증에 사용)
# ---------------------------------------------------------------------------
def var_hist(r, window=250, conf=0.99):
    """역사적 시뮬레이션: 과거 window 일 수익률의 하위 (1-conf) 분위수"""
    return -r.rolling(window).quantile(1 - conf)


def var_param(r, window=250, conf=0.99):
    """분산-공분산(정규분포 가정): z × σ − μ"""
    return Z[conf] * r.rolling(window).std() - r.rolling(window).mean()


def es_hist(r, window=250, conf=0.975):
    """기대손실(ES): VaR 를 넘는 손실들의 평균 — '꼬리가 얼마나 두꺼운가'"""
    def f(a):
        q = np.quantile(a, 1 - conf)
        return -a[a <= q].mean()
    return r.rolling(window).apply(f, raw=True)


def es_param(r, window=250, conf=0.975):
    z = Z[conf]
    return r.rolling(window).std() * _pdf(z) / (1 - conf) - r.rolling(window).mean()


# ---------------------------------------------------------------------------
# 2. VaR 사후검증 (백테스팅) — 모델이 위험을 제대로 잡았나?
# ---------------------------------------------------------------------------
def exceptions(r, var):
    """다음 날 실제 손실이 전날 계산한 VaR 를 넘은 날 = 예외(exception)"""
    v = var.shift(1)
    m = v.notna() & r.notna()
    return (-r[m] > v[m])


def kupiec(n, x, conf):
    """Kupiec POF 검정: 예외 비율이 (1-conf) 와 통계적으로 같은가?  p < 0.05 면 모델 기각"""
    p = 1 - conf
    if n == 0:
        return np.nan, np.nan
    ph = x / n
    def ll(q):
        a = (n - x) * math.log(1 - q) if n - x > 0 else 0.0
        b = x * math.log(q) if x > 0 else 0.0
        return a + b
    lr = -2 * (ll(p) - (ll(ph) if 0 < ph < 1 else 0.0))
    lr = max(lr, 0.0)
    return lr, math.erfc(math.sqrt(lr / 2))          # 자유도 1 카이제곱 p-value


def traffic_light(x_250):
    """바젤 신호등 (99% VaR, 최근 250일 예외 수): 0~4 녹색, 5~9 황색, 10+ 적색"""
    return "🟢 Green" if x_250 <= 4 else ("🟡 Yellow" if x_250 <= 9 else "🔴 Red")


def backtest_table(r, models, conf=0.99):
    """models: {이름: VaR 시계열}"""
    rows = {}
    for name, v in models.items():
        e = exceptions(r, v)
        n, x = len(e), int(e.sum())
        lr, pv = kupiec(n, x, conf)
        last = int(e.iloc[-250:].sum())
        rows[name] = {"관측일": n, "예외 수": x, "예상 예외 수": n * (1 - conf),
                      "예외율": x / n if n else np.nan, "Kupiec LR": lr, "p-value": pv,
                      "판정(5%)": "통과" if pv >= 0.05 else "기각",
                      "최근250일 예외": last,
                      "바젤 신호등": traffic_light(last) if conf == 0.99 else "-"}   # 바젤 기준은 99% 전용
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 3. 현재 보유 포트폴리오 기준 VaR — 3가지 방법 비교
# ---------------------------------------------------------------------------
def position_returns(ind, tickers, end, window=250):
    px = pd.DataFrame({t: ind[t]["close"].loc[:end] for t in tickers}).iloc[-(window + 1):]
    return px.pct_change().dropna(how="all").fillna(0)


def portfolio_var(R, w, conf=0.99, es_conf=0.975, n_sims=20000, seed=0):
    """
    R : 보유 종목 일별 수익률(행=날짜, 열=종목),  w : 계좌 대비 비중 (현금 몫은 위험 0)
    반환: 방법별 VaR, ES
    """
    w = np.asarray(w, float)
    port = R.values @ w
    mu, cov = R.mean().values, R.cov().values
    sig = math.sqrt(max(w @ cov @ w, 0.0))
    rng = np.random.default_rng(seed)
    sim = rng.multivariate_normal(mu, cov, size=n_sims, method="svd") @ w

    def h_var(a, c): return -np.quantile(a, 1 - c)
    def h_es(a, c):
        q = np.quantile(a, 1 - c)
        return -a[a <= q].mean()

    return pd.DataFrame({
        "역사적 시뮬레이션": [h_var(port, 0.95), h_var(port, conf), h_es(port, es_conf)],
        "분산-공분산(정규)": [Z[0.95] * sig - w @ mu, Z[conf] * sig - w @ mu,
                         sig * _pdf(Z[es_conf]) / (1 - es_conf) - w @ mu],
        "몬테카를로": [h_var(sim, 0.95), h_var(sim, conf), h_es(sim, es_conf)],
    }, index=["VaR 95%", f"VaR {conf:.0%}", f"ES {es_conf:.1%}"])


def component_var(R, w, conf=0.99):
    """리스크 기여도: 각 종목이 포트폴리오 VaR 에 얼마나 기여하나 (합 = 전체 VaR)"""
    w = np.asarray(w, float)
    cov = R.cov().values
    sig = math.sqrt(w @ cov @ w)
    comp = w * (cov @ w) / sig * Z[conf]
    out = pd.DataFrame({"비중": w, "단독 변동성(연)": np.sqrt(np.diag(cov) * 252),
                        "VaR 기여": comp}, index=R.columns)
    out["VaR 기여 비율"] = out["VaR 기여"] / out["VaR 기여"].sum()
    out["분산효과"] = out["VaR 기여 비율"] - out["비중"] / out["비중"].sum()
    return out.sort_values("VaR 기여", ascending=False)


# ---------------------------------------------------------------------------
# 4. 스트레스 테스트
# ---------------------------------------------------------------------------
def window_return(s, start, end):
    s = s.loc[:end]
    before = s.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)]
    if before.empty or s.empty:
        return np.nan
    return s.iloc[-1] / before.iloc[-1] - 1


def stress_historical(ind, kospi, weights, betas, scenarios):
    """
    '지금 포트폴리오가 그때 그 사건을 다시 겪으면?'
    종목별 그 기간 실제 수익률 × 현재 비중. 그때 상장 전이던 종목은 베타 × KOSPI 로 대체
    """
    rows = {}
    for name, (s, e) in scenarios.items():
        k = window_return(kospi, s, e)
        loss, proxy = 0.0, 0
        for t, w in weights.items():
            rt = window_return(ind[t]["close"], s, e) if t in ind else np.nan
            if not (rt == rt):
                rt, proxy = betas.get(t, 1.0) * k, proxy + 1
            loss += w * rt
        rows[name] = {"기간": f"{s} ~ {e}", "KOSPI": k, "현재 포트폴리오": loss,
                      "KOSPI 대비": loss - k, "베타 대체 종목 수": proxy}
    return pd.DataFrame(rows).T


def stress_hypothetical(port_beta, shocks=(-0.05, -0.10, -0.20, -0.30)):
    """가상 시나리오: KOSPI 가 x% 빠지면 베타만큼 손실"""
    return pd.DataFrame({"KOSPI 충격": list(shocks),
                         "예상 손실(베타 기준)": [port_beta * s for s in shocks]}).set_index("KOSPI 충격")


def betas_to(ind, tickers, kospi, end, window=250):
    rm = kospi.loc[:end].pct_change().iloc[-window:]
    out = {}
    for t in tickers:
        r = ind[t]["close"].loc[:end].pct_change().reindex(rm.index)
        m = r.notna() & rm.notna()
        out[t] = np.cov(r[m], rm[m])[0, 1] / rm[m].var() if m.sum() > 60 else 1.0
    return out


# ---------------------------------------------------------------------------
# 5. 유동성 리스크
# ---------------------------------------------------------------------------
def liquidity(ind, weights, end, aum, participation=0.10, adv_days=20):
    """보유 금액을 하루 거래대금의 participation(10%)씩만 판다면 며칠 걸리나"""
    rows = {}
    for t, w in weights.items():
        adv = ind[t]["value"].loc[:end].iloc[-adv_days:].mean()
        pos = w * aum
        rows[t] = {"보유금액(억)": pos / 1e8, "20일 평균 거래대금(억)": adv / 1e8,
                   "청산 소요일": pos / (participation * adv) if adv > 0 else np.inf}
    return pd.DataFrame(rows).T.sort_values("청산 소요일", ascending=False)


# ---------------------------------------------------------------------------
# 6. 한도 관리
# ---------------------------------------------------------------------------
def limit_table(items):
    """items: [(항목, 현재값, 한도, '이하'/'이상')]"""
    rows = []
    for name, val, lim, kind in items:
        ok = val <= lim if kind == "이하" else val >= lim
        usage = val / lim if lim else np.nan
        status = "🟢 정상" if ok and usage < 0.8 else ("🟡 주의(80%↑)" if ok else "🔴 초과")
        rows.append({"항목": name, "현재": val, "한도": lim, "기준": kind, "한도 사용률": usage, "상태": status})
    return pd.DataFrame(rows).set_index("항목")
