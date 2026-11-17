"""추세추종 백테스트 엔진 (02 노트북 로직을 재사용할 수 있게 모듈로 분리)

흐름: 지표 계산(indicators) → 판단일마다 신호 판단(entry/exit_signal)
      → 다음 날 시가 체결 · 매일 손절 체크 · 종가 평가(run)
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 파라미터 기본값 — 노트북에서 dict 로 덮어써서 실험
# ---------------------------------------------------------------------------
PARAMS = dict(
    MACD_FAST=12, MACD_SLOW=26, MACD_SIG=9,
    MONTH_MA=10, VOL_MA=20, OBV_MA=20, BB_N=20, BB_K=2.0, BB_POS_MIN=0.8,
    MA_LIST=(5, 10, 20, 60, 120, 240), PULLBACK_TOL=0.02,
    ATR_N=14, ATR_MULT=2.0, RISK_PER_TRADE=0.01, MAX_WEIGHT=0.20,
    DD_GUARD=-0.15, DD_SCALE=0.5, EVENT_BLOCK_DAYS=3,
    COST_BUY=0.0015, COST_SELL=0.0035, CASH_RATE=0.025,
)

# 전략 정의 — 새 규칙은 여기 한 줄 추가
LAYERS = {
    "L1": dict(entry="macd"),
    "L2": dict(entry="macd", vol_obv=True),
    "L3": dict(entry="macd", vol_obv=True, bb=True),
    "L4": dict(entry="macd", vol_obv=True, risk=True),
    "M1": dict(entry="macd", vol_obv=True, aligned=True),
    "M2": dict(entry="ma_cross", vol_obv=True),
    "M3": dict(entry="pullback", vol_obv=True),
    "M4": dict(entry="macd", vol_obv=True, exit="ma20"),
    "M5": dict(entry="macd", vol_obv=True, long_trend=True),
    "M6": dict(entry="macd", vol_obv=True, aligned=True, long_trend=True, exit="ma20"),
    "M7": dict(entry="ma_cross", vol_obv=True, aligned=True),
}

# 진입 신호: (지난 판단일 이후 있었어야 하는 이벤트, 오늘도 유지돼야 하는 상태)
ENTRY_EVENTS = {"macd": ("golden", "macd_above"),
                "ma_cross": ("ma_golden", "ma_above"),
                "pullback": ("pullback", "above20")}


# ---------------------------------------------------------------------------
# 1. 지표
# ---------------------------------------------------------------------------
def macd(s, p):
    line = s.ewm(span=p["MACD_FAST"], adjust=False).mean() - s.ewm(span=p["MACD_SLOW"], adjust=False).mean()
    return line, line.ewm(span=p["MACD_SIG"], adjust=False).mean()


def indicators(df, p=PARAMS):
    """pykrx 일봉(시가·고가·저가·종가·거래량) → 매매 판단에 쓰는 지표 표"""
    df = df[df["종가"] > 0].copy()
    c, h, l, v = df["종가"], df["고가"], df["저가"], df["거래량"]
    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"] = df["시가"], h, l, c
    out["value"] = c * v                                   # 거래대금(근사) — 유동성 리스크용

    # 월봉·주봉: 그날까지 완성된 봉만 사용 (미래 정보 X)
    m = c.resample("ME").last()
    out["month_up"] = (m > m.rolling(p["MONTH_MA"]).mean()).astype(float) \
        .reindex(out.index, method="ffill").fillna(0).astype(bool)
    w = c.resample("W-FRI").last()
    wl, ws = macd(w, p)
    out["week_up"] = ((wl - ws) > 0).astype(float).reindex(out.index, method="ffill").fillna(0).astype(bool)

    dl, ds = macd(c, p)
    out["macd_above"] = dl > ds
    out["golden"] = out["macd_above"] & ~out["macd_above"].shift(1, fill_value=False)

    out["vol_ok"] = v > v.rolling(p["VOL_MA"]).mean()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["obv_ok"] = obv > obv.rolling(p["OBV_MA"]).mean()

    mid, sd = c.rolling(p["BB_N"]).mean(), c.rolling(p["BB_N"]).std()
    out["bb_pos"] = (c - (mid - p["BB_K"] * sd)) / (2 * p["BB_K"] * sd)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(p["ATR_N"]).mean()

    for n in p["MA_LIST"]:
        out[f"ma{n}"] = c.rolling(n).mean()
    out["aligned"] = (out["ma5"] > out["ma20"]) & (out["ma20"] > out["ma60"]) & (out["ma60"] > out["ma120"])
    above = out["ma5"] > out["ma20"]
    out["ma_above"] = above
    out["ma_golden"] = above & ~above.shift(1, fill_value=False)
    tol = 1 + p["PULLBACK_TOL"]
    touch20 = (l <= out["ma20"] * tol) & (c > out["ma20"]) & (out["ma20"] > out["ma20"].shift(5))
    touch60 = (l <= out["ma60"] * tol) & (c > out["ma60"]) & (out["ma60"] > out["ma60"].shift(5))
    out["pullback"] = touch20 | touch60
    out["above20"] = c > out["ma20"]
    out["long_up"] = (c > out["ma120"]) & (c > out["ma240"])
    return out


# ---------------------------------------------------------------------------
# 2. 신호
# ---------------------------------------------------------------------------
def entry_signal(x, prev_day, day, cfg):
    if day not in x.index:
        return False
    win = x.loc[(x.index > prev_day) & (x.index <= day)]
    ev, keep = ENTRY_EVENTS[cfg["entry"]]
    r = x.loc[day]
    ok = win[ev].any() and r[keep] and r["month_up"] and r["week_up"]
    if cfg.get("vol_obv"):    ok = ok and r["vol_ok"] and r["obv_ok"]
    if cfg.get("bb"):         ok = ok and r["bb_pos"] >= PARAMS["BB_POS_MIN"]
    if cfg.get("aligned"):    ok = ok and r["aligned"]
    if cfg.get("long_trend"): ok = ok and r["long_up"]
    return bool(ok)


def exit_signal(x, day, cfg):
    if day not in x.index:
        return False
    r = x.loc[day]
    if cfg.get("exit") == "ma20":
        return not r["above20"]
    return (not r["macd_above"]) or (not r["week_up"])


def make_decision_days(days, start, end):
    """매월 1일·16일 이후 첫 거래일"""
    out = set()
    for m in pd.date_range(start, end, freq="MS"):
        for target in (m, m + pd.Timedelta(days=15)):
            after = days[days >= target]
            if len(after):
                out.add(after[0])
    return out


# ---------------------------------------------------------------------------
# 3. 엔진
# ---------------------------------------------------------------------------
def run(ind, yearly_uni, days, decision_days, layer, max_pos, p=PARAMS, events=None):
    """
    ind         : {종목코드: indicators() 결과}
    yearly_uni  : {연도: [그 해 매매 가능 종목]}
    반환        : equity(Series), trades(DataFrame), holdings(DataFrame: 날짜 × 종목 평가금액)
    """
    cfg = LAYERS[layer]
    risk = cfg.get("risk", False)
    cash, pos, trades, equity, hold_rows = 1.0, {}, [], [], []
    peak = 1.0
    pending_buy, pending_sell = [], []
    prev_dec = days[0] - pd.Timedelta(days=15)
    daily_cash = (1 + p["CASH_RATE"]) ** (1 / 252)

    def px(t, day, col):
        x = ind.get(t)
        return x.at[day, col] if x is not None and day in x.index else np.nan

    def blocked(day):
        if events is None:
            return False
        return ((events["date"] - day).abs() <= pd.Timedelta(days=p["EVENT_BLOCK_DAYS"])).any()

    def sell(t, day, price, reason):
        nonlocal cash
        q = pos.pop(t)
        cash += q["sh"] * price * (1 - p["COST_SELL"])
        ret = price * (1 - p["COST_SELL"]) / (q["entry"] * (1 + p["COST_BUY"])) - 1
        trades.append(dict(ticker=t, entry_day=q["day"], exit_day=day, ret=ret,
                           days=(day - q["day"]).days, reason=reason))

    for day in days:
        uni = yearly_uni.get(day.year, [])

        # (1) 전날 결정한 주문 → 오늘 시가 체결 (매도 먼저)
        for t in pending_sell:
            if t in pos:
                o = px(t, day, "open")
                sell(t, day, o if o == o else pos[t]["last"], "signal")
        port_val = cash + sum(q["sh"] * q["last"] for q in pos.values())
        for t, w in pending_buy:
            o = px(t, day, "open")
            if t in pos or not (o == o) or o <= 0:
                continue
            amt = min(w * port_val, cash)
            if amt <= port_val * 0.01:
                continue
            cash -= amt
            stop = o - p["ATR_MULT"] * px(t, day, "atr") if risk else -np.inf
            pos[t] = dict(sh=amt / (o * (1 + p["COST_BUY"])), entry=o, day=day, stop=stop, last=o)
        pending_buy, pending_sell = [], []

        # (2) 손절 (리스크 규칙 전략만)
        for t in list(pos):
            lo, o = px(t, day, "low"), px(t, day, "open")
            if lo == lo and lo <= pos[t]["stop"]:
                sell(t, day, min(o, pos[t]["stop"]) if o == o else pos[t]["stop"], "stop")

        # (3) 현금 이자 · 종가 평가
        cash *= daily_cash
        for t, q in pos.items():
            c = px(t, day, "close")
            if c == c:
                q["last"] = c
        val = cash + sum(q["sh"] * q["last"] for q in pos.values())
        equity.append((day, val))
        hold_rows.append({"date": day, "cash": cash, **{t: q["sh"] * q["last"] for t, q in pos.items()}})
        peak = max(peak, val)

        # (4) 판단일 → 내일 주문
        if day in decision_days:
            for t in list(pos):
                x = ind.get(t)
                if t not in uni or x is None or exit_signal(x, day, cfg):
                    pending_sell.append(t)
            slots = max_pos - (len(pos) - len(pending_sell))
            if slots > 0 and not (risk and blocked(day)):
                cands = [t for t in uni if t in ind and t not in pos
                         and entry_signal(ind[t], prev_dec, day, cfg)]
                mom = {t: ind[t]["close"].loc[:day].pct_change(250).iloc[-1] for t in cands}
                for t in sorted(cands, key=lambda t: -np.nan_to_num(mom[t], nan=-9))[:slots]:
                    if risk:
                        stop_dist = p["ATR_MULT"] * ind[t].at[day, "atr"] / ind[t].at[day, "close"]
                        w = min(p["RISK_PER_TRADE"] / stop_dist, p["MAX_WEIGHT"]) if stop_dist > 0 else 0
                        if val / peak - 1 <= p["DD_GUARD"]:
                            w *= p["DD_SCALE"]
                    else:
                        w = 1 / max_pos
                    pending_buy.append((t, w))
            prev_dec = day

    eq = pd.Series(dict(equity))
    holdings = pd.DataFrame(hold_rows).set_index("date").fillna(0.0)
    return eq, pd.DataFrame(trades), holdings


def run_hold(ind, yearly_uni, days, p=PARAMS):
    """L0: 해마다 1월에 유니버스를 동일가중으로 사서 1년 보유"""
    val, vals = 1.0, []
    for y in sorted(yearly_uni):
        uni = [t for t in yearly_uni[y] if t in ind]
        d = days[days.year == y]
        if not len(d) or not uni:
            continue
        rel = pd.DataFrame({t: ind[t]["close"].reindex(d).ffill() for t in uni})
        rel = rel / rel.bfill().iloc[0]
        path = val * rel.mean(axis=1) * (1 - p["COST_BUY"])
        vals.append(path)
        val = path.iloc[-1] * (1 - p["COST_SELL"])
    return pd.concat(vals)


# ---------------------------------------------------------------------------
# 4. 성과 지표
# ---------------------------------------------------------------------------
ORDER = ["CAGR", "Alpha", "Vol", "MDD", "VaR95_15d", "Beta", "Sharpe", "Calmar",
         "Trades", "WinRate", "Payoff", "Expectancy", "MaxLossStreak", "AvgHoldDays"]


def perf_metrics(eq, bench, trades=None, rf=0.03):
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (eq / eq.cummax() - 1).min()
    rm = bench.pct_change().reindex(r.index).fillna(0)
    beta = np.cov(r, rm)[0, 1] / rm.var()
    out = {"CAGR": cagr, "Alpha": (r - beta * rm).mean() * 252, "Vol": vol, "MDD": mdd,
           "VaR95_15d": eq.pct_change(10).dropna().quantile(0.05), "Beta": beta,
           "Sharpe": (cagr - rf) / vol, "Calmar": cagr / abs(mdd) if mdd else np.nan}
    if trades is not None and len(trades):
        win = trades["ret"] > 0
        aw, al = trades.loc[win, "ret"].mean(), -trades.loc[~win, "ret"].mean()
        out.update({"Trades": len(trades), "WinRate": win.mean(),
                    "Payoff": aw / al if al else np.nan,
                    "Expectancy": win.mean() * aw - (1 - win.mean()) * al,
                    "MaxLossStreak": (~win).astype(int).groupby(win.cumsum()).sum().max(),
                    "AvgHoldDays": trades["days"].mean()})
    return pd.Series(out)


def fmt_perf(t):
    t = t[[c for c in ORDER if c in t]]
    pct = ["CAGR", "Vol", "MDD", "VaR95_15d", "Alpha", "WinRate", "Expectancy"]
    num = ["Sharpe", "Calmar", "Beta", "Payoff", "AvgHoldDays"]
    return (t.style.format("{:.1%}", subset=[c for c in pct if c in t], na_rep="-")
             .format("{:.2f}", subset=[c for c in num if c in t], na_rep="-")
             .format("{:.0f}", subset=[c for c in ["Trades", "MaxLossStreak"] if c in t], na_rep="-"))
