# -*- coding: utf-8 -*-
"""
qant_recipes — 금융 지표 이름 인덱스 v0 (레시피 사전).

목적 (T10 교훈): 사용자가 "RSI", "볼린저밴드"라고 말할 때 그 이름을 현재
레지스트리 어휘의 전개식(AST)과 연결한다. 이름 검색 실패로 인한 거짓 거절을
과제별 예외가 아니라 일반 사전으로 해결한다 (피드백 §12).

원칙:
  - TA-Lib 을 대형 수식 코퍼스로 가져오지 않는다. 이름 -> 별칭 -> 전개식의
    골드 사전으로만 쓴다 (피드백 §2).
  - 각 레시피의 expanded_dsl 은 (1) DSL 파싱 (2) 레지스트리 검증 0 에러
    (3) 실행기 실행 (4) 독립 pandas 기준 구현과 수치 비교를 전부 통과해야 한다.
    비교 결과로 expansion_status 를 '측정해서' 기록한다 (선언하지 않는다):
      exact       정상상태에서 최대 오차 < 1e-8
      approximate 정상상태 상관 > 0.995 (원인은 notes 에 명시)
      mismatch    회귀 실패 대상
  - 재귀·상태 기반 지표(ADX, SAR, KAMA 등)는 억지로 전개하지 않고
    extension_queue 에 둔다 (피드백 §2).

정상상태(steady-state): EMA 계열은 초기 시드 차이가 지수적으로 소멸하므로
종목별 첫 유효값 이후 5*window 를 버리고 비교한다. 누적 지표(OBV/AD)는
시작점 상수 차이가 있으므로 종목별 평균 제거 후 비교한다.

실행:  python qant_recipes.py           # 검증 + qant_out/qant_recipes.json
       python qant_recipes.py --list    # 상태 요약만
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qant_dsl import parse as dsl_parse, to_dsl
from qant_ir import normalize, NORMALIZER_VERSION
from qant_validate import derive_spec, validate
from qant_eval import make_panel, ev

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "qant_out"; OUTDIR.mkdir(exist_ok=True)

RECIPES_VERSION = "0.1.0"

# ---------------------------------------------------------------- 레시피 정의
# template: {n} 등 자리표시자. expanded_dsl 은 default_params 대입으로 생성.
# ref: 독립 기준 구현 키 (아래 REFS). cumulative=True 면 종목별 demean 비교.
_TP = "((high + low + close) / 3)"                      # typical price
_TR = ("max(high - low, max(abs(high - ts_delay(close, 1)), "
       "abs(low - ts_delay(close, 1))))")               # true range
_DC = "ts_delta(close, 1)"

RECIPES = [
 dict(name="SMA", aliases=["단순이동평균", "이동평균", "simple moving average", "이평선"],
      default_params={"n": 20}, inputs=["close"],
      template="ts_mean(close, {n})", ref="sma"),
 dict(name="EMA", aliases=["지수이동평균", "exponential moving average", "지수평활"],
      default_params={"n": 20}, inputs=["close"],
      template="ts_ewm(close, {n})", ref="ema",
      notes="실행기 ewm 은 adjust=True — 재귀 정의(SMA 시드)와 초기 구간이 다르고 지수적으로 수렴"),
 dict(name="WMA", aliases=["가중이동평균", "weighted moving average", "선형가중"],
      default_params={"n": 20}, inputs=["close"],
      template="ts_decay_linear(close, {n})", ref="wma"),
 dict(name="RSI", aliases=["상대강도지수", "relative strength index", "과매수", "과매도", "오실레이터"],
      default_params={"n": 14}, inputs=["close"],
      template=("100 * ts_sma_cn(max({dc}, 0), {n}, 1) / "
                "(ts_sma_cn(max({dc}, 0), {n}, 1) + ts_sma_cn(max(0 - {dc}, 0), {n}, 1) + 1e-12)"
                ).replace("{dc}", _DC), ref="rsi",
      notes="Wilder 평활 = ts_sma_cn(x, n, 1). 시드(SMA vs 첫값) 차이는 정상상태에서 소멸"),
 dict(name="MACD", aliases=["macd", "이동평균수렴확산", "moving average convergence divergence"],
      default_params={"fast": 12, "slow": 26}, inputs=["close"],
      template="ts_ewm(close, {fast}) - ts_ewm(close, {slow})", ref="macd",
      notes="adjust=True vs 재귀 시드 차이 — 정상상태 수렴"),
 dict(name="MACD_SIGNAL", aliases=["macd 시그널", "시그널선", "macd signal"],
      default_params={"fast": 12, "slow": 26, "sig": 9}, inputs=["close"],
      template="ts_ewm(ts_ewm(close, {fast}) - ts_ewm(close, {slow}), {sig})", ref="macd_signal",
      notes="adjust=True vs 재귀 시드 차이 — 정상상태 수렴"),
 dict(name="PPO", aliases=["ppo", "percentage price oscillator", "가격 오실레이터"],
      default_params={"fast": 12, "slow": 26}, inputs=["close"],
      template="100 * (ts_ewm(close, {fast}) - ts_ewm(close, {slow})) / ts_ewm(close, {slow})",
      ref="ppo", notes="adjust=True vs 재귀 시드 차이 — 정상상태 수렴"),
 dict(name="ROC", aliases=["변화율", "rate of change", "등락률", "수익률"],
      default_params={"n": 10}, inputs=["close"],
      template="100 * ts_returns(close, {n})", ref="roc",
      notes="ts_returns 재작성(0.4.0)의 대표 수요처"),
 dict(name="MOM", aliases=["모멘텀", "momentum", "가격차"],
      default_params={"n": 10}, inputs=["close"],
      template="ts_delta(close, {n})", ref="mom"),
 dict(name="BB_UPPER", aliases=["볼린저밴드 상단", "볼린저 상단", "bollinger upper", "상단밴드"],
      default_params={"n": 20, "k": 2}, inputs=["close"],
      template="ts_mean(close, {n}) + {k} * ts_std(close, {n})", ref="bb_upper",
      notes="ts_std 는 표본표준편차(ddof=1). TA-Lib 은 모표준편차(ddof=0) — 상관은 1에 가까움"),
 dict(name="BB_LOWER", aliases=["볼린저밴드 하단", "볼린저 하단", "bollinger lower", "하단밴드"],
      default_params={"n": 20, "k": 2}, inputs=["close"],
      template="ts_mean(close, {n}) - {k} * ts_std(close, {n})", ref="bb_lower",
      notes="ddof 주석은 BB_UPPER 와 동일"),
 dict(name="BB_PERCENT_B", aliases=["볼린저밴드", "bollinger bands", "%b", "밴드 위치"],
      default_params={"n": 20, "k": 2}, inputs=["close"],
      template=("(close - (ts_mean(close, {n}) - {k} * ts_std(close, {n}))) / "
                "(2 * {k} * ts_std(close, {n}) + 1e-12)"), ref="bb_pctb",
      notes="밴드 내 위치 0~1. ddof 주석 동일"),
 dict(name="TRANGE", aliases=["true range", "트루레인지", "실질변동폭"],
      default_params={}, inputs=["high", "low", "close"],
      template=_TR, ref="trange"),
 dict(name="ATR", aliases=["평균실질변동폭", "average true range", "변동폭"],
      default_params={"n": 14}, inputs=["high", "low", "close"],
      template=f"ts_sma_cn({_TR}, {{n}}, 1)", ref="atr",
      notes="Wilder 평활. 시드 차이는 정상상태에서 소멸"),
 dict(name="NATR", aliases=["정규화 atr", "normalized atr"],
      default_params={"n": 14}, inputs=["high", "low", "close"],
      template=f"100 * ts_sma_cn({_TR}, {{n}}, 1) / close", ref="natr",
      notes="ATR 주석과 동일"),
 dict(name="STOCH_K", aliases=["스토캐스틱", "stochastic", "%k", "패스트 스토캐스틱"],
      default_params={"n": 14}, inputs=["high", "low", "close"],
      template=("100 * (close - ts_min(low, {n})) / "
                "(ts_max(high, {n}) - ts_min(low, {n}) + 1e-12)"), ref="stoch_k"),
 dict(name="STOCH_D", aliases=["스토캐스틱 시그널", "%d", "슬로우 스토캐스틱"],
      default_params={"n": 14, "d": 3}, inputs=["high", "low", "close"],
      template=("ts_mean(100 * (close - ts_min(low, {n})) / "
                "(ts_max(high, {n}) - ts_min(low, {n}) + 1e-12), {d})"), ref="stoch_d"),
 dict(name="WILLR", aliases=["윌리엄스", "williams %r", "윌리엄스 퍼센트"],
      default_params={"n": 14}, inputs=["high", "low", "close"],
      template=("(0 - 100) * (ts_max(high, {n}) - close) / "
                "(ts_max(high, {n}) - ts_min(low, {n}) + 1e-12)"), ref="willr"),
 dict(name="CCI", aliases=["상품채널지수", "commodity channel index"],
      default_params={"n": 14}, inputs=["high", "low", "close"],
      template=(f"({_TP} - ts_mean({_TP}, {{n}})) / "
                f"(0.015 * ts_mean(abs({_TP} - ts_mean({_TP}, {{n}})), {{n}}) + 1e-12)"),
      ref="cci",
      notes="TA-Lib 평균편차는 창 고정 SMA 기준 — 본 전개는 롤링 SMA 편차의 롤링 평균 (근사)"),
 dict(name="MFI", aliases=["자금흐름지수", "money flow index", "자금흐름"],
      default_params={"n": 14}, inputs=["high", "low", "close", "volume"],
      template=("100 * ts_sum(where(gt({tp}, ts_delay({tp}, 1)), {tp} * volume, 0), {n}) / "
                "(ts_sum(where(gt({tp}, ts_delay({tp}, 1)), {tp} * volume, 0), {n}) + "
                "ts_sum(where(lt({tp}, ts_delay({tp}, 1)), {tp} * volume, 0), {n}) + 1e-12)"
                ).replace("{tp}", _TP), ref="mfi"),
 dict(name="OBV", aliases=["온밸런스볼륨", "on balance volume", "거래량균형"],
      default_params={}, inputs=["close", "volume"],
      template="ts_cumsum(sign(ts_delta(close, 1)) * volume, 1)", ref="obv",
      cumulative=True, notes="시작점 상수(첫날 거래량 시드)만 다름 — demean 비교"),
 dict(name="AD", aliases=["누적분산", "accumulation distribution", "차이킨 ad"],
      default_params={}, inputs=["high", "low", "close", "volume"],
      template=("ts_cumsum(((close - low) - (high - close)) / "
                "(high - low + 1e-12) * volume, 1)"), ref="ad", cumulative=True),
 dict(name="ADOSC", aliases=["차이킨 오실레이터", "chaikin oscillator", "adosc"],
      default_params={"fast": 3, "slow": 10}, inputs=["high", "low", "close", "volume"],
      template=("ts_ewm(ts_cumsum(((close - low) - (high - close)) / "
                "(high - low + 1e-12) * volume, 1), {fast}) - "
                "ts_ewm(ts_cumsum(((close - low) - (high - close)) / "
                "(high - low + 1e-12) * volume, 1), {slow})"), ref="adosc",
      notes="누적 신호의 EMA 차 — adjust 시드 차이는 정상상태 수렴"),
 dict(name="AROON_UP", aliases=["아룬 업", "aroon up"],
      default_params={"n": 25, "n1": 26}, inputs=["high"],
      template="100 * ({n} - ts_argmax(high, {n1})) / {n}", ref="aroon_up",
      notes="ts_argmax 는 '최대였던 날이 며칠 전인가' 반환 — 창 n1 = n+1 (DSL window 는 상수만 허용)"),
 dict(name="AROON_DOWN", aliases=["아룬 다운", "aroon down"],
      default_params={"n": 25, "n1": 26}, inputs=["low"],
      template="100 * ({n} - ts_argmin(low, {n1})) / {n}", ref="aroon_down"),
 dict(name="AROON_OSC", aliases=["아룬", "aroon", "아룬 오실레이터"],
      default_params={"n": 25, "n1": 26}, inputs=["high", "low"],
      template=("100 * ({n} - ts_argmax(high, {n1})) / {n} - "
                "100 * ({n} - ts_argmin(low, {n1})) / {n}"), ref="aroon_osc"),
 dict(name="BETA", aliases=["베타", "회귀 베타", "시장 베타", "beta"],
      default_params={"n": 60}, inputs=["close", "vwap"],
      template=("ts_cov(ts_returns(close, 1), ts_returns(vwap, 1), {n}) / "
                "(ts_var(ts_returns(vwap, 1), {n}) + 1e-12)"), ref="beta",
      notes="일반형 beta(x, y, n). 시장지수 필드가 없어 예시는 vwap 대비 — "
            "실사용은 시장 필드 추가 후 (extension)"),
 dict(name="CORREL", aliases=["상관계수", "correlation", "상관"],
      default_params={"n": 30}, inputs=["close", "volume"],
      template="ts_corr(close, volume, {n})", ref="correl"),
 dict(name="LINEARREG_SLOPE", aliases=["회귀 기울기", "linear regression slope", "추세 기울기"],
      default_params={"n": 14}, inputs=["close"],
      template="ts_slope(close, {n})", ref="lr_slope"),
 dict(name="STDDEV", aliases=["표준편차", "standard deviation", "변동성"],
      default_params={"n": 20}, inputs=["close"],
      template="ts_std(close, {n})", ref="stddev",
      notes="표본표준편차(ddof=1). TA-Lib 기본은 ddof=0. "
            "변동성 팩터 의도라면 가격이 아니라 수익률에 적용할 것 (W_STD_PRICE)"),
 dict(name="VAR", aliases=["분산", "variance"],
      default_params={"n": 20}, inputs=["close"],
      template="ts_var(close, {n})", ref="var", notes="STDDEV 주석과 동일"),
 dict(name="TRIX", aliases=["트릭스", "trix", "삼중 지수평활"],
      default_params={"n": 15}, inputs=["close"],
      template="100 * ts_returns(ts_ewm(ts_ewm(ts_ewm(close, {n}), {n}), {n}), 1)",
      ref="trix", notes="삼중 EMA 의 1기간 변화율 — adjust 시드 차이는 정상상태 수렴"),
]

# 처음부터 프리미티브로 전개하지 않는 재귀·상태 기반 지표 (피드백 §2)
EXTENSION_QUEUE = [
    dict(name="ADX", reason="Wilder DI 재귀 + 이중 평활 — 상태 기반"),
    dict(name="SAR", reason="가속계수 상태머신 — AST 전개 불가"),
    dict(name="KAMA", reason="효율비 기반 가변 alpha — 상태 기반"),
    dict(name="MAMA", reason="Hilbert 변환 계열"),
    dict(name="T3", reason="6중 EMA 합성 — 전개 가능하나 검증 비용 대비 후순위"),
    dict(name="STOCHRSI", reason="RSI 의 롤링 min/max 재정규화 — 전개 가능, 다음 배치"),
    dict(name="ULTOSC", reason="3중 창 가중 합성 — 전개 가능, 다음 배치"),
    dict(name="HT_TRENDLINE", reason="Hilbert 변환 계열"),
]


# ---------------------------------------------------------------- 기준 구현
# 전부 '풀 윈도우' (min_periods=n) pandas 로 독립 구현. 실행기와 코드 공유 없음.
def _G(s):
    return s.groupby(level="code", group_keys=False)


def _rollf(s, n, fn, **kw):
    return _G(s).transform(lambda x: getattr(x.rolling(int(n), min_periods=int(n)), fn)(**kw))


def _rec(s, alpha, seed_n):
    """재귀 평활 y_t=(1-a)y_{t-1}+a*x_t, 처음 seed_n 개 평균으로 시드."""
    def one(x):
        v = x.to_numpy(float)
        out = np.full(len(v), np.nan)
        fin = np.where(np.isfinite(v))[0]
        if len(fin) < seed_n:
            return pd.Series(out, index=x.index)
        i0 = fin[0]
        seed_end = i0 + seed_n
        if not np.all(np.isfinite(v[i0:seed_end])):
            return pd.Series(out, index=x.index)
        prev = float(np.mean(v[i0:seed_end]))
        out[seed_end - 1] = prev
        for i in range(seed_end, len(v)):
            if not np.isfinite(v[i]):
                continue
            prev = (1 - alpha) * prev + alpha * v[i]
            out[i] = prev
        return pd.Series(out, index=x.index)
    return _G(s).apply(one)


def _ema_ref(s, n):
    return _rec(s, 2.0 / (n + 1), int(n))


def _wilder_ref(s, n):
    return _rec(s, 1.0 / n, int(n))


def _tp(df):
    return (df["high"] + df["low"] + df["close"]) / 3


def _trange_ref(df):
    pc = _G(df["close"]).shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                      (df["low"] - pc).abs()], axis=1).max(axis=1)


def _stoch_k_ref(df, n):
    hh = _rollf(df["high"], n, "max")
    ll = _rollf(df["low"], n, "min")
    return 100 * (df["close"] - ll) / (hh - ll)


def _obv_ref(df):
    sgn = np.sign(_G(df["close"]).diff())
    return _G(sgn * df["volume"]).cumsum()


def _ad_ref(df):
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    return _G(clv * df["volume"]).cumsum()


def _days_since(s, w, which):
    fn = np.argmax if which == "max" else np.argmin
    return _G(s).transform(lambda x: x.rolling(int(w), min_periods=int(w))
                           .apply(lambda a: len(a) - 1 - int(fn(a)), raw=True))


REFS = {
    "sma": lambda df, p: _rollf(df["close"], p["n"], "mean"),
    "ema": lambda df, p: _ema_ref(df["close"], p["n"]),
    "wma": lambda df, p: _G(df["close"]).transform(
        lambda x: x.rolling(int(p["n"]), min_periods=int(p["n"]))
        .apply(lambda a: float(np.dot(a, np.arange(1, len(a) + 1)) /
                               np.arange(1, len(a) + 1).sum()), raw=True)),
    "rsi": lambda df, p: (lambda up, dn: 100 * up / (up + dn))(
        _wilder_ref((_G(df["close"]).diff()).clip(lower=0), p["n"]),
        _wilder_ref((-_G(df["close"]).diff()).clip(lower=0), p["n"])),
    "macd": lambda df, p: _ema_ref(df["close"], p["fast"]) - _ema_ref(df["close"], p["slow"]),
    "macd_signal": lambda df, p: _ema_ref(
        _ema_ref(df["close"], p["fast"]) - _ema_ref(df["close"], p["slow"]), p["sig"]),
    "ppo": lambda df, p: 100 * (_ema_ref(df["close"], p["fast"]) -
                                _ema_ref(df["close"], p["slow"])) / _ema_ref(df["close"], p["slow"]),
    "roc": lambda df, p: 100 * _G(df["close"]).pct_change(int(p["n"])),
    "mom": lambda df, p: _G(df["close"]).diff(int(p["n"])),
    "bb_upper": lambda df, p: _rollf(df["close"], p["n"], "mean") +
        p["k"] * _rollf(df["close"], p["n"], "std"),
    "bb_lower": lambda df, p: _rollf(df["close"], p["n"], "mean") -
        p["k"] * _rollf(df["close"], p["n"], "std"),
    "bb_pctb": lambda df, p: (df["close"] - (_rollf(df["close"], p["n"], "mean") -
        p["k"] * _rollf(df["close"], p["n"], "std"))) /
        (2 * p["k"] * _rollf(df["close"], p["n"], "std")),
    "trange": lambda df, p: _trange_ref(df),
    "atr": lambda df, p: _wilder_ref(_trange_ref(df), p["n"]),
    "natr": lambda df, p: 100 * _wilder_ref(_trange_ref(df), p["n"]) / df["close"],
    "stoch_k": lambda df, p: _stoch_k_ref(df, p["n"]),
    "stoch_d": lambda df, p: _rollf(_stoch_k_ref(df, p["n"]), p["d"], "mean"),
    "willr": lambda df, p: -100 * (_rollf(df["high"], p["n"], "max") - df["close"]) /
        (_rollf(df["high"], p["n"], "max") - _rollf(df["low"], p["n"], "min")),
    "cci": lambda df, p: (lambda tp, ma: (tp - ma) /
        (0.015 * _rollf((tp - ma).abs(), p["n"], "mean")))(_tp(df), _rollf(_tp(df), p["n"], "mean")),
    "mfi": lambda df, p: (lambda tp, mf: (lambda pos, neg:
        100 * _rollf(pos, p["n"], "sum") /
        (_rollf(pos, p["n"], "sum") + _rollf(neg, p["n"], "sum")))(
        mf.where(tp > _G(tp).shift(1), 0.0), mf.where(tp < _G(tp).shift(1), 0.0)))(
        _tp(df), _tp(df) * df["volume"]),
    "obv": lambda df, p: _obv_ref(df),
    "ad": lambda df, p: _ad_ref(df),
    "adosc": lambda df, p: _ema_ref(_ad_ref(df), p["fast"]) - _ema_ref(_ad_ref(df), p["slow"]),
    "aroon_up": lambda df, p: 100 * (p["n"] - _days_since(df["high"], p["n"] + 1, "max")) / p["n"],
    "aroon_down": lambda df, p: 100 * (p["n"] - _days_since(df["low"], p["n"] + 1, "min")) / p["n"],
    "aroon_osc": lambda df, p:
        100 * (p["n"] - _days_since(df["high"], p["n"] + 1, "max")) / p["n"] -
        100 * (p["n"] - _days_since(df["low"], p["n"] + 1, "min")) / p["n"],
    "beta": lambda df, p: (lambda rx, rm:
        _ts2_ref(rx, rm, p["n"], "cov") / _rollf(rm, p["n"], "var"))(
        _G(df["close"]).pct_change(), _G(df["vwap"]).pct_change()),
    "correl": lambda df, p: _ts2_ref(df["close"], df["volume"], p["n"], "corr"),
    "lr_slope": lambda df, p: _G(df["close"]).transform(
        lambda x: x.rolling(int(p["n"]), min_periods=int(p["n"]))
        .apply(lambda a: np.polyfit(np.arange(len(a), dtype=float), a, 1)[0], raw=True)),
    "stddev": lambda df, p: _rollf(df["close"], p["n"], "std"),
    "var": lambda df, p: _rollf(df["close"], p["n"], "var"),
    "trix": lambda df, p: (lambda e3: 100 * _G(e3).pct_change())(
        _ema_ref(_ema_ref(_ema_ref(df["close"], p["n"]), p["n"]), p["n"])),
}


def _ts2_ref(x, y, w, fn):
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    out = j.groupby(level="code", group_keys=False).apply(
        lambda g: getattr(g["x"].rolling(int(w), min_periods=int(w)), fn)(g["y"]))
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
        out = out.reset_index(level=0, drop=True)
    return out.reindex(x.index)


# ---------------------------------------------------------------- 검증
def _steady_mask(df, burn):
    """종목별 첫 유효 close 이후 burn 행을 버리는 마스크."""
    ok = df["close"].notna()
    rank = ok.groupby(level="code").cumsum()          # 유효 관측 순번
    return (rank > burn) & ok


def verify_recipe(rec, df, spec):
    """레시피 하나: 파싱 -> 검증 -> 실행 -> 기준 비교. 결과 dict 반환."""
    out = dict(name=rec["name"])
    expanded = rec["template"].format(**rec["default_params"])
    out["expanded_dsl"] = expanded
    try:
        ir = normalize(dsl_parse(expanded))
    except Exception as e:
        out.update(expansion_status="mismatch", error=f"파싱 실패: {str(e)[:80]}")
        return out, None
    out["canonical_dsl"] = to_dsl(ir)
    errs, warns, _ = validate(ir, spec)
    out["n_validate_errors"] = len(errs)
    if errs:
        out.update(expansion_status="mismatch",
                   error=f"검증 에러: {[c for c, _, _ in errs][:3]}")
        return out, ir
    try:
        got = ev(ir, df)
    except Exception as e:
        out.update(expansion_status="mismatch", error=f"실행 실패: {str(e)[:80]}")
        return out, ir
    ref = REFS[rec["ref"]](df, rec["default_params"])

    burn = 5 * max([v for v in rec["default_params"].values() if isinstance(v, (int, float))] or [20])
    burn = min(int(burn), 200)          # 패널 길이 대비 상한
    mask = _steady_mask(df, burn)
    j = pd.concat([got.rename("g"), ref.rename("r")], axis=1)[mask]
    j = j.replace([np.inf, -np.inf], np.nan).dropna()
    if len(j) < 200:
        out.update(expansion_status="mismatch", error=f"비교 표본 부족 {len(j)}")
        return out, ir
    if rec.get("cumulative"):
        j = j - j.groupby(level="code").transform("mean")
    scale = float(j["r"].abs().mean()) + 1e-12
    maxdiff = float((j["g"] - j["r"]).abs().max())
    corr = float(j["g"].corr(j["r"]))
    out.update(steady_max_abs_diff=round(maxdiff, 10),
               steady_rel_diff=round(maxdiff / scale, 8),
               steady_corr=round(corr, 6), n_compared=len(j))
    if maxdiff < 1e-8 * (1 + scale):
        out["expansion_status"] = "exact"
    elif corr > 0.995:
        out["expansion_status"] = "approximate"
    else:
        out["expansion_status"] = "mismatch"
    return out, ir


# ---------------------------------------------------------------- 검색 (이름 인덱스)
def load_index(path=None):
    p = Path(path) if path else OUTDIR / "qant_recipes.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def find_recipes(nl, index=None, k=2):
    """자연어 요청에서 지표 이름/별칭 매치. (일반 사전 — 과제 전용 규칙 아님)"""
    idx = index or load_index()
    if not idx:
        return []
    low = nl.lower()
    hits = []
    for r in idx["recipes"]:
        if r["expansion_status"] == "mismatch":
            continue
        score = 0
        if r["name"].lower() in low:
            score += 3
        for al in r["aliases"]:
            if al.lower() in low:
                score += 2 if len(al) >= 3 else 1
        if score:
            hits.append((score, -len(r["expanded_dsl"]), r))
    hits.sort(key=lambda x: (-x[0], x[1]))
    return [r for _, _, r in hits[:k]]


# ---------------------------------------------------------------- 실행
def build_all():
    df = make_panel(n_dates=320)
    corpus = []
    for fn in ("qant_101_ir.json", "qant_158_ir.json"):
        p = OUTDIR / fn
        if p.exists():
            corpus += [a["ir"] for a in json.loads(p.read_text(encoding="utf-8"))["alphas"]]
    spec = derive_spec(corpus)
    rows = []
    for rec in RECIPES:
        res, _ = verify_recipe(rec, df, spec)
        res.update(aliases=rec["aliases"], default_params=rec["default_params"],
                   inputs=rec["inputs"], template=rec["template"],
                   notes=rec.get("notes", ""))
        rows.append(res)
    return rows


if __name__ == "__main__":
    rows = build_all()
    from collections import Counter
    st = Counter(r["expansion_status"] for r in rows)
    print(f"{'지표':16s} {'상태':12s} {'정상상태상관':>10s} {'상대오차':>12s}")
    for r in rows:
        print(f"{r['name']:16s} {r['expansion_status']:12s} "
              f"{r.get('steady_corr', float('nan')):10.4f} "
              f"{r.get('steady_rel_diff', float('nan')):12.2e}"
              + (f"  ! {r.get('error','')}" if r["expansion_status"] == "mismatch" else ""))
    print(f"\n상태 분포: {dict(st)}  /  extension_queue: {len(EXTENSION_QUEUE)}종")
    if "--list" in sys.argv:
        sys.exit(0)
    out = {"meta": {"version": RECIPES_VERSION,
                    "normalizer_version": NORMALIZER_VERSION,
                    "n_recipes": len(rows),
                    "status_counts": dict(st),
                    "verification": "합성 패널에서 독립 pandas 기준 구현과 정상상태 비교"},
           "recipes": rows,
           "extension_queue": EXTENSION_QUEUE}
    (OUTDIR / "qant_recipes.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUTDIR / 'qant_recipes.json'}")
    sys.exit(1 if st.get("mismatch") else 0)
