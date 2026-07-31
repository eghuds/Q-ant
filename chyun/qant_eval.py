"""
qant_eval — IR 실행기 (의미 검증용).

문법 검증기는 '형태'만 본다. 이 파일은 '뜻'을 본다.
IR 을 패널 데이터에 실제로 실행해서 기준 구현/속성과 비교한다.

패널 형식: MultiIndex (date, code)
  TS 축 = code 별 시계열,  CS 축 = date 별 횡단면,  GRP 축 = (date, sector)

설계 원칙 (지적 ③ 반영):
  실행기는 디스패치 테이블(EVAL_TABLE)로 구성한다. 레지스트리의 모든 연산자가
  여기 있는지 프로그램으로 검사할 수 있어야, "레지스트리엔 있는데 실행 못 함"
  때문에 라벨이 조용히 unverifiable 로 빠지는 사고가 안 난다.
  qant_regression.py 가 REGISTRY ⊆ EVAL_OPS 를 강제한다.
"""
import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 합성 패널
def make_panel(n_dates=160, n_codes=28, seed=3):
    """합성 패널 v2: 종목 간 이질성 필수.
    가격수준(로그정규) / 변동성 / 거래량수준 / 시총(가격연동) / 섹터 / 결측.
    이질성이 없으면 '가격차 vs 수익률' 혼동을 테스트가 못 잡는다."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    codes = [f"S{i:03d}" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    n = len(idx)

    base_px = rng.lognormal(np.log(30), 1.2, n_codes)
    vol_i = rng.uniform(0.008, 0.04, n_codes)
    rets = rng.normal(0, 1, (n_dates, n_codes)) * vol_i
    close = pd.Series((np.exp(np.cumsum(rets, axis=0)) * base_px).ravel(), index=idx)

    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"] * (1 + rng.normal(0, 0.005, n))
    df["high"] = df[["close", "open"]].max(axis=1) * (1 + abs(rng.normal(0, 0.004, n)))
    df["low"] = df[["close", "open"]].min(axis=1) * (1 - abs(rng.normal(0, 0.004, n)))
    vol_base = rng.lognormal(12, 1.0, n_codes)
    df["volume"] = (abs(rng.lognormal(0, 0.5, n).reshape(n_dates, n_codes))
                    * vol_base).ravel()
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    df["returns"] = df["close"].groupby(level="code").pct_change()
    shares_i = rng.lognormal(16, 0.8, n_codes)
    df["cap"] = (df["close"].to_numpy().reshape(n_dates, n_codes) * shares_i).ravel()

    # 후기 상장 2종목 (결측)
    late = df.index.get_level_values("code").isin(codes[-2:]) & \
        (df.index.get_level_values("date") < dates[30])
    for c in ("close", "open", "high", "low", "volume", "vwap", "returns"):
        df.loc[late, c] = np.nan

    sec = {c: f"SEC{i % 4}" for i, c in enumerate(codes)}
    df["__sector"] = [sec[c] for _, c in df.index]
    return df


def ensure_adv(df, w):
    """adv{N} 은 요청 시 생성 (레지스트리 필드 패턴)."""
    col = f"adv{int(w)}"
    if col not in df.columns:
        dv = df["volume"] * df["close"]
        df[col] = dv.groupby(level="code").transform(
            lambda s, w=int(w): s.rolling(w, min_periods=max(2, int(w) // 2)).mean())
    return df[col]


# ---------------------------------------------------------------- 헬퍼
def _code(s):
    return s.groupby(level="code")


def _date(s):
    return s.groupby(level="date")


def _grp_keys(s, df):
    return [s.index.get_level_values("date"), df["__sector"].reindex(s.index)]


def _roll(s, w, fn, **kw):
    w = max(1, int(w))
    mp = min(w, max(2, w // 2))          # window=1 이면 min_periods 도 1
    return _code(s).transform(
        lambda x: getattr(x.rolling(w, min_periods=mp), fn)(**kw))


def _roll_apply(s, w, fn):
    w = max(1, int(w))
    mp = min(w, max(2, w // 2))
    return _code(s).transform(
        lambda x: x.rolling(w, min_periods=mp).apply(fn, raw=True))


def _safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    return a / b


def _linfit(x, what):
    if np.any(~np.isfinite(x)) or len(x) < 2:
        return np.nan
    t = np.arange(len(x), dtype=float)
    b, c0 = np.polyfit(t, x, 1)
    if what == "slope":
        return b
    fitted = b * t + c0
    if what == "resi":
        return x[-1] - fitted[-1]
    ssr = np.sum((x - fitted) ** 2)
    sst = np.sum((x - x.mean()) ** 2)
    return 1 - ssr / sst if sst > 0 else np.nan


def _decay(x):
    w = np.arange(1, len(x) + 1, dtype=float)
    return float(np.dot(x, w) / w.sum())


def _ts2(x, y, w, fn):
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    w = max(2, int(w))
    mp = max(2, w // 2)
    out = j.groupby(level="code", group_keys=False).apply(
        lambda g: getattr(g["x"].rolling(w, min_periods=mp), fn)(g["y"]))
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
        out = out.reset_index(level=0, drop=True)
    return out.reindex(x.index)


# ---------------------------------------------------------------- 디스패치 테이블
# 각 항목: op -> fn(args:list[Series], params:dict, df) -> Series
EVAL_TABLE = {
    # --- 원소별 산술 ---
    "add": lambda a, p, d: a[0] + a[1],
    "sub": lambda a, p, d: a[0] - a[1],
    "mul": lambda a, p, d: a[0] * a[1],
    "div": lambda a, p, d: _safe_div(a[0], a[1]),
    "pow": lambda a, p, d: np.sign(a[0]) * (a[0].abs() ** a[1]),
    "neg": lambda a, p, d: -a[0],
    "abs": lambda a, p, d: a[0].abs(),
    "log": lambda a, p, d: np.log(a[0].where(a[0] > 0)),
    "log1p": lambda a, p, d: np.log1p(a[0].where(a[0] > -1)),
    "sqrt": lambda a, p, d: np.sqrt(a[0].where(a[0] >= 0)),
    "exp": lambda a, p, d: np.exp(a[0].clip(-50, 50)),
    "sign": lambda a, p, d: np.sign(a[0]),
    "signedpower": lambda a, p, d: np.sign(a[0]) * (a[0].abs() ** a[1]),
    "min": lambda a, p, d: pd.concat(a, axis=1).min(axis=1),
    "max": lambda a, p, d: pd.concat(a, axis=1).max(axis=1),
    "where": lambda a, p, d: a[1].where(a[0] > 0, a[2]),
    # --- 비교/논리 (bool -> float) ---
    "lt": lambda a, p, d: (a[0] < a[1]).astype(float),
    "gt": lambda a, p, d: (a[0] > a[1]).astype(float),
    "le": lambda a, p, d: (a[0] <= a[1]).astype(float),
    "ge": lambda a, p, d: (a[0] >= a[1]).astype(float),
    "eq": lambda a, p, d: (a[0] == a[1]).astype(float),
    "ne": lambda a, p, d: (a[0] != a[1]).astype(float),
    "or_": lambda a, p, d: ((a[0] > 0) | (a[1] > 0)).astype(float),
    "and_": lambda a, p, d: ((a[0] > 0) & (a[1] > 0)).astype(float),
    # --- 횡단면 ---
    "cs_rank": lambda a, p, d: _date(a[0]).rank(pct=True),
    "cs_scale": lambda a, p, d: _date(a[0]).transform(
        lambda s: s / (s.abs().sum() or np.nan)),
    "cs_mean": lambda a, p, d: _date(a[0]).transform("mean"),
    "cs_std": lambda a, p, d: _date(a[0]).transform("std"),
    "cs_median": lambda a, p, d: _date(a[0]).transform("median"),
    # --- 그룹(섹터) ---
    "grp_demean": lambda a, p, d: a[0] - a[0].groupby(_grp_keys(a[0], d)).transform("mean"),
    "grp_mean": lambda a, p, d: a[0].groupby(_grp_keys(a[0], d)).transform("mean"),
    "grp_rank": lambda a, p, d: a[0].groupby(_grp_keys(a[0], d)).rank(pct=True),
    # --- 시계열 단항 ---
    "ts_delay": lambda a, p, d: _code(a[0]).shift(int(p["window"])),
    "ts_delta": lambda a, p, d: a[0] - _code(a[0]).shift(int(p["window"])),
    "ts_returns": lambda a, p, d: _code(a[0]).pct_change(int(p["window"])),
    "ts_mean": lambda a, p, d: _roll(a[0], p["window"], "mean"),
    "ts_std": lambda a, p, d: _roll(a[0], p["window"], "std"),
    "ts_var": lambda a, p, d: _roll(a[0], p["window"], "var"),
    "ts_sum": lambda a, p, d: _roll(a[0], p["window"], "sum"),
    "ts_min": lambda a, p, d: _roll(a[0], p["window"], "min"),
    "ts_max": lambda a, p, d: _roll(a[0], p["window"], "max"),
    "ts_median": lambda a, p, d: _roll(a[0], p["window"], "median"),
    "ts_skew": lambda a, p, d: _roll(a[0], p["window"], "skew"),
    "ts_rank": lambda a, p, d: _roll(a[0], p["window"], "rank", pct=True),
    "ts_product": lambda a, p, d: _roll_apply(a[0], p["window"], np.prod),
    "ts_argmax": lambda a, p, d: _roll_apply(
        a[0], p["window"], lambda x: len(x) - 1 - int(np.argmax(x))),
    "ts_argmin": lambda a, p, d: _roll_apply(
        a[0], p["window"], lambda x: len(x) - 1 - int(np.argmin(x))),
    "ts_slope": lambda a, p, d: _roll_apply(a[0], p["window"],
                                            lambda x: _linfit(x, "slope")),
    "ts_rsquare": lambda a, p, d: _roll_apply(a[0], p["window"],
                                              lambda x: _linfit(x, "rsq")),
    "ts_resi": lambda a, p, d: _roll_apply(a[0], p["window"],
                                           lambda x: _linfit(x, "resi")),
    "ts_cumsum": lambda a, p, d: _code(a[0]).cumsum(),
    "ts_cumprod": lambda a, p, d: _code(a[0]).cumprod(),
    "ts_decay_linear": lambda a, p, d: _roll_apply(a[0], p["window"], _decay),
    "ts_ewm": lambda a, p, d: _code(a[0]).transform(
        lambda s: s.ewm(span=max(2, int(p.get("span", p.get("window")))),
                        min_periods=2).mean()),
    "ts_sma_cn": lambda a, p, d: _code(a[0]).transform(
        lambda s: s.ewm(alpha=float(p.get("m", 1)) / float(p["window"]),
                        adjust=False, min_periods=1).mean()),
    "ts_quantile": lambda a, p, d: _code(a[0]).transform(
        lambda s: s.rolling(max(2, int(p["window"])),
                            min_periods=max(2, int(p["window"]) // 2))
        .quantile(float(p.get("q", 0.5)))),
    # --- 시계열 이항 ---
    "ts_corr": lambda a, p, d: _ts2(a[0], a[1], p["window"], "corr"),
    "ts_cov": lambda a, p, d: _ts2(a[0], a[1], p["window"], "cov"),
}

EVAL_OPS = set(EVAL_TABLE)          # 회귀 테스트가 참조 (REGISTRY ⊆ EVAL_OPS)
ADV_PAT = re.compile(r"^adv(\d+)$")


# ---------------------------------------------------------------- 실행
def ev(ir, df):
    """IR -> pd.Series."""
    k = ir.get("kind")
    if k == "field":
        nm = ir["name"]
        m = ADV_PAT.match(str(nm))
        if m:
            return ensure_adv(df, int(m.group(1)))
        if nm not in df.columns:
            raise NotImplementedError(f"field '{nm}'")
        return df[nm]
    if k == "const":
        v = ir["value"]
        if not isinstance(v, (int, float)):
            raise NotImplementedError(f"const {v!r}")
        return pd.Series(float(v), index=df.index, dtype=float)
    if k == "param":
        raise NotImplementedError("param 노드는 실행 전 바인딩 필요")
    if k != "op":
        raise NotImplementedError(str(k))

    o = ir["op"]
    fn = EVAL_TABLE.get(o)
    if fn is None:
        raise NotImplementedError(o)
    a = [ev(x, df) for x in ir.get("args", [])]
    p = dict(ir.get("params") or {})
    if "window" not in p:                     # 위치 인자로 온 윈도우 흡수
        for v in (p.get("_args") or []):
            if isinstance(v, (int, float)):
                p["window"] = v
                break
    out = fn(a, p, df)
    return pd.Series(out, index=df.index, dtype=float)


def corr_check(ir, ref, df, name=""):
    """IR 실행 결과와 기준 구현의 상관계수."""
    got = ev(ir, df)
    j = pd.concat([got.rename("got"), ref.rename("ref")], axis=1).dropna()
    if len(j) < 30:
        return name, len(j), float("nan")
    return name, len(j), j["got"].corr(j["ref"])
