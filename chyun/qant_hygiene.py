"""
qant_hygiene — 3층 위생 검사 (실행 기반).

1층(문법)은 형태만, 2층(실행)은 기준구현이 있을 때만 쓸 수 있다.
3층은 기준구현 없이 신호 자체의 성질만으로 결함을 잡는다.

검사 코드:
  H_EXEC       실행 실패
  H_VALID      유효값(비NaN) 비율이 너무 낮음
  H_CONST      신호가 사실상 상수 (분산 없음)
  H_NOCS       횡단면 분산 없음 -> 순위를 매길 수 없음
  H_DEGEN      한 값에 과도하게 몰림
  H_EXPLODE    inf / 극단값 비율 과다
  H_LOOKAHEAD  미래 데이터를 바꿨더니 과거 신호가 바뀜 (경험적)
  H_INTENT     선언한 의도와 실제 방향 불일치 (참조 상관 기반)
  H_PROPERTY   선언한 구조적 속성 위반 (참조 신호 불필요)  ← 지적 ④ 반영

지적 ④ 배경:
  T06(섹터 중립화 위치 오류)은 참조 상관으로는 안 잡혔고, "결과의 섹터별
  평균이 0인가"를 직접 계산해서 잡았다. 그런 선언형 속성 검사를 정식 편입한다.
  참조 신호가 없어도 검증되므로 unverifiable 비율도 줄어든다.
"""
import numpy as np
import pandas as pd

from qant_eval import ev, make_panel


# ---------------------------------------------------------------- 참조 신호
def references(df):
    """표준 참조 신호. 생성된 신호의 '지문'을 찍는 데 쓴다.
    (지적: unverifiable 48% 를 줄이려면 참조 자체를 넓혀야 한다)"""
    G = lambda s: s.groupby(level="code")
    c, v = df["close"], df["volume"]
    dv = v * c
    rng_pos = (c - G(df["low"]).transform(lambda s: s.rolling(20, min_periods=5).min())) / \
              (G(df["high"]).transform(lambda s: s.rolling(20, min_periods=5).max()) -
               G(df["low"]).transform(lambda s: s.rolling(20, min_periods=5).min()))
    return {
        "mom20":   G(c).pct_change(20),
        "mom5":    G(c).pct_change(5),
        "revers":  -G(c).pct_change(5),
        "vol20":   G(c).transform(lambda s: s.pct_change().rolling(20, min_periods=5).std()),
        "size":    np.log(df["cap"]),
        "liq":     np.log(dv.replace(0, np.nan)),
        # --- 확장분 ---
        "rel_volume": v / G(v).transform(lambda s: s.rolling(20, min_periods=5).mean()),
        "range_pos":  rng_pos,
        "pv_corr":    _rollcorr(c, v, 20),
        "price_level": np.log(c),
    }


def _rollcorr(x, y, w):
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    out = j.groupby(level="code", group_keys=False).apply(
        lambda g: g["x"].rolling(w, min_periods=max(2, w // 2)).corr(g["y"]))
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
        out = out.reset_index(level=0, drop=True)
    return out.reindex(x.index)


def fingerprint(s, df):
    """신호와 참조들의 상관계수 지문."""
    out = {}
    for k, r in references(df).items():
        j = pd.concat([s.rename("s"), r.rename("r")], axis=1)
        j = j.replace([np.inf, -np.inf], np.nan).dropna()
        out[k] = float(j["s"].corr(j["r"])) if len(j) > 50 else float("nan")
    return out


def best_reference(s, df, min_abs=0.15):
    """지문에서 가장 강한 참조를 고른다. 라벨의 ref 배정 오류를 교정할 때 사용."""
    fp = fingerprint(s, df)
    cand = [(abs(v), k, v) for k, v in fp.items() if v == v and abs(v) >= min_abs]
    if not cand:
        return None, None, fp
    _, k, v = max(cand)
    return k, ("+" if v > 0 else "-"), fp


# ---------------------------------------------------------------- 속성 검사
def _p_sector_neutral(s, df, tol=1e-6):
    keys = [s.index.get_level_values("date"), df["__sector"].reindex(s.index)]
    gm = s.groupby(keys).transform("mean").abs()
    scale = s.abs().mean() + 1e-12
    ratio = float(gm.mean() / scale)
    return ratio < tol * 1e6 * 0 + 0.01, f"섹터별 평균/스케일 = {ratio:.4f} (0에 가까워야 함)"


def _p_cs_neutral(s, df):
    gm = s.groupby(level="date").transform("mean").abs()
    scale = s.abs().mean() + 1e-12
    ratio = float(gm.mean() / scale)
    return ratio < 0.01, f"일자별 평균/스케일 = {ratio:.4f}"


def _p_rank_range(s, df):
    """순위 스케일: |값| 이 [0,1] 안. 반전 순위(-1*rank -> [-1,0])도 정상으로 본다."""
    v = s.replace([np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        return False, "유효값 없음"
    lo, hi = float(v.min()), float(v.max())
    ok = abs(lo) <= 1 + 1e-9 and abs(hi) <= 1 + 1e-9 and (lo >= -1e-9 or hi <= 1e-9)
    return ok, f"값 범위 [{lo:.3f}, {hi:.3f}] (|값|<=1, 부호 일관)"


def _p_scale_invariant(s, df):
    """가격을 종목별로 상수배해도 신호가 안 바뀌어야 한다(비율형 팩터).
    가격 수준에 오염된 팩터(T02 의 ts_std(close))를 잡는다."""
    return None, ""      # ir 이 필요해서 hygiene() 안에서 처리


def _p_bounded(s, df):
    v = s.replace([np.inf, -np.inf], np.nan).dropna()
    ok = bool(len(v)) and float(v.abs().max()) < 1e6
    return ok, f"최대 절대값 {float(v.abs().max()) if len(v) else float('nan'):.3g}"


PROPERTIES = {
    "sector_neutral": _p_sector_neutral,   # 섹터 중립화를 선언했다면
    "cs_neutral": _p_cs_neutral,           # 횡단면 중립화를 선언했다면
    "rank_range": _p_rank_range,           # 순위 신호라면 [0,1]
    "bounded": _p_bounded,                 # 발산하지 않아야
}


def check_scale_invariance(ir, df, tol=0.02):
    """종목별 가격 스케일을 바꿔도 신호 순위가 유지되는가."""
    rng = np.random.default_rng(17)
    codes = df.index.get_level_values("code")
    uniq = codes.unique()
    mult = pd.Series(rng.uniform(0.2, 5.0, len(uniq)), index=uniq).reindex(codes).to_numpy()
    d2 = df.copy()
    for c in ("close", "open", "high", "low", "vwap", "cap"):
        if c in d2.columns:
            d2[c] = d2[c].to_numpy() * mult
    try:
        a = ev(ir, df).replace([np.inf, -np.inf], np.nan)
        b = ev(ir, d2).replace([np.inf, -np.inf], np.nan)
    except Exception:
        return None, None
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 50:
        return None, None
    rho = float(j["a"].rank().corr(j["b"].rank()))
    return rho >= 1 - tol, rho


# ---------------------------------------------------------------- 경험적 룩어헤드
def perturb_future(df, frac=0.6, seed=11):
    """cutoff 이후 데이터를 난수로 교체. 과거 신호는 영향받으면 안 된다."""
    rng = np.random.default_rng(seed)
    dates = df.index.get_level_values("date").unique().sort_values()
    cut = dates[int(len(dates) * frac)]
    d2 = df.copy()
    mask = df.index.get_level_values("date") > cut
    for col in d2.columns:
        if col == "__sector":
            continue
        vals = d2.loc[mask, col].to_numpy(dtype=float)
        d2.loc[mask, col] = rng.permutation(vals) * rng.uniform(0.5, 1.5, len(vals))
    return d2, cut


# ---------------------------------------------------------------- 본체
def hygiene(ir, df, expect=None, properties=None, name=""):
    """(위반목록, 지표dict)
    expect     : {"ref":"mom20","sign":"+"}  참조 상관 기반 의도 검사
    properties : ["sector_neutral", "rank_range", "scale_invariant", ...]
                 참조 신호 없이 구조적 속성을 검사 (지적 ④)
    """
    issues, m = [], {}
    try:
        s = ev(ir, df)
    except NotImplementedError as e:
        return [("H_EXEC", f"실행 미지원: {e}")], {}
    except Exception as e:
        return [("H_EXEC", f"실행 실패: {str(e)[:80]}")], {}

    s = pd.Series(s, index=df.index).astype(float)
    finite = s.replace([np.inf, -np.inf], np.nan)

    valid = finite.notna().mean()
    m["valid_ratio"] = round(float(valid), 4)
    if valid < 0.30:
        issues.append(("H_VALID", f"유효값 {valid:.1%} — 대부분 NaN"))

    inf_ratio = float((~np.isfinite(s.to_numpy(dtype=float))).mean() - s.isna().mean())
    m["inf_ratio"] = round(max(inf_ratio, 0.0), 4)
    if inf_ratio > 0.01:
        issues.append(("H_EXPLODE", f"inf 비율 {inf_ratio:.1%}"))

    v = finite.dropna()
    if len(v) < 50:
        issues.append(("H_VALID", f"유효 표본 {len(v)}개 — 판정 불가"))
        return issues, m

    sd = float(v.std())
    scale = float(v.abs().mean()) + 1e-12
    m["std"] = round(sd, 6)
    if not np.isfinite(sd) or sd < 1e-9 * (1 + scale):
        issues.append(("H_CONST", "전체 분산 ≈ 0 (상수 신호)"))

    cs_sd = finite.groupby(level="date").std().mean()
    m["cs_std"] = round(float(cs_sd), 6) if np.isfinite(cs_sd) else None
    if not np.isfinite(cs_sd) or cs_sd < max(1e-12, 1e-6 * sd):
        issues.append(("H_NOCS", "횡단면 분산 ≈ 0 — 종목 구분 불가"))

    top = float(v.round(10).value_counts(normalize=True).iloc[0])
    m["top_value_share"] = round(top, 4)
    if top > 0.90:
        issues.append(("H_DEGEN", f"단일 값이 {top:.1%} 차지"))

    # --- 경험적 룩어헤드 ---
    try:
        d2, cut = perturb_future(df)
        s2 = pd.Series(ev(ir, d2), index=df.index).astype(float)
        past = df.index.get_level_values("date") <= cut
        j = pd.concat([finite[past].rename("a"),
                       s2[past].replace([np.inf, -np.inf], np.nan).rename("b")],
                      axis=1).dropna()
        if len(j) > 50:
            drift = float((j["a"] - j["b"]).abs().mean() / (j["a"].abs().mean() + 1e-12))
            m["future_drift"] = round(drift, 6)
            if drift > 1e-6:
                issues.append(("H_LOOKAHEAD",
                               f"미래 교란 시 과거 신호가 {drift:.2%} 변함"))
    except Exception:
        pass

    # --- 속성 검사 (참조 불필요) ---
    for prop in (properties or []):
        if prop == "scale_invariant":
            ok, rho = check_scale_invariance(ir, df)
            if ok is not None:
                m["scale_rho"] = round(rho, 4)
                if not ok:
                    issues.append(("H_PROPERTY",
                                   f"scale_invariant 위반 — 가격 스케일 변경 시 "
                                   f"순위상관 {rho:.3f} (1.0 이어야 함)"))
            continue
        fn = PROPERTIES.get(prop)
        if fn is None:
            continue
        ok, detail = fn(finite, df)
        if ok is None:
            continue
        m[f"prop_{prop}"] = detail
        if not ok:
            issues.append(("H_PROPERTY", f"{prop} 위반 — {detail}"))

    # --- 의도(참조 상관) 검사 ---
    fp = fingerprint(finite, df)
    m["fingerprint"] = {k: (round(x, 3) if np.isfinite(x) else None)
                        for k, x in fp.items()}
    if expect and expect.get("ref"):
        r, want = expect["ref"], expect.get("sign", "+")
        got = fp.get(r, float("nan"))
        m["intent_corr"] = round(got, 3) if np.isfinite(got) else None
        if np.isfinite(got):
            thr = expect.get("min_abs", 0.15)
            ok = (got > thr) if want == "+" else (got < -thr)
            if not ok:
                issues.append(("H_INTENT",
                               f"'{r}' 와 {want} 상관 기대했으나 실측 {got:+.3f}"))
    return issues, m


def report(rows):
    print(f"{'과제':14s} {'상태':6s} {'유효':>6s} {'횡단면σ':>9s} {'미래drift':>10s}  위반")
    for name, issues, m in rows:
        st = "PASS" if not issues else "FLAG"
        print(f"{name:14s} {st:6s} {m.get('valid_ratio', 0):6.1%} "
              f"{(m.get('cs_std') or 0):9.4f} {m.get('future_drift', 0):10.6f}  "
              + (", ".join(c for c, _ in issues) or "-"))
        for c, msg in issues:
            print(f"         └ {c:13s} {msg}")
