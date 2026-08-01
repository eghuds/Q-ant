"""
qant_regression — 전체 보증 항목의 회귀 테스트.

실행:  python qant_regression.py
전제:  qant_out/qant_101_ir.json, qant_out/qant_158_ir.json
       (없으면 qant_101_alphas.py / qant_158_alphas.py 먼저 실행)

여기 항목이 하나라도 깨지면 이후 실험(라벨링·벤치마크)을 진행하면 안 된다.
"""
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

from qant_ir import normalize, sexpr, shash, NORMALIZER_VERSION
from qant_registry import REGISTRY, REGISTRY_VERSION
from qant_eval import make_panel, ev, EVAL_OPS
from qant_hygiene import hygiene
from qant_dsl import parse as dsl_parse, to_dsl, roundtrip_ok, axis_of
from qant_validate import derive_spec, validate

HERE = Path(__file__).resolve().parent
OUT = HERE / "qant_out"
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


F = lambda n: {"kind": "field", "name": n}
C = lambda v: {"kind": "const", "value": v}


def op(o, args, axis=None, **p):
    return {"kind": "op", "op": o, "axis": axis, "args": args, "params": p}


print("== 1. 정규화기 의미 보존 ==")
eps_ir = op("div", [F("close"), op("add", [op("sub", [F("high"), F("low")]), C(1e-12)])])
check("안정화 상수(1e-12) 보존", "1e-12" in sexpr(normalize(eps_ir)))
obf = op("add", [op("mul", [F("vwap"), C(0.728317)]),
                 op("mul", [F("vwap"), C(1 - 0.728317)])])
check("난독화 계수 접기", sexpr(normalize(obf)) == "$vwap")
check("리프 args 미주입", "args" not in normalize(F("close")))
one = normalize(eps_ir)
check("정규화 멱등성", shash(one) == shash(normalize(one)))
check("정수값 float 통합 (2.0 == 2)",
      shash(normalize(C(2.0))) == shash(normalize(C(2))))

print("== 2. 레지스트리 ↔ 실행기 커버리지 (지적 ③) ==")
missing = sorted(set(REGISTRY) - EVAL_OPS)
extra = sorted(EVAL_OPS - set(REGISTRY))
check("REGISTRY ⊆ EVAL_OPS", not missing, f"미구현 {missing}" if missing else f"{len(REGISTRY)}종 전부")
check("EVAL_OPS ⊆ REGISTRY", not extra, f"레지스트리 밖 {extra}" if extra else "")
check("축 유도 일관성",
      all(axis_of(o) in (None, "TS", "CS", "GRP") for o in REGISTRY))
for o, want in (("ts_mean", "TS"), ("cs_rank", "CS"), ("grp_demean", "GRP"), ("add", None)):
    check(f"  axis_of({o}) == {want}", axis_of(o) == want)

print("== 3. 코퍼스 무결성 ==")
corpora = {}
for fn, n_expect in (("qant_101_ir.json", 101), ("qant_158_ir.json", 158)):
    p = OUT / fn
    if not p.exists():
        check(f"{fn} 존재", False, "재생성 필요")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    corpora[fn] = d
    rows = d["alphas"]
    check(f"{fn} 개수", len(rows) == n_expect, f"{len(rows)}/{n_expect}")
    check(f"{fn} normalizer_version",
          d["meta"].get("normalizer_version") == NORMALIZER_VERSION,
          str(d["meta"].get("normalizer_version")))
    check(f"{fn} raw/canonical 키",
          all(all(k in r for k in ("ir_raw", "hash_raw", "ir_canonical",
                                   "hash_canonical")) for r in rows))
    check(f"{fn} canonical 유니크", len({r["hash_canonical"] for r in rows}) == n_expect)
    check(f"{fn} 레거시 ir == canonical",
          all(r["hash"] == r["hash_canonical"] for r in rows))

print("== 4. DSL 표면 불변식 (지적 ①②) ==")
all_rows = [r for d in corpora.values() for r in d["alphas"]]
rt = [roundtrip_ok(r["ir_canonical"])[0] for r in all_rows]
check("IR→DSL→IR 라운드트립 전건", all(rt), f"{sum(rt)}/{len(rt)}")
if all_rows:
    jl = sum(len(json.dumps(r["ir_canonical"], separators=(",", ":"))) for r in all_rows)
    dl = sum(len(to_dsl(r["ir_canonical"])) for r in all_rows)
    check("DSL 이 JSON 대비 4배 이상 짧음", jl / max(dl, 1) >= 4.0, f"{jl/max(dl,1):.1f}배")
check("DSL 은 축을 유도한다", dsl_parse("ts_mean(close, 20)")["axis"] == "TS")


def _rejects(expr):
    try:
        dsl_parse(expr)
        return False
    except Exception:
        return True


check("DSL 이 미지의 함수를 거부", _rejects("nosuch_op(close, 5)"))
check("DSL 이 미지의 필드를 거부", _rejects("ts_mean(nosuchfield, 5)"))

print("== 5. 실행기 전 코퍼스 커버리지 ==")
df = make_panel()
ok_run = fail_run = 0
for r in all_rows:
    try:
        ev(r["ir_canonical"], df)
        ok_run += 1
    except Exception:
        fail_run += 1
check("코퍼스 전건 실행", fail_run == 0, f"성공 {ok_run} / 실패 {fail_run}")

print("== 6. 합성 패널 이질성 ==")
pm = df["close"].groupby(level="code").mean().dropna()
check("가격수준 스펙트럼 > 50배", pm.max() / pm.min() > 50, f"{pm.max()/pm.min():.0f}배")
G = lambda s: s.groupby(level="code")
D = lambda s: s.groupby(level="date")
c1 = pd.concat([D(G(df["close"]).diff(20)).rank(pct=True).rename("d"),
                D(G(df["close"]).pct_change(20)).rank(pct=True).rename("r")],
               axis=1).dropna()
corr = c1["d"].corr(c1["r"])
check("가격차 vs 수익률 구별 가능(<0.95)", corr < 0.95, f"상관 {corr:+.3f}")
check("결측(후기상장) 존재", 0 < df["close"].isna().mean() < 0.1,
      f"결측 {df['close'].isna().mean():.1%}")

print("== 7. 위생 검사 동작 (지적 ④) ==")
iss, _ = hygiene(op("sub", [F("close"), F("close")]), df)
check("상수 신호 탐지", any(c == "H_CONST" for c, _ in iss))
iss, m = hygiene(dsl_parse("cs_rank(ts_returns(close, 20))"), df,
                 {"ref": "mom20", "sign": "+"}, properties=["rank_range", "scale_invariant"])
check("정상 모멘텀 통과", not iss, f"의도상관 {m.get('intent_corr')}")
iss, _ = hygiene(op("cs_rank", [op("ts_delay", [F("close")], "TS", window=-5)], "CS"), df)
check("경험적 룩어헤드 탐지", any(c == "H_LOOKAHEAD" for c, _ in iss))
# 속성: 섹터 중립 위반 (T06 재현)
iss, _ = hygiene(dsl_parse("close - grp_demean(vwap, sector)"), df,
                 properties=["sector_neutral"])
check("sector_neutral 위반 탐지", any(c == "H_PROPERTY" for c, _ in iss))
# 속성: 가격 스케일 오염 (T02 재현)
iss, _ = hygiene(dsl_parse("cs_rank(ts_std(close, 60))"), df,
                 properties=["scale_invariant"])
check("scale_invariant 위반 탐지", any(c == "H_PROPERTY" for c, _ in iss))
# 정답 구현은 통과해야 (오탐 0)
for name, expr, props in (
        ("섹터중립 괴리", "grp_demean((close - vwap) / vwap, sector)", ["sector_neutral"]),
        ("저변동성 순위", "-1 * cs_rank(ts_std(ts_returns(close, 1), 60))",
         ["rank_range", "scale_invariant"]),
        ("거래량 비율", "volume / ts_mean(volume, 10)", ["scale_invariant"])):
    iss, _ = hygiene(dsl_parse(expr), df, properties=props)
    check(f"  정답 통과: {name}", not iss, str([c for c, _ in iss]))

print("== 8. ts_returns canonical 재작성 (0.4.0) ==")
from qant_ir import fold_returns

def _same(e1, e2):
    return shash(normalize(dsl_parse(e1))) == shash(normalize(dsl_parse(e2)))

def _kept(e):
    ir0 = dsl_parse(e)
    return shash(normalize(ir0, fold_returns_field=False)) == \
           shash(fold_returns(normalize(ir0, fold_returns_field=False),
                              fold_returns_field=False))

# 양성: 수익률의 산술 변형들이 전부 같은 canonical 로 접힘
check("R1  x/delay - 1", _same("close / ts_delay(close, 20) - 1",
                               "ts_returns(close, 20)"))
check("R2  (x-delay)/delay", _same("(close - ts_delay(close, 20)) / ts_delay(close, 20)",
                                   "ts_returns(close, 20)"))
check("R2b delta/delay (JPX형)", _same("ts_delta(close, 1) / ts_delay(close, 1)",
                                       "ts_returns(close, 1)"))
check("R3  x/delay + (-1)", _same("close / ts_delay(close, 10) + (-1)",
                                  "ts_returns(close, 10)"))
check("중첩 접힘", _same("cs_rank(close / ts_delay(close, 20) - 1)",
                        "cs_rank(ts_returns(close, 20))"))
check("x=부분식", _same("(close * volume) / ts_delay(close * volume, 5) - 1",
                       "ts_returns(close * volume, 5)"))
# 음성: 접으면 안 되는 것들
check("x 불일치 보존", _kept("close / ts_delay(open, 20) - 1"))
check("상수 2 보존", _kept("close / ts_delay(close, 20) - 2"))
check("ratio형(1+r) 보존", _kept("close / ts_delay(close, 20)"))
check("역수형(roc) 보존", _kept("ts_delay(close, 5) / close"))
check("R2 분모 window 불일치 보존",
      _kept("(close - ts_delay(close, 5)) / ts_delay(close, 10)"))
# 멱등성
_r1 = normalize(dsl_parse("close / ts_delay(close, 20) - 1"))
check("재작성 멱등성", shash(normalize(_r1)) == shash(_r1))
# R4: returns 필드 -> ts_returns(close, 1) (canonical 층 전용)
check("R4 returns 필드 치환",
      shash(normalize(F("returns"))) == shash(normalize(dsl_parse("ts_returns(close, 1)"))))
check("R4 off 시 보존",
      normalize(F("returns"), fold_returns_field=False).get("kind") == "field")
# 코퍼스 노출: 재작성 후 두 코퍼스 모두 ts_returns 가 존재해야 함
_n_tr = {fn: sum(1 for r in d["alphas"] for o, _ in r["ops"] if o == "ts_returns")
         for fn, d in corpora.items()}
check("코퍼스 ts_returns 노출 (101)", _n_tr.get("qant_101_ir.json", 0) >= 18,
      f"{_n_tr.get('qant_101_ir.json', 0)}회")
check("코퍼스 ts_returns 노출 (158)", _n_tr.get("qant_158_ir.json", 0) >= 10,
      f"{_n_tr.get('qant_158_ir.json', 0)}회")

print("== 9. ts_ewm 파라미터 명시화 + ts_sma_cn (0.4.1 / registry 0.2.0) ==")
_e1 = dsl_parse("ts_ewm(close, 20)")
check("ts_ewm 위치 인자 -> span", _e1["params"] == {"span": 20})
check("ts_ewm 라운드트립", roundtrip_ok(_e1)[0])
_e2 = dsl_parse("ts_sma_cn(close, 14, 1)")
check("ts_sma_cn 파라미터 (n, m)", _e2["params"] == {"window": 14, "m": 1})
check("ts_sma_cn 라운드트립", roundtrip_ok(_e2)[0])
_sp2 = derive_spec([r["ir"] for r in all_rows]) if all_rows else None
if _sp2:
    _err1, _, _ = validate(_e1, _sp2)
    _err2, _, _ = validate(_e2, _sp2)
    check("ts_ewm/ts_sma_cn 검증 에러 0", not _err1 and not _err2)
    _bad, _, _ = validate(dsl_parse("ts_sma_cn(close, 14, 20)"), _sp2)
    check("ts_sma_cn m>n 거부", any(c == "E_WINDOW" for c, _, _ in _bad))
_legacy = op("ts_ewm", [F("close")], "TS", window=20)
_mig = normalize(_legacy)
check("구표기 window -> span 이관", _mig["params"] == {"span": 20})
check("이관 멱등성", shash(normalize(_mig)) == shash(_mig))
# 수치: ts_sma_cn(x, n, 1) == Wilder 재귀 (y = (1-1/n)y + x/n)
_got = ev(_e2, df)
def _wilder(s, n):
    v = s.to_numpy(float); out = np.full(len(v), np.nan); prev = None
    for i, x in enumerate(v):
        if not np.isfinite(x):
            prev = None; continue
        prev = x if prev is None else (1 - 1.0 / n) * prev + x / n
        out[i] = prev
    return pd.Series(out, index=s.index)
_ref = df["close"].groupby(level="code", group_keys=False).apply(lambda s: _wilder(s, 14))
_j = pd.concat([_got.rename("g"), _ref.rename("r")], axis=1).dropna()
check("ts_sma_cn == Wilder 재귀", float((_j.g - _j.r).abs().max()) < 1e-9,
      f"maxdiff {float((_j.g - _j.r).abs().max()):.2e}")

print("== 10. 이름 인덱스 (레시피 사전 v0) ==")
_rp = OUT / "qant_recipes.json"
check("qant_recipes.json 존재", _rp.exists(), "없으면 python qant_recipes.py 실행")
if _rp.exists():
    _R = json.loads(_rp.read_text(encoding="utf-8"))
    _recs = _R["recipes"]
    check("레시피 25개 이상", len(_recs) >= 25, f"{len(_recs)}개")
    check("mismatch 0", all(r["expansion_status"] != "mismatch" for r in _recs),
          str([r["name"] for r in _recs if r["expansion_status"] == "mismatch"]))
    check("정규화 버전 일치", _R["meta"].get("normalizer_version") == NORMALIZER_VERSION,
          str(_R["meta"].get("normalizer_version")))
    # 전개식 전건: 파싱 + 검증 0 에러 (수치 비교는 qant_recipes.py 자체 관문)
    _n_ok = 0
    for r in _recs:
        try:
            _ir = normalize(dsl_parse(r["expanded_dsl"]))
            _er, _, _ = validate(_ir, _sp2)
            _n_ok += (not _er)
        except Exception:
            pass
    check("전개식 전건 파싱+검증", _n_ok == len(_recs), f"{_n_ok}/{len(_recs)}")
    from qant_recipes import find_recipes
    _idx = _R
    check("검색: 'RSI가 30 이하' -> RSI",
          any(r["name"] == "RSI" for r in find_recipes("14일 RSI가 30 이하인 종목", _idx)))
    check("검색: '볼린저밴드' 히트",
          any("BB" in r["name"] for r in find_recipes("볼린저밴드 하단 터치", _idx)))
    check("검색: 'EMA' -> EMA",
          any(r["name"] == "EMA" for r in find_recipes("종가의 20일 지수이동평균(EMA)", _idx)))
    check("검색: 무관 문장 무히트", not find_recipes("섹터 중립화한 vwap 대비 종가 괴리", _idx))

n_fail = sum(1 for _, okk, _ in RESULTS if not okk)
print(f"\n{'='*50}\n총 {len(RESULTS)}개 중 실패 {n_fail}개  "
      f"(normalizer {NORMALIZER_VERSION} / registry {REGISTRY_VERSION})")
sys.exit(1 if n_fail else 0)
