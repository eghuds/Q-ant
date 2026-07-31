"""
qant_validate — IR 문법 계약 추출 + 검증기.

계약(어휘·arity·축 규칙)을 사람이 상상해서 쓰지 않는다.
코퍼스(qant_101_ir.json 등)에서 관측된 사용법을 그대로 계약으로 삼는다.
따라서 "LLM이 우리 데이터의 문법을 지켰는가"를 순환논증 없이 측정할 수 있다.

검증 코드:
  E_JSON      JSON 파싱 실패
  E_SCHEMA    노드 구조 불량 (kind 누락/미지원, args 타입 오류)
  E_VOCAB     코퍼스에 없는 연산자
  E_ARITY     해당 연산자에서 관측된 적 없는 인자 개수
  E_AXIS      연산자 이름 규약과 축 불일치 (ts_* 는 TS 등)
  E_WINDOW    시계열 연산자에 window 누락 / 비정상 값
  E_FIELD     코퍼스에 없는 필드명
  E_LOOKAHEAD 음수 window, center=True 등 미래 참조
  E_DEPTH     과도한 깊이(사실상 무한 중첩)
"""
import json, re
from collections import defaultdict, Counter

MAX_DEPTH = 40
ADV_RE = re.compile(r"^adv\d+$")


# ---------------------------------------------------------------- 계약 추출
def derive_spec(irs):
    """코퍼스 IR 목록에서 문법 계약을 관측으로 도출."""
    ops = defaultdict(lambda: {"axis": Counter(), "arity": Counter(),
                               "window": 0, "count": 0})
    fields = Counter()

    def walk(ir, d=0):
        if not isinstance(ir, dict) or d > MAX_DEPTH:
            return
        k = ir.get("kind")
        if k == "field":
            fields[ir.get("name")] += 1
            return
        if k != "op":
            return
        rec = ops[ir["op"]]
        rec["count"] += 1
        rec["axis"][ir.get("axis")] += 1
        rec["arity"][len(ir.get("args", []))] += 1
        p = ir.get("params") or {}
        if "window" in p:
            rec["window"] += 1
            w = p["window"]
            if isinstance(w, (int, float)):
                rec.setdefault("wvals", []).append(w)
        for a in ir.get("args", []):
            walk(a, d + 1)

    for ir in irs:
        walk(ir)

    profile = {}
    for op, r in ops.items():
        axes = {a for a in r["axis"]}
        wv = r.get("wvals", [])
        profile[op] = {
            "axes": axes,
            "arity": set(r["arity"]),
            "count": r["count"],
            "wmin": (min(wv) if wv else None),
            "wmax": (max(wv) if wv else None),
        }
    from qant_registry import REGISTRY, BASE_FIELDS, REGISTRY_VERSION
    return {
        "registry": REGISTRY,                 # 구조적 진리 -> E_ 하드 에러
        "profile": profile,                   # 관측 통계   -> W_ 경고
        "fields": set(fields) | BASE_FIELDS,
        "registry_version": REGISTRY_VERSION,
    }


# ---------------------------------------------------------------- 검증
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------- 값 도메인 (3층 피드백 반영)
# 지난 실험의 실제 실패 3건(T07 단위불일치, T02 가격std, C09 부호결합)을
# 생성 시점에 잡기 위한 의미 태그. 위반은 '경고'다 — 코퍼스 자체(Alpha#7)에도
# 같은 패턴이 있으므로 하드 에러로 하면 계약이 자기모순이 된다.
FIELD_DOMAIN = {
    "open": "price", "high": "price", "low": "price", "close": "price",
    "vwap": "price", "volume": "shares", "cap": "currency", "returns": "ratio",
}

def domain_of(ir, d=0):
    if not isinstance(ir, dict) or d > MAX_DEPTH:
        return "unknown"
    k = ir.get("kind")
    if k == "field":
        nm = ir.get("name", "")
        if ADV_RE.match(nm):
            return "currency"
        return FIELD_DOMAIN.get(nm, "unknown")
    if k == "const":
        return "const"
    if k != "op":
        return "unknown"
    op = ir.get("op", "")
    a = ir.get("args", [])
    da = [domain_of(x, d + 1) for x in a]

    if op in ("cs_rank",):
        return "rank01"
    if op in ("lt", "gt", "le", "ge", "eq", "ne", "or_", "and_"):
        return "bool"
    if op == "div" and len(da) == 2:
        if da[0] == da[1] and da[0] in ("price", "shares", "currency"):
            return "ratio"
        return "signed"
    if op in ("ts_returns", "ts_rsquare"):
        return "ratio"
    if op in ("neg",):
        return "neg_" + da[0] if da and da[0] == "rank01" else "signed"
    if op == "mul" and len(da) == 2:
        # 음수 상수 × rank01 -> 반전된 랭크
        for i, j in ((0, 1), (1, 0)):
            if da[i] == "const" and _is_neg_const(a[i]) and da[j] == "rank01":
                return "neg_rank01"
        if "rank01" in da and "neg_rank01" in da:
            return "signed"
        return _join(da)
    if op in ("add", "sub"):
        return _join(da)
    if op in ("ts_mean", "ts_sum", "ts_min", "ts_max", "ts_delay",
              "ts_std", "ts_quantile", "ts_resi", "ts_decay_linear"):
        return da[0] if da else "unknown"
    if op in ("abs", "log", "sign", "signedpower", "pow", "where",
              "min", "max", "cs_scale", "grp_demean",
              "ts_rank", "ts_argmax", "ts_argmin", "ts_corr", "ts_cov",
              "ts_delta", "ts_slope", "ts_product"):
        return "signed" if op in ("ts_delta", "ts_slope") else "unknown"
    return "unknown"

def _is_neg_const(n):
    return isinstance(n, dict) and n.get("kind") == "const" and \
           isinstance(n.get("value"), (int, float)) and n["value"] < 0

def _join(da):
    da = [x for x in da if x != "const"]
    if not da:
        return "const"
    return da[0] if all(x == da[0] for x in da) else "mixed"

HARD_UNITS = {"price", "shares", "currency"}


def validate(ir, spec, path="root"):
    """IR 하나를 검증. (errors, warnings, n_nodes) 반환. E_* 는 차단, W_* 는 검토 권고."""
    errs, warns = [], []
    n = [0]

    def walk(node, path, d):
        if d > MAX_DEPTH:
            errs.append(("E_DEPTH", path, f"깊이 {MAX_DEPTH} 초과"))
            return
        if not isinstance(node, dict):
            errs.append(("E_SCHEMA", path, f"노드가 dict 아님: {type(node).__name__}"))
            return
        n[0] += 1
        k = node.get("kind")

        if k == "field":
            nm = node.get("name")
            if not isinstance(nm, str):
                errs.append(("E_SCHEMA", path, "field.name 누락"))
            elif nm not in spec["fields"] and not ADV_RE.match(nm):
                errs.append(("E_FIELD", path, f"미지의 필드 '{nm}'"))
            return

        if k == "const":
            if "value" not in node:
                errs.append(("E_SCHEMA", path, "const.value 누락"))
            return

        if k == "param":
            if not node.get("name"):
                errs.append(("E_SCHEMA", path, "param.name 누락"))
            return

        if k != "op":
            errs.append(("E_SCHEMA", path, f"미지원 kind '{k}'"))
            return

        op = node.get("op")
        if not isinstance(op, str):
            errs.append(("E_SCHEMA", path, "op 누락"))
            return

        args = node.get("args")
        if not isinstance(args, list):
            errs.append(("E_SCHEMA", f"{path}.{op}", "args 가 리스트 아님"))
            args = []

        params = node.get("params")
        if params is not None and not isinstance(params, dict):
            errs.append(("E_SCHEMA", f"{path}.{op}", "params 가 dict 아님"))
            params = {}
        params = params or {}

        here = f"{path}.{op}"
        reg = spec["registry"].get(op)
        prof = spec["profile"].get(op)
        if reg is None:
            # 레지스트리에 없음 = 정말로 존재하지 않는 연산자 -> 하드 에러
            errs.append(("E_VOCAB", here, f"레지스트리에 없는 연산자 '{op}'"))
        else:
            na = len(args)
            if not (reg["min_args"] <= na <= reg["max_args"]):
                errs.append(("E_ARITY", here,
                             f"인자 {na}개 (허용: {reg['min_args']}~{reg['max_args']})"))
            ax = node.get("axis")
            if ax not in reg["axes"]:
                errs.append(("E_AXIS", here,
                             f"axis={ax!r} (허용: {sorted(str(a) for a in reg['axes'])})"))
            if reg["window"] == "required" and "window" not in params:
                errs.append(("E_WINDOW", here, "window 필수인데 누락"))
            # ---- 관측 프로파일 기준 경고 (창의적 사용은 막지 않고 표시만) ----
            if prof is None:
                warns.append(("W_UNSEEN_OP", here, "유효하나 코퍼스에서 관측된 적 없음"))
            else:
                if na not in prof["arity"]:
                    warns.append(("W_UNSEEN_ARITY", here,
                                  f"인자 {na}개는 관측된 적 없음 (관측: {sorted(prof['arity'])})"))
                wv = params.get("window")
                if _num(wv) and prof.get("wmin") is not None and wv > 0 and wv < prof["wmin"]:
                    warns.append(("W_WINDOW_RANGE", here,
                                  f"window={wv} — 코퍼스 관측 최소 {prof['wmin']} 미만 (퇴화 가능)"))

        # 이름 규약과 축 일치 (어휘에 없는 연산자에도 적용)
        ax = node.get("axis")
        if op.startswith("ts_") and ax != "TS":
            errs.append(("E_AXIS", here, f"ts_* 인데 axis={ax!r}"))
        if op.startswith("cs_") and ax != "CS":
            errs.append(("E_AXIS", here, f"cs_* 인데 axis={ax!r}"))
        if op.startswith("grp_") and ax != "GRP":
            errs.append(("E_AXIS", here, f"grp_* 인데 axis={ax!r}"))

        # 값 도메인 검사 (경고)
        if op in ("lt", "gt", "le", "ge", "eq", "ne") and len(args) == 2:
            d0, d1 = domain_of(args[0]), domain_of(args[1])
            if d0 in HARD_UNITS and d1 in HARD_UNITS and d0 != d1:
                warns.append(("W_UNIT_CMP", here,
                              f"단위 불일치 비교: {d0} vs {d1} — 조건이 퇴화할 수 있음"))
        if op == "ts_std" and args and domain_of(args[0]) == "price":
            warns.append(("W_STD_PRICE", here,
                          "가격 수준의 std — 변동성 의도라면 수익률 기준이어야 함"))
        if op == "mul" and len(args) == 2:
            d0, d1 = domain_of(args[0]), domain_of(args[1])
            if {"rank01", "neg_rank01"} == {d0, d1}:
                warns.append(("W_RANK_MUL", here,
                              "rank × 반전rank 곱결합 — 순서가 뒤집힐 수 있음. 덧셈+1-rank 권장"))

        # 룩어헤드 / 윈도우 위생
        w = params.get("window")
        # ts_ewm: span 이 window 를 대신한다 (0.4.1 의미 명시화)
        if op == "ts_ewm":
            sp = params.get("span", w)
            if sp is None:
                errs.append(("E_WINDOW", here, "ts_ewm 은 span 필수 (구표기 window 허용)"))
            elif not _num(sp) or sp <= 0:
                errs.append(("E_WINDOW", here, f"span={sp!r} — 양수여야 함"))
        # ts_sma_cn: 0 < m <= n (alpha = m/n 이 (0,1] 이어야 재귀 평활)
        if op == "ts_sma_cn":
            mv = params.get("m")
            if mv is None:
                errs.append(("E_WINDOW", here, "ts_sma_cn 은 m 필수 (alpha=m/n)"))
            elif not _num(mv) or not _num(w) or not (0 < mv <= w):
                errs.append(("E_WINDOW", here,
                             f"m={mv!r}, n={w!r} — 0 < m <= n 이어야 함"))
        if w is not None:
            if not _num(w):
                errs.append(("E_WINDOW", here, f"window 가 숫자 아님: {w!r}"))
            elif w < 0:
                errs.append(("E_LOOKAHEAD", here, f"음수 window {w}"))
            elif w == 0:
                errs.append(("E_WINDOW", here, "window=0"))
        if params.get("center") is True:
            errs.append(("E_LOOKAHEAD", here, "center=True"))

        for i, a in enumerate(args):
            walk(a, f"{here}[{i}]", d + 1)

    walk(ir, path, 0)
    return errs, warns, n[0]


def validate_json_text(txt, spec):
    """LLM 출력 문자열 -> 검증. 코드펜스 제거 포함."""
    t = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except Exception as e:
        return [("E_JSON", "root", str(e)[:120])], [], 0, None
    errs, warns, n = validate(obj, spec)
    return errs, warns, n, obj


# ---------------------------------------------------------------- 계약 요약
def spec_summary(spec, top=None):
    """LLM 프롬프트에 넣을 문법 계약 텍스트 (레지스트리 기준, 관측 빈도순)."""
    lines = []
    prof = spec["profile"]
    items = sorted(spec["registry"].items(),
                   key=lambda kv: -(prof.get(kv[0], {}).get("count", 0)))
    if top:
        items = items[:top]
    for op, s in items:
        ax = sorted(str(a) for a in s["axes"])
        ax = "/".join("null" if a == "None" else a for a in ax)
        ar = f"{s['min_args']}" if s["min_args"] == s["max_args"] \
             else f"{s['min_args']}~{s['max_args']}"
        w = " +window" if s["window"] == "required" else ""
        lines.append(f"  {op:18s} axis={ax:6s} args={ar}{w}")
    flds = sorted(spec["fields"])
    return ("연산자 (axis / 인자개수 / window필수):\n" + "\n".join(lines) +
            "\n\n필드: " + ", ".join(flds) + ", adv{N}")


if __name__ == "__main__":
    import os
    _p = "qant_101_ir.json"
    if not os.path.exists(_p):
        _p = os.path.join("qant_out", "qant_101_ir.json")
    d = json.load(open(_p, encoding="utf-8"))
    irs = [a["ir"] for a in d["alphas"]]
    spec = derive_spec(irs)
    print(f"레지스트리 {len(spec['registry'])}종 / 관측 {len(spec['profile'])}종 / 필드 {len(spec['fields'])}종\n")
    print(spec_summary(spec))

    # 자기검증: 코퍼스 자신은 100% 통과해야 함 (계약이 여기서 나왔으므로)
    bad, warned = 0, 0
    wsamples = []
    for a in d["alphas"]:
        e, w, _ = validate(a["ir"], spec)
        if e:
            bad += 1
            print("SELF-FAIL", a["alpha_id"], e[:2])
        if w:
            warned += 1
            wsamples += [(a["alpha_id"], *x) for x in w]
    print(f"\n[자기검증] 에러 {bad}개 / 경고 있는 알파 {warned}개")
    from collections import Counter
    print("경고 분포:", dict(Counter(c for _, c, _, _ in wsamples)))
    for aid, c, _, m in wsamples[:5]:
        print(f"  {aid} {c}: {m[:60]}")
