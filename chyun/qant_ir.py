"""
qant_ir — 코퍼스 독립 IR 계층 (불변식).

모든 프론트엔드 파서(JPX pandas, 101 alphas DSL, ...)는 이 스키마로 수렴한다:

  op   노드: {"kind":"op", "op":str, "axis":"TS"|"CS"|"GRP"|None,
              "args":[<IR>...], "params":{...}}
  리프:      {"kind":"field","name":str} | {"kind":"const","value":...}
             | {"kind":"param","name":str,"domain":...}

이 파일에는 코퍼스 지식이 없다. 직렬화/해시/순회/분류만 있다.
"""
import json, hashlib

MAXD = 60
NORMALIZER_VERSION = "0.4.1"   # 정규화 규칙 변경 시 반드시 올릴 것


# ---------- 순회 ----------
def ops_in(ir, acc=None, depth=0):
    acc = acc if acc is not None else []
    if not isinstance(ir, dict) or depth > MAXD:
        return acc
    if ir.get("kind") == "op":
        acc.append((ir["op"], ir.get("axis")))
    for a in ir.get("args", []):
        ops_in(a, acc, depth + 1)
    return acc


def fields_in(ir, acc=None, depth=0):
    acc = acc if acc is not None else set()
    if not isinstance(ir, dict) or depth > MAXD:
        return acc
    if ir.get("kind") == "field":
        acc.add(ir["name"])
    for a in ir.get("args", []):
        fields_in(a, acc, depth + 1)
    return acc


# ---------- 직렬화 ----------
def sexpr(ir, depth=0):
    if not isinstance(ir, dict) or depth > 40:
        return "?"
    k = ir.get("kind")
    if k == "field":
        return "$" + str(ir["name"])
    if k == "root":
        return "${" + str(ir["name"]) + "}"
    if k == "param":
        return "@" + str(ir["name"])
    if k == "const":
        v = ir["value"]
        return v.get("__ref__", "?") if isinstance(v, dict) else repr(v)
    if k == "ctx":
        return sexpr(ir["args"][0], depth + 1)
    if k == "residue":
        return "<R:" + str(ir.get("op")) + ">"
    ax = "[" + ir["axis"] + "]" if ir.get("axis") else ""
    inner = [sexpr(a, depth + 1) for a in ir.get("args", []) if isinstance(a, dict)]
    p = ir.get("params") or {}
    for a in p.get("_args", []):
        inner.append("@" + a["__param__"] if isinstance(a, dict) and "__param__" in a else str(a))
    for kk, vv in p.items():
        if kk == "_args":
            continue
        inner.append(f"{kk}={sexpr(vv, depth+1) if isinstance(vv, dict) and 'kind' in vv else vv}")
    if ir.get("by"):
        inner.append(f"by={ir['by']}")
    return f"{ir.get('op','?')}{ax}(" + ", ".join(inner) + ")"


def strip(ir):
    if not isinstance(ir, dict):
        return ir
    return {k: ([strip(a) for a in v] if k == "args" else v)
            for k, v in ir.items() if k not in ("raw", "src")}


def shash(ir):
    return hashlib.sha1(json.dumps(strip(ir), sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------- 분류 (결정론적) ----------
# 코퍼스 양쪽(pandas 이름 + DSL 정규 이름)을 모두 인식한다.
FACTOR_TS = {
    # pandas 계열
    "rolling", "ewm", "expanding", "shift", "diff", "pct_change",
    "cumsum", "cumprod", "cummax", "cummin", "resample",
    # DSL 정규 계열
    "ts_delay", "ts_delta", "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max",
    "ts_rank", "ts_argmax", "ts_argmin", "ts_corr", "ts_cov", "ts_product",
    "ts_decay_linear", "ts_returns", "ts_slope", "ts_rsquare", "ts_resi",
    "ts_quantile", "ts_ewm", "ts_sma_cn",
}
FACTOR_CS = {"rank", "qcut", "pd.qcut", "transform", "cs_rank", "cs_scale"}
FACTOR_GRP = {"grp_demean"}
FACTOR_MTH = {"add", "sub", "mul", "div", "pow", "neg", "where", "signedpower",
              "min", "max", "gt", "lt", "ge", "le", "eq", "ne", "or_", "and_",
              "np.log", "np.log1p", "np.sqrt", "np.abs", "np.sign", "np.exp",
              "log", "abs", "sign", "sqrt"}
PREP_OPS = {"map", "replace", "ffill", "bfill", "fillna", "pd.to_datetime",
            "to_datetime", "field_dynamic", "select", "filter", "dropna"}


def classify(ir):
    ops = [o for o, ax in ops_in(ir)]
    axes = [ax for o, ax in ops_in(ir) if ax]
    flds = fields_in(ir)
    if not ops:
        return "field_ref"
    if set(ops) <= PREP_OPS:
        return "preprocessing"
    if (set(ops) & FACTOR_TS) or (set(ops) & FACTOR_CS) or (set(ops) & FACTOR_GRP) or axes:
        return "factor"
    if (set(ops) & FACTOR_MTH) and len(flds) >= 1:
        return "factor"
    return "preprocessing"


# ---------- 구조 지표 ----------
def tree_depth(ir):
    """실제 AST 최대 깊이 (재귀). sexpr 괄호 세기 아님."""
    if not isinstance(ir, dict):
        return 0
    ch = [a for a in ir.get("args", []) if isinstance(a, dict)]
    if not ch:
        return 1
    return 1 + max(tree_depth(a) for a in ch)


def tree_size(ir):
    if not isinstance(ir, dict):
        return 0
    return 1 + sum(tree_size(a) for a in ir.get("args", []) if isinstance(a, dict))


# ---------- 이름 정규화 (Stage B-lite) ----------
# 목적: 코퍼스마다 다른 표면 이름을 하나의 의미 어휘로. 구조는 건드리지 않는다.
_RENAME = {
    "diff": "ts_delta", "shift": "ts_delay", "pct_change": "ts_returns",
    "np.log": "log", "np.log1p": "log1p", "np.sqrt": "sqrt",
    "np.abs": "abs", "np.sign": "sign", "np.exp": "exp",
    "cumsum": "ts_cumsum", "cumprod": "ts_cumprod",
}
_ROLL_AGG = {"mean": "ts_mean", "std": "ts_std", "sum": "ts_sum", "min": "ts_min",
             "max": "ts_max", "skew": "ts_skew", "median": "ts_median",
             "var": "ts_var", "corr": "ts_corr", "cov": "ts_cov",
             "rank": "ts_rank", "apply": "ts_apply"}
_XSECT_AGG = {"mean": "mean", "median": "median", "std": "std", "sum": "sum",
              "min": "min", "max": "max", "rank": "rank"}


def _win_of(node):
    p = node.get("params") or {}
    if "window" in p:
        return p["window"]
    a = p.get("_args") or []
    return a[0] if a else None


def normalize(ir, depth=0, fold_returns_field=None):
    """표면 이름 -> 정규 어휘 + 대수 단순화 + ts_returns 재작성(0.4.0). 멱등."""
    if depth == 0:
        out = simplify(_normalize(ir, 0), 0)
        return fold_returns(out, FOLD_RETURNS_FIELD if fold_returns_field is None
                            else fold_returns_field)
    return _normalize(ir, depth)

def _normalize(ir, depth=0):
    if not isinstance(ir, dict) or depth > MAXD:
        return ir
    n = dict(ir)
    if "args" in ir:
        n["args"] = [_normalize(a, depth + 1) for a in ir["args"]]
    if n.get("kind") != "op":
        return n
    op, params = n["op"], dict(n.get("params") or {})

    # rolling/ewm/expanding + 집계 -> ts_* 융합
    if op in _ROLL_AGG and len(n["args"]) >= 1:
        inner = n["args"][0]
        if isinstance(inner, dict) and inner.get("kind") == "op" and \
           inner["op"] in ("rolling", "ewm", "expanding"):
            w = _win_of(inner)
            new_op = _ROLL_AGG[op] if inner["op"] != "ewm" else "ts_ewm"
            if inner["op"] == "expanding":
                w = "inf"
            n["op"] = new_op
            n["args"] = list(inner.get("args", [])) + list(n["args"][1:])
            if w is not None:
                params.pop("_args", None); params["window"] = w
            n["axis"] = n.get("axis") or inner.get("axis") or "TS"
            n["params"] = params
            return n

    # transform[CS/GRP](x, 'mean') -> cs_mean / grp_mean
    if op == "transform":
        a0 = (params.get("_args") or [None])[0]
        if isinstance(a0, str) and a0 in _XSECT_AGG:
            ax = n.get("axis")
            prefix = {"CS": "cs_", "GRP": "grp_", "TS": "ts_"}.get(ax, "")
            if prefix:
                n["op"] = prefix + _XSECT_AGG[a0]
                params["_args"] = [x for x in params["_args"] if x != a0]
                if not params["_args"]:
                    params.pop("_args")
                n["params"] = params
                return n

    # rank -> 축에 따라 cs_rank / ts_rank
    if op == "rank":
        ax = n.get("axis")
        if ax == "CS":
            n["op"] = "cs_rank"
        elif ax == "TS":
            n["op"] = "ts_rank"
        return n

    # ts_ewm 파라미터 의미 명시화 (0.4.1): 구표기 window -> span 이관 (멱등)
    if op == "ts_ewm" and "window" in params and "span" not in params:
        params["span"] = params.pop("window")
        n["params"] = params
        return n

    # 단순 개명 + 윈도우 파라미터 승격
    if op in _RENAME:
        n["op"] = _RENAME[op]
        a = params.get("_args") or []
        if len(a) == 1 and isinstance(a[0], (int, float)):
            params.pop("_args"); params["window"] = a[0]
            n["params"] = params
        return n
    return n


# ---------- 대수 단순화 ----------
# 101 코퍼스의 난독화 잔재(예: vwap*0.728 + vwap*(1-0.728) = vwap)를 접는다.
# 정규형 해시가 의미 단위로 붕괴하려면 필수.
def _c(v):
    return {"kind": "const", "value": v}

def _is_c(n):
    return isinstance(n, dict) and n.get("kind") == "const" and \
           isinstance(n.get("value"), (int, float)) and not isinstance(n.get("value"), bool)

def simplify(ir, depth=0):
    """상수 접기 + 항등원 제거 + 동일항 분배 접기. 멱등."""
    if not isinstance(ir, dict) or depth > MAXD:
        return ir
    n = dict(ir)
    if "args" in ir:
        n["args"] = [simplify(a, depth + 1) for a in ir["args"]]
    if n.get("kind") == "const":
        v = n.get("value")          # 2.0 과 2 는 같은 수 -> 정수로 통합
        if isinstance(v, float) and not isinstance(v, bool) and v.is_integer():
            n["value"] = int(v)
        return n
    if n.get("kind") != "op":
        return n
    # params 안의 정수값 float 도 통합 (window=20.0 -> 20)
    if n.get("params"):
        pr = dict(n["params"])
        for k2, v2 in list(pr.items()):
            if isinstance(v2, float) and not isinstance(v2, bool) and v2.is_integer():
                pr[k2] = int(v2)
        n["params"] = pr
    op, a = n["op"], n["args"]

    # 상수 접기
    if op in ("add", "sub", "mul", "div", "pow") and len(a) == 2 and _is_c(a[0]) and _is_c(a[1]):
        x, y = a[0]["value"], a[1]["value"]
        try:
            v = {"add": x + y, "sub": x - y, "mul": x * y,
                 "div": (x / y if y != 0 else None), "pow": x ** y}[op]
        except Exception:
            v = None
        if v is not None and isinstance(v, (int, float)):
            return _c(v)
    if op == "neg" and len(a) == 1 and _is_c(a[0]):
        return _c(-a[0]["value"])

    # 항등원 제거는 '정확한' 0/1 에만 적용한다 (부동소수 오차 한계 1e-15).
    # 1e-12 같은 0나눗셈 방지 상수는 의미가 있으므로 절대 지우지 않는다.
    EXACT = 1e-15
    if op == "mul" and len(a) == 2:
        for i, j in ((0, 1), (1, 0)):
            if _is_c(a[i]) and abs(a[i]["value"] - 1) < EXACT:
                return a[j]
    if op == "add" and len(a) == 2:
        for i, j in ((0, 1), (1, 0)):
            if _is_c(a[i]) and abs(a[i]["value"]) < EXACT:
                return a[j]
    if op in ("sub", "div") and len(a) == 2 and _is_c(a[1]):
        if op == "sub" and abs(a[1]["value"]) < EXACT:
            return a[0]
        if op == "div" and abs(a[1]["value"] - 1) < EXACT:
            return a[0]

    # 동일항 분배: mul(X,c1)+mul(X,c2) -> mul(X, c1+c2)  (X 는 구조해시로 동일 판정)
    if op == "add" and len(a) == 2:
        def split(t):
            if isinstance(t, dict) and t.get("kind") == "op" and t.get("op") == "mul" \
               and len(t.get("args", [])) == 2:
                l, r = t["args"]
                if _is_c(r): return l, r["value"]
                if _is_c(l): return r, l["value"]
            return t, 1.0
        x1, c1 = split(a[0]); x2, c2 = split(a[1])
        if shash(x1) == shash(x2):
            COEF_EPS = 1e-9      # 0.728317 + (1-0.728317) 류의 산술 잔차 허용
            tot = c1 + c2
            if abs(tot - 1) < COEF_EPS:
                return x1
            return simplify({"kind": "op", "op": "mul", "axis": None,
                             "args": [x1, _c(tot)], "params": {}}, depth + 1)
    return n


# ---------- ts_returns canonical 재작성 (0.4.0) ----------
# 배경: 코퍼스가 수익률을 sub(div(x, ts_delay(x,n)), 1) 등 산술 전개로만 표현해
# ts_returns 노출이 0회였고, 그 결과 LLM 이 ts_returns 를 절대 쓰지 않았다.
# 아래 규칙이 수익률의 산술 변형들을 canonical 하게 ts_returns 로 접는다.
#
#   R1 : sub(div(x, ts_delay(x,n)), 1)                    -> ts_returns(x, n)
#   R2 : div(sub(x, ts_delay(x,n)), ts_delay(x,n))        -> ts_returns(x, n)
#   R2b: div(ts_delta(x,n), ts_delay(x,n))                -> ts_returns(x, n)
#        (ts_delta(x,n) == x - ts_delay(x,n) 이므로 R2 와 동일 의미.
#         JPX 골드·158 코퍼스가 이 형태를 씀)
#   R3 : add(div(x, ts_delay(x,n)), -1)  (인자 순서 무관)  -> ts_returns(x, n)
#   R4 : field 'returns' -> ts_returns(close, 1)
#        (canonical 층 전용 치환. raw IR 은 각 프론트엔드가 보존하며,
#         실행기에서 returns == pct_change(1) == ts_returns(close,1) 수치 동일)
#
# 매칭 조건: x 구조 동일성은 shash 비교(문자열 치환 아님), window 는 양의 정수
# (정수값 float 허용 — simplify 가 int 로 통합), 상수는 정확히 1 / -1 (bool 제외).
# 적용: bottom-up 1패스 후 shash 불변까지 반복 (fixpoint) -> 멱등성 보장.
# ratio 형 div(x, ts_delay(x,n)) (= 1+r) 과 역수형 div(ts_delay(x,n), x)
# (158 roc_*) 는 의도적으로 접지 않는다: 트리가 오히려 깊어지고 원 의미가 다르다.

FOLD_RETURNS_FIELD = True     # R4 스위치 (코퍼스 재생성/벤치마크 공통 기본값)


def _pos_int_window(p):
    w = (p or {}).get("window")
    if isinstance(w, bool):
        return None
    if isinstance(w, int) and w > 0:
        return w
    if isinstance(w, float) and w > 0 and w.is_integer():
        return int(w)
    return None


def _delay_of(n):
    """ts_delay(x, n) 이면 (x, n) 반환, 아니면 None."""
    if isinstance(n, dict) and n.get("kind") == "op" and n.get("op") == "ts_delay" \
            and len(n.get("args", [])) == 1:
        w = _pos_int_window(n.get("params"))
        if w is not None:
            return n["args"][0], w
    return None


def _delta_of(n):
    """ts_delta(x, n) 이면 (x, n) 반환, 아니면 None."""
    if isinstance(n, dict) and n.get("kind") == "op" and n.get("op") == "ts_delta" \
            and len(n.get("args", [])) == 1:
        w = _pos_int_window(n.get("params"))
        if w is not None:
            return n["args"][0], w
    return None


def _is_exact(n, val):
    return _is_c(n) and float(n["value"]) == float(val)


def _mk_returns(x, w):
    return {"kind": "op", "op": "ts_returns", "axis": "TS",
            "args": [x], "params": {"window": int(w)}}


def _ratio_xn(n):
    """div(x, ts_delay(x,n)) 이면 (x, n)."""
    if isinstance(n, dict) and n.get("kind") == "op" and n.get("op") == "div" \
            and len(n.get("args", [])) == 2:
        x, d = n["args"]
        hit = _delay_of(d)
        if hit and shash(hit[0]) == shash(x):
            return x, hit[1]
    return None


def _fold_returns_node(n):
    """단일 op 노드에 R1/R2/R2b/R3 적용. 매치 시 새 노드, 아니면 원본."""
    if not (isinstance(n, dict) and n.get("kind") == "op"):
        return n
    op, a = n["op"], n.get("args", [])

    if op == "sub" and len(a) == 2 and _is_exact(a[1], 1):        # R1
        m = _ratio_xn(a[0])
        if m:
            return _mk_returns(*m)

    if op == "add" and len(a) == 2:                               # R3
        for u, v in ((a[0], a[1]), (a[1], a[0])):
            if _is_exact(v, -1):
                m = _ratio_xn(u)
                if m:
                    return _mk_returns(*m)

    if op == "div" and len(a) == 2:
        dn = _delay_of(a[1])
        if dn is not None:
            num = a[0]
            if isinstance(num, dict) and num.get("kind") == "op" \
                    and num.get("op") == "sub" and len(num.get("args", [])) == 2:
                x, d1 = num["args"]                               # R2
                h1 = _delay_of(d1)
                if h1 and shash(h1[0]) == shash(x) and shash(d1) == shash(a[1]):
                    return _mk_returns(x, h1[1])
            hd = _delta_of(num)                                   # R2b
            if hd and shash(hd[0]) == shash(dn[0]) and hd[1] == dn[1]:
                return _mk_returns(hd[0], hd[1])
    return n


def _returns_field_node(n):
    """R4: field 'returns' -> ts_returns(close, 1)."""
    if isinstance(n, dict) and n.get("kind") == "field" and n.get("name") == "returns":
        return _mk_returns({"kind": "field", "name": "close"}, 1)
    return n


def _bottom_up(n, rule, depth=0):
    if not isinstance(n, dict) or depth > MAXD:
        return n
    if n.get("args"):
        n = dict(n)
        n["args"] = [_bottom_up(a, rule, depth + 1) for a in n["args"]]
    return rule(n)


def fold_returns(ir, fold_returns_field=True):
    """R1~R3 (+R4) 를 fixpoint 까지 적용. 멱등. 새 dict 반환(원본 불변)."""
    cur = ir
    if fold_returns_field:
        cur = _bottom_up(cur, _returns_field_node)
    for _ in range(MAXD):
        nxt = _bottom_up(cur, _fold_returns_node)
        if shash(nxt) == shash(cur):
            return nxt
        cur = nxt
    raise RuntimeError("fold_returns: fixpoint 미수렴 (규칙 재귀 의심)")
