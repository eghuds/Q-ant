"""
qant_dsl — LLM 입출력 표면용 문자열 수식 ↔ IR.

지적 ①② 반영.

① 표면: LLM 에게 JSON 트리 대신 문자열 수식을 쓰게 한다.
     JSON : {"kind":"op","op":"cs_rank","axis":"CS","args":[...],"params":{...}}  (186자)
     DSL  : cs_rank(ts_delta(close, 20))                                          (28자)
   같은 IR 로 파싱되므로 허브(IR)는 그대로다. 바뀌는 건 껍데기뿐.
   이득: 토큰 6~7배 절감, JSON 구조 파손 오류 계급 소멸, 코퍼스 예시 재사용.

② axis: 연산자 이름이 축을 완전히 결정한다(ts_*→TS, cs_*→CS, grp_*→GRP).
   따라서 DSL 에는 축을 쓸 자리가 없고, 파싱 시 레지스트리에서 유도해 붙인다.
   => LLM 이 축을 틀릴 방법 자체가 없어진다 (E_AXIS 원천 차단).

윈도우 인자 규약: 시계열 연산자의 '마지막 상수 인자'가 window 다.
   ts_mean(close, 20)          -> params {"window": 20}
   ts_corr(close, volume, 20)  -> params {"window": 20}
   ts_quantile(close, 20, 0.8) -> params {"window": 20, "q": 0.8}
"""
import re

from qant_registry import REGISTRY, BASE_FIELDS

ADV_RE = re.compile(r"^adv\d+$")
# 그룹 축의 세분 수준. grp_* 연산자의 마지막 인자로 올 수 있다.
GROUP_LEVELS = {"sector", "industry", "subindustry"}

TOK = re.compile(r"""
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op><=|>=|==|!=|\|\||&&|[-+*/^<>(),])
  | (?P<ws>\s+)
""", re.X)

BIN = {"+": "add", "-": "sub", "*": "mul", "/": "div", "^": "pow",
       "<": "lt", ">": "gt", "<=": "le", ">=": "ge", "==": "eq", "!=": "ne",
       "||": "or_", "&&": "and_"}


def axis_of(op):
    """연산자 이름에서 축을 유도 (②). 레지스트리가 단일 진리."""
    reg = REGISTRY.get(op)
    if not reg:
        return None
    axes = [a for a in reg["axes"] if a is not None]
    return axes[0] if len(axes) == 1 else None


def tokenize(s):
    out, i = [], 0
    while i < len(s):
        m = TOK.match(s, i)
        if not m:
            raise SyntaxError(f"토큰화 실패 (위치 {i}): {s[i:i+20]!r}")
        i = m.end()
        if m.lastgroup != "ws":
            out.append((m.lastgroup, m.group()))
    out.append(("eof", ""))
    return out


def _mk(op, args, params=None):
    return {"kind": "op", "op": op, "axis": axis_of(op),
            "args": args, "params": params or {}}


def _const(v):
    return {"kind": "const", "value": v}


def _is_num(n):
    return isinstance(n, dict) and n.get("kind") == "const" and \
        isinstance(n.get("value"), (int, float))


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i]

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def eat(self, v):
        k, s = self.next()
        if s != v:
            raise SyntaxError(f"{v!r} 예상, {s!r} 나옴")

    def parse(self):
        n = self.expr()
        if self.peek()[0] != "eof":
            raise SyntaxError(f"수식 뒤에 잔여 토큰: {self.peek()[1]!r}")
        return n

    def expr(self):
        return self.or_()

    def or_(self):
        n = self.and_()
        while self.peek()[1] == "||":
            self.next()
            n = _mk("or_", [n, self.and_()])
        return n

    def and_(self):
        n = self.cmp()
        while self.peek()[1] == "&&":
            self.next()
            n = _mk("and_", [n, self.cmp()])
        return n

    def cmp(self):
        n = self.add()
        while self.peek()[1] in ("<", ">", "<=", ">=", "==", "!="):
            o = self.next()[1]
            n = _mk(BIN[o], [n, self.add()])
        return n

    def add(self):
        n = self.mul()
        while self.peek()[1] in ("+", "-"):
            o = self.next()[1]
            n = _mk(BIN[o], [n, self.mul()])
        return n

    def mul(self):
        n = self.unary()
        while self.peek()[1] in ("*", "/"):
            o = self.next()[1]
            n = _mk(BIN[o], [n, self.unary()])
        return n

    def unary(self):
        if self.peek()[1] == "-":
            self.next()
            u = self.unary()
            return _const(-u["value"]) if _is_num(u) else _mk("neg", [u])
        if self.peek()[1] == "+":
            self.next()
            return self.unary()
        return self.power()

    def power(self):
        n = self.atom()
        if self.peek()[1] == "^":
            self.next()
            return _mk("pow", [n, self.unary()])
        return n

    def atom(self):
        kind, s = self.next()
        if s == "(":
            n = self.expr()
            self.eat(")")
            return n
        if kind == "num":
            return _const(float(s) if any(c in s for c in ".eE") else int(s))
        if kind == "id":
            name = s
            if self.peek()[1] == "(":
                self.next()
                args = []
                if self.peek()[1] != ")":
                    args.append(self.expr())
                    while self.peek()[1] == ",":
                        self.next()
                        args.append(self.expr())
                self.eat(")")
                return self.call(name, args)
            low = name.lower()
            if low in BASE_FIELDS or ADV_RE.match(low):
                return {"kind": "field", "name": low}
            if low in GROUP_LEVELS:
                return {"kind": "__grplevel__", "name": low}
            raise SyntaxError(f"미지의 식별자 {name!r} — 필드도 함수도 아님")
        raise SyntaxError(f"예상 밖 토큰 {s!r}")

    def call(self, name, args):
        reg = REGISTRY.get(name)
        if reg is None:
            raise SyntaxError(f"레지스트리에 없는 연산자 {name!r}")
        params = {}
        by = None
        if args and isinstance(args[-1], dict) and args[-1].get("kind") == "__grplevel__":
            by = [args.pop(-1)["name"]]
        # ts_quantile(x, w, q) 처럼 추가 상수 파라미터
        extra = sorted(reg.get("extra_params") or [])
        for key in reversed(extra):
            if len(args) > reg["min_args"] and _is_num(args[-1]):
                params[key] = args.pop(-1)["value"]
        if reg["window"] == "required":
            if args and _is_num(args[-1]) and len(args) > reg["min_args"] - 1:
                params["window"] = args.pop(-1)["value"]
        node = _mk(name, args, params)
        if by:
            node["by"] = by
        return node


def parse(text):
    """DSL 문자열 -> IR (axis 자동 부여)."""
    return Parser(tokenize(text.strip())).parse()


# ---------------------------------------------------------------- 직렬화
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/",
          "lt": "<", "gt": ">", "le": "<=", "ge": ">=",
          "eq": "==", "ne": "!=", "or_": "||", "and_": "&&"}
_PREC = {"or_": 1, "and_": 2, "lt": 3, "gt": 3, "le": 3, "ge": 3, "eq": 3, "ne": 3,
         "add": 4, "sub": 4, "mul": 5, "div": 5}


def to_dsl(ir, parent_prec=0):
    """IR -> DSL 문자열 (LLM 프롬프트 예시용). axis 는 출력하지 않는다."""
    if not isinstance(ir, dict):
        return "?"
    k = ir.get("kind")
    if k == "field":
        return str(ir["name"])
    if k == "const":
        v = ir["value"]
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    if k == "param":
        return "@" + str(ir.get("name"))
    if k != "op":
        return "?"
    op, args = ir["op"], ir.get("args", [])
    p = dict(ir.get("params") or {})

    if op == "neg" and len(args) == 1:
        return "-" + to_dsl(args[0], 6)
    if op in _INFIX and len(args) == 2:
        prec = _PREC[op]
        body = f"{to_dsl(args[0], prec)} {_INFIX[op]} {to_dsl(args[1], prec + 1)}"
        return f"({body})" if prec < parent_prec else body

    inner = [to_dsl(a, 0) for a in args]
    if ir.get("by"):
        inner.append(str(ir["by"][0]))
    if "window" in p:
        w = p["window"]
        inner.append(str(int(w)) if isinstance(w, float) and w.is_integer() else str(w))
    for key in sorted(p):
        if key in ("window", "_args"):
            continue
        inner.append(str(p[key]))
    return f"{op}(" + ", ".join(inner) + ")"


def roundtrip_ok(ir):
    """IR -> DSL -> IR 이 구조적으로 동일한지 (해시 비교)."""
    from qant_ir import shash
    try:
        back = parse(to_dsl(ir))
    except Exception:
        return False, None
    return shash(back) == shash(ir), to_dsl(ir)
