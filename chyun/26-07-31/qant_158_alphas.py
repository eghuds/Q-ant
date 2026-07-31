# %% ===== Alpha158 (vn.py / Qlib) -> 공통 IR =====
"""
vn.py 의 alpha_158.py 에서 158개 팩터 수식을 추출해 공통 IR 로 변환한다.

- 수식이 문자열 DSL (add_feature("ma_20", "ts_mean(close, 20) / close"))
- 파서는 101 파서(qant_101_alphas.P)를 상속하고 연산자 표만 교체
- 주의 1: vnpy 의 ts_greater/ts_less 는 이름과 달리 '원소별' max/min
- 주의 2: 라벨은 ts_delay(close, -3) 로 의도적 미래참조 -> 팩터와 분리 저장

실행:  python qant_158_alphas.py [alpha_158.py 경로]
       경로 생략 시 GitHub 에서 자동 다운로드
결과:  ./qant_out/qant_158_ir.json, qant_158_census.csv
"""
import ast as pyast
import json, re, sys, urllib.request
from pathlib import Path
from collections import Counter
import pandas as pd

from qant_ir import sexpr, shash, ops_in, fields_in, classify, normalize, tree_depth
from qant_101_alphas import P, tokenize, mk, const, is_num

OUTDIR = Path(__file__).resolve().parent / "qant_out"
OUTDIR.mkdir(exist_ok=True)
RAW_URL = ("https://raw.githubusercontent.com/vnpy/vnpy/master/"
           "vnpy/alpha/dataset/datasets/alpha_158.py")


# ---------------------------------------------------------------- 소스 확보
def load_source(argv):
    if len(argv) > 1 and Path(argv[1]).exists():
        return Path(argv[1]).read_text(encoding="utf-8")
    cache = OUTDIR / "alpha_158_source.py"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    print(f"[다운로드] {RAW_URL}")
    src = urllib.request.urlopen(RAW_URL, timeout=30).read().decode("utf-8")
    cache.write_text(src, encoding="utf-8")
    return src


# ---------------------------------------------------------------- 정적 언롤
def extract_features(src):
    """__init__ 본문의 add_feature 호출을 for 루프까지 언롤해 (이름, 수식) 수집.
    실행(exec)하지 않는다 — vnpy 의존성 없이 AST 만으로."""
    tree = pyast.parse(src)
    feats, label = [], None

    def fmt(node, env):
        # Constant 또는 JoinedStr(f-string) 을 env 로 평가
        if isinstance(node, pyast.Constant):
            return str(node.value)
        if isinstance(node, pyast.JoinedStr):
            out = []
            for v in node.values:
                if isinstance(v, pyast.Constant):
                    out.append(str(v.value))
                elif isinstance(v, pyast.FormattedValue) and \
                        isinstance(v.value, pyast.Name) and v.value.id in env:
                    out.append(str(env[v.value.id]))
                else:
                    return None
            return "".join(out)
        return None

    def walk(body, env):
        nonlocal label
        for st in body:
            if isinstance(st, pyast.Assign) and len(st.targets) == 1 and \
               isinstance(st.targets[0], pyast.Name):
                try:
                    env[st.targets[0].id] = pyast.literal_eval(st.value)
                except Exception:
                    pass
            elif isinstance(st, pyast.AnnAssign) and isinstance(st.target, pyast.Name) \
                 and st.value is not None:
                # windows: list[int] = [5,10,20,30,60] — 타입 주석 달린 할당
                try:
                    env[st.target.id] = pyast.literal_eval(st.value)
                except Exception:
                    pass
            elif isinstance(st, pyast.For) and isinstance(st.target, pyast.Name):
                try:
                    dom = pyast.literal_eval(st.iter)
                except Exception:
                    dom = env.get(getattr(st.iter, "id", None), [])
                for v in (dom or []):
                    walk(st.body, {**env, st.target.id: v})
            elif isinstance(st, pyast.Expr) and isinstance(st.value, pyast.Call):
                f = st.value.func
                if isinstance(f, pyast.Attribute) and f.attr in ("add_feature", "set_label"):
                    args = [fmt(a, env) for a in st.value.args]
                    if f.attr == "add_feature" and len(args) == 2 and all(args):
                        feats.append((args[0], args[1]))
                    elif f.attr == "set_label" and args and args[0]:
                        label = args[0]

    for node in pyast.walk(tree):
        if isinstance(node, pyast.FunctionDef) and node.name == "__init__":
            walk(node.body, {})
    return feats, label


# ---------------------------------------------------------------- DSL 파서
# 101 파서 상속, 연산자 표만 vnpy/Qlib 규약으로 교체.
TS158_1 = {  # (x, w) — 마지막 상수 인자가 윈도우
    "ts_delay": "ts_delay", "ts_mean": "ts_mean", "ts_std": "ts_std",
    "ts_sum": "ts_sum", "ts_min": "ts_min", "ts_max": "ts_max",
    "ts_rank": "ts_rank", "ts_argmax": "ts_argmax", "ts_argmin": "ts_argmin",
    "ts_slope": "ts_slope", "ts_rsquare": "ts_rsquare", "ts_resi": "ts_resi",
}
ELEM158 = {"ts_log": "log", "ts_abs": "abs"}   # ts_ 접두지만 원소별


class P158(P):
    def call(self, name, args):
        if name in ELEM158:
            return mk(ELEM158[name], args)
        if name == "ts_greater":               # 원소별 max (이름 함정)
            return mk("max", args)
        if name == "ts_less":                  # 원소별 min
            return mk("min", args)
        if name in TS158_1:
            params = {}
            if args and is_num(args[-1]):
                params["window"] = args.pop(-1)["value"]
            return mk(TS158_1[name], args, params, axis="TS")
        if name == "ts_corr":
            params = {}
            if args and is_num(args[-1]):
                params["window"] = args.pop(-1)["value"]
            return mk("ts_corr", args, params, axis="TS")
        if name == "ts_quantile":              # (x, w, q)
            params = {}
            if len(args) == 3 and is_num(args[-1]) and is_num(args[-2]):
                params["q"] = args.pop(-1)["value"]
                params["window"] = args.pop(-1)["value"]
            return mk("ts_quantile", args, params, axis="TS")
        raise SyntaxError(f"unknown function {name!r}")


def parse158(formula):
    return P158(tokenize(formula)).expr()


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    src = load_source(sys.argv)
    feats, label = extract_features(src)
    print(f"추출된 add_feature: {len(feats)}개  /  라벨: {label!r}")

    rows, fails = [], []
    for name, formula in feats:
        try:
            raw = parse158(formula)
            can = normalize(raw)
            rows.append({
                "corpus": "alpha158", "alpha_id": name, "formula": formula,
                "ir_raw": raw, "sexpr_raw": sexpr(raw), "hash_raw": shash(raw),
                "ir_canonical": can, "sexpr_canonical": sexpr(can),
                "hash_canonical": shash(can),
                "ir": can, "sexpr": sexpr(can), "hash": shash(can),
                "kind": classify(can), "depth": tree_depth(can),
                "ops": ops_in(can), "fields": sorted(fields_in(can)),
            })
        except Exception as e:
            fails.append((name, formula[:60], str(e)[:80]))

    print(f"파싱: {len(rows)}/{len(feats)}  실패: {len(fails)}")
    for nm, f, msg in fails:
        print(f"  [FAIL] {nm}: {f} -> {msg}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ir"} for r in rows])
    dup = len(df) - df["hash"].nunique()
    print(f"분류: {dict(df['kind'].value_counts())}  유니크 해시: {df['hash'].nunique()} (중복 {dup})")
    if dup:
        vc = df["hash"].value_counts()
        for h in vc[vc > 1].index:
            print("  [중복]", list(df[df['hash'] == h]['alpha_id']))

    # 축 자기검증: ts_* 는 전부 TS 여야 함
    axbad = sum(1 for r in rows for op, ax in r["ops"]
                if op.startswith("ts_") and ax != "TS")
    print(f"ts_* 축 오류: {axbad}건")

    # 라벨 (의도적 미래참조 — 팩터와 분리)
    label_row = None
    if label:
        lraw = parse158(label); lcan = normalize(lraw)
        label_row = {"formula": label, "ir_raw": lraw, "ir_canonical": lcan,
                     "ir": lcan, "sexpr": sexpr(lcan),
                     "note": "라벨 정의. ts_delay 음수 = 의도적 미래참조 (타깃이므로 정상)"}
        print(f"라벨 IR: {label_row['sexpr']}")

    # 연산자 센서스
    cen = Counter()
    for r in rows:
        for op, ax in r["ops"]:
            cen[op] += 1
    census = pd.DataFrame([{"op": o, "count": c} for o, c in cen.most_common()])
    census["cum_pct"] = (census["count"].cumsum() / census["count"].sum() * 100).round(1)
    print("\n=== Alpha158 연산자 센서스 ===")
    print(census.to_string(index=False))

    # 101 과의 어휘 비교
    p101 = OUTDIR / "qant_101_ir.json"
    if not p101.exists():
        p101 = Path(__file__).resolve().parent / "qant_101_ir.json"
    if p101.exists():
        d101 = json.loads(p101.read_text(encoding="utf-8"))
        ops101 = Counter()
        for a in d101["alphas"]:
            for op, ax in ops_in(normalize(a["ir"])):
                ops101[op] += 1
        both = sorted(set(cen) & set(ops101))
        only158 = sorted(set(cen) - set(ops101))
        print(f"\n101∩158 공통 연산자 {len(both)}종: {both}")
        print(f"158 전용 {len(only158)}종: {only158}")

    from qant_ir import NORMALIZER_VERSION
    meta = {"source": "Alpha158 (Qlib feature set, vn.py port)",
            "normalizer_version": NORMALIZER_VERSION,
            "url": RAW_URL, "license_note": "vnpy MIT",
            "n_parsed": len(rows)}
    out = {"meta": meta, "alphas": rows}
    if label_row:
        out["label"] = label_row
    with open(OUTDIR / "qant_158_ir.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1, default=str)
    census.to_csv(OUTDIR / "qant_158_census.csv", index=False)
    print("\n저장: qant_158_ir.json / qant_158_census.csv")
