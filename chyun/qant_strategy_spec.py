"""
qant_strategy_spec — StrategySpec v0: 전략 '전체'의 슬롯 스키마.

배경: 벤치마크 X03 이 보여줬듯 "시총 상위 500 유니버스" 같은 요청은
팩터 IR 의 범위 밖이다. 서비스는 이를 거절하면 안 되고, 이 스키마의
universe/selection 슬롯으로 받아야 한다.

정직한 전제 (내 지적 ③): 현재 코퍼스는 features 슬롯만 실증 데이터가 있다.
selection/portfolio 는 JPX 대회 규칙(고정)에서 온 실물 1건(2nd)뿐이고,
universe/execution 은 골드 예제로만 채워진다. 스키마에 evidence 필드로
이 출처를 명시해 LLM 이 근거 없는 슬롯을 지어낼 때 추적 가능하게 한다.

v0 범위: 스키마 + 검증기 + 실물 골드 1건(JPX 2nd 추출) + 손작성 골드 1건.
risk 슬롯은 의도적으로 제외 (합의된 후순위).
"""
import json
from qant_validate import validate

# ---------------------------------------------------------------- 스키마
SLOT_ENUMS = {
    "universe.type":  {"all", "top_n_by", "explicit_list"},
    "selection.type": {"top_bottom_n", "top_n", "threshold", "all"},
    "portfolio.type": {"equal_weight", "rank_weight", "signal_weight"},
    "execution.rebalance": {"daily", "weekly", "monthly"},
}

SCHEMA_DOC = """
StrategySpec v0 (JSON):
{
 "name": str,
 "universe":  {"type":"all"|"top_n_by"|"explicit_list",
               "by": <필드, top_n_by 일 때>, "n": <정수>},
 "features":  [ {"name": str, "ir": <팩터 IR>} , ... ],   // 1개 이상
 "signal":    {"combine": "single"|"model"|"weighted_sum",
               "of": [<feature name>...],
               "model": {"kind": str, "params": {...}} | null},
 "selection": {"type":"top_bottom_n"|"top_n"|"threshold"|"all",
               "n": <정수> | null, "threshold": <수> | null},
 "portfolio": {"type":"equal_weight"|"rank_weight"|"signal_weight"},
 "execution": {"rebalance":"daily"|"weekly"|"monthly"},
 "evidence":  {"슬롯이름": "corpus"|"gold"|"user"|"assumed", ...}
}
"""


def validate_spec(spec_obj, ir_spec):
    """(errors, warnings) — 슬롯 구조 + features 의 각 IR 을 기존 검증기로."""
    E, W = [], []
    if not isinstance(spec_obj, dict):
        return [("E_SPEC", "root", "객체가 아님")], []

    # 필수 슬롯
    for slot in ("universe", "features", "signal", "selection",
                 "portfolio", "execution"):
        if slot not in spec_obj:
            E.append(("E_SPEC", slot, "필수 슬롯 누락"))
    if E:
        return E, W

    # enum 검사
    def enum_ok(path, val):
        allowed = SLOT_ENUMS[path]
        if val not in allowed:
            E.append(("E_SPEC", path, f"{val!r} (허용: {sorted(allowed)})"))
    enum_ok("universe.type", spec_obj["universe"].get("type"))
    enum_ok("selection.type", spec_obj["selection"].get("type"))
    enum_ok("portfolio.type", spec_obj["portfolio"].get("type"))
    enum_ok("execution.rebalance", spec_obj["execution"].get("rebalance"))

    if spec_obj["universe"].get("type") == "top_n_by":
        if not spec_obj["universe"].get("by") or not spec_obj["universe"].get("n"):
            E.append(("E_SPEC", "universe", "top_n_by 는 by/n 필요"))
    if spec_obj["selection"].get("type") in ("top_bottom_n", "top_n") and \
       not spec_obj["selection"].get("n"):
        E.append(("E_SPEC", "selection", "top*_n 은 n 필요"))

    # features: 각 IR 을 기존 3층 검증기의 1층으로
    feats = spec_obj.get("features") or []
    if not feats:
        E.append(("E_SPEC", "features", "팩터가 하나도 없음"))
    names = set()
    for i, f in enumerate(feats):
        nm = f.get("name")
        if not nm or nm in names:
            E.append(("E_SPEC", f"features[{i}]", "이름 누락/중복"))
        names.add(nm)
        errs, warns, _ = validate(f.get("ir", {}), ir_spec, path=f"features[{i}]")
        E += errs
        W += warns

    # signal 참조 무결성
    sig = spec_obj["signal"]
    for ref in (sig.get("of") or []):
        if ref not in names:
            E.append(("E_SPEC", "signal.of", f"미정의 feature 참조 '{ref}'"))
    if sig.get("combine") == "model" and not sig.get("model"):
        E.append(("E_SPEC", "signal", "combine=model 인데 model 없음"))

    # evidence 미기재 슬롯 경고 (근거 추적)
    ev = spec_obj.get("evidence") or {}
    for slot in ("universe", "selection", "portfolio", "execution"):
        if slot not in ev:
            W.append(("W_NO_EVIDENCE", slot, "근거 출처 미기재 (corpus/gold/user/assumed)"))
    return E, W


# ---------------------------------------------------------------- 골드 예제
def _F(n): return {"kind": "field", "name": n}
def _C(v): return {"kind": "const", "value": v}
def _op(o, args, axis=None, **p):
    return {"kind": "op", "op": o, "axis": axis, "args": args, "params": p}

def _ret(w):    # 수익률형 모멘텀
    return _op("div", [_F("close"), _op("ts_delay", [_F("close")], "TS", window=w)])

# 실물 골드: JPX 2위 솔루션에서 추출한 구조 (qant_strategy_graphs.json 근거).
# 팩터는 대표 3계열 × 20일만 수록 (전체 9개는 윈도우 {20,40,60} 파라미터화).
GOLD_JPX_2ND = {
    "name": "jpx_2nd_extracted",
    "universe": {"type": "all",
                 "_note": "대회가 2000종목 고정 — 코드에 유니버스 로직 없음"},
    "features": [
        {"name": "ret_20", "ir": _ret(20)},
        {"name": "vol_20", "ir": _op("ts_std", [_op("div",
            [_op("ts_delta", [_F("close")], "TS", window=1),
             _op("ts_delay", [_F("close")], "TS", window=1)])], "TS", window=20)},
        {"name": "ma_gap_20", "ir": _op("div", [_F("close"),
            _op("ts_mean", [_F("close")], "TS", window=20)])},
    ],
    "signal": {"combine": "model", "of": ["ret_20", "vol_20", "ma_gap_20"],
               "model": {"kind": "lgb.train",
                         "params": {"objective": "regression",
                                    "num_boost_round": 3000}}},
    "selection": {"type": "top_bottom_n", "n": 200, "threshold": None},
    "portfolio": {"type": "rank_weight"},
    "execution": {"rebalance": "daily"},
    "evidence": {"universe": "corpus", "features": "corpus", "signal": "corpus",
                 "selection": "corpus", "portfolio": "corpus",
                 "execution": "corpus"},
}

# 손작성 골드: 랭킹형이 아닌 예 (레짐 조건부) — 골드 균질화 방지(수정①).
GOLD_REGIME = {
    "name": "regime_momentum_or_lowvol",
    "universe": {"type": "top_n_by", "by": "cap", "n": 500},
    "features": [
        {"name": "mom_20", "ir": _op("cs_rank", [_ret(20)], "CS")},
        {"name": "lowvol", "ir": _op("sub", [_C(1),
            _op("cs_rank", [_op("ts_std", [_op("div",
                [_op("ts_delta", [_F("close")], "TS", window=1),
                 _op("ts_delay", [_F("close")], "TS", window=1)])],
                "TS", window=60)], "CS")])},
        {"name": "regime_up", "ir": _op("gt",
            [_op("ts_mean", [_F("close")], "TS", window=20),
             _op("ts_mean", [_F("close")], "TS", window=60)])},
        {"name": "sig", "ir": _op("where", [
            _op("gt", [_op("ts_mean", [_F("close")], "TS", window=20),
                       _op("ts_mean", [_F("close")], "TS", window=60)]),
            _op("cs_rank", [_ret(20)], "CS"),
            _op("sub", [_C(1), _op("cs_rank", [_op("ts_std", [_op("div",
                [_op("ts_delta", [_F("close")], "TS", window=1),
                 _op("ts_delay", [_F("close")], "TS", window=1)])],
                "TS", window=60)], "CS")])])},
    ],
    "signal": {"combine": "single", "of": ["sig"], "model": None},
    "selection": {"type": "top_n", "n": 50, "threshold": None},
    "portfolio": {"type": "equal_weight"},
    "execution": {"rebalance": "weekly"},
    "evidence": {"universe": "gold", "features": "gold", "signal": "gold",
                 "selection": "gold", "portfolio": "gold", "execution": "gold"},
}


if __name__ == "__main__":
    from qant_validate import derive_spec
    from pathlib import Path
    here = Path(__file__).resolve().parent
    p101 = here / "qant_101_ir.json"
    if not p101.exists():
        p101 = here / "qant_out" / "qant_101_ir.json"
    d = json.loads(p101.read_text(encoding="utf-8"))
    ir_spec = derive_spec([a["ir"] for a in d["alphas"]])

    for g in (GOLD_JPX_2ND, GOLD_REGIME):
        E, W = validate_spec(g, ir_spec)
        print(f"{g['name']:28s} 에러 {len(E)}  경고 {len(W)}")
        for c, pth, m in E:
            print(f"   E {c} {pth}: {m}")
        for c, pth, m in W[:4]:
            print(f"   W {c} {pth}: {m}")

    out = here / "qant_out"; out.mkdir(exist_ok=True)
    (out / "qant_strategy_gold.json").write_text(
        json.dumps({"schema_doc": SCHEMA_DOC,
                    "golds": [GOLD_JPX_2ND, GOLD_REGIME]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장: qant_out/qant_strategy_gold.json")
