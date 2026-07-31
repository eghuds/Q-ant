"""
qant_intent — 의도 추출 (생성과 분리된 독립 경로).

내 지적 ① + ChatGPT 수정안 반영:
  - 벤치마크 과제는 '사람 골드'가 최우선 기준 (GOLD_INTENT 표).
    ※ 이 표는 사람이 확정/수정하는 곳이다. TODO 표시 = 아직 미확정.
  - 실사용자 요청처럼 골드가 없는 입력은 '생성과 별도의 호출'로 추출한다.
    같은 호출에 넣으면 요청 오독이 트리와 expect 에 동시에 들어가
    자기일관 검증이 무력화되기 때문 (T03 교훈).

추출 결과 스키마:
  {"ref": <참조신호 키>, "sign": "+"|"-", "confidence": 0~1}
  ref ∈ qant_hygiene.references() 의 키: mom20, mom5, vol20, size, liq, revers
  방향을 특정할 수 없으면 {"ref": null, "sign": null, "confidence": ...}
"""
import json, os, re

# ---------------------------------------------------------------- 사람 골드
# 벤치마크 13개 과제의 정답 의도. (직접 확인 후 확정할 것 — TODO 는 미확정)
GOLD_INTENT = {
    "T01": {"ref": "mom20", "sign": "+"},
    "T02": {"ref": "vol20", "sign": "-"},
    "T03": {"ref": "rel_volume", "sign": "+",
            "note": "당일거래량/10일평균 — rel_volume 참조로 검증 가능"},
    "T04": {"ref": "mom20", "sign": "+"},
    "T05": {"ref": "pv_corr", "sign": "+"},
    "T06": {"ref": None, "sign": None,
            "note": "참조 대신 sector_neutral 속성 검사로 판정 (GOLD_PROPERTY)"},
    "T07": {"ref": "mom5",  "sign": "-"},
    "T08": {"ref": None, "sign": None,
            "note": "argmax 경과일 × 부호 — 표준 참조 없음. bounded 속성만"},
    "T09": {"ref": "mom5",  "sign": "+"},
    "T10": {"ref": "revers", "sign": "+",
            "note": "RSI<30 매수 = 과매도 반전 — revers(-5일수익률)와 + 상관 기대"},
}

REF_KEYS = ("mom20", "mom5", "vol20", "size", "liq", "revers",
            "rel_volume", "range_pos", "pv_corr", "price_level")

# 과제별 '구조적 속성' 골드 (지적 ④).
# 참조 상관이 없어도 검증되는 선언형 성질. T06/T02 가 여기서 잡힌다.
GOLD_PROPERTY = {
    "T01": ["rank_range", "scale_invariant"],
    "T02": ["rank_range", "scale_invariant"],
    "T03": ["scale_invariant"],
    "T04": ["scale_invariant"],
    "T05": ["rank_range", "scale_invariant"],
    "T06": ["sector_neutral", "scale_invariant"],
    "T07": ["bounded"],
    "T08": ["bounded"],
    "T09": ["sector_neutral"],
    "T10": ["bounded", "scale_invariant"],
}


def properties_for(task_id):
    return GOLD_PROPERTY.get(task_id, [])


def gold_or_none(task_id):
    g = GOLD_INTENT.get(task_id)
    if g and g.get("ref"):
        return {"ref": g["ref"], "sign": g["sign"]}
    return None


# ---------------------------------------------------------------- LLM 추출
SYS = "\n".join([
    "너는 퀀트 전략 요청의 '의도 방향'만 추출하는 분석기다. 전략을 만들지 않는다.",
    "요청이 완성됐을 때 그 신호가 어떤 표준 신호와 상관을 가져야 하는지만 판단한다.",
    "표준 신호: mom20(20일 수익률), mom5(5일 수익률), vol20(수익률 변동성),",
    "          size(시가총액), liq(거래량 유동성), revers(단기 반전 = -5일 수익률)",
    '출력은 JSON 하나: {"ref": <키 또는 null>, "sign": "+"|"-"|null, "confidence": 0~1}',
    "해당되는 표준 신호가 없으면 ref=null. 애매하면 confidence 를 낮게.",
])


def extract_intent(nl, model=None):
    """생성과 완전히 분리된 호출로 요청 텍스트만 보고 의도를 추출."""
    model = model or os.environ.get("QANT_MODEL", "gpt-4o-2024-11-20")
    from openai import OpenAI
    client = OpenAI(timeout=30, max_retries=1)
    r = client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": f"요청: {nl}\nJSON:"}],
        max_tokens=200)
    try:
        o = json.loads(r.choices[0].message.content)
        ref = o.get("ref")
        if ref not in REF_KEYS:
            ref = None
        sign = o.get("sign") if o.get("sign") in ("+", "-") else None
        return {"ref": ref, "sign": sign,
                "confidence": float(o.get("confidence", 0.5)), "source": "llm"}
    except Exception:
        return {"ref": None, "sign": None, "confidence": 0.0, "source": "llm-fail"}


def intent_for(task_id, nl, use_llm=False):
    """골드 우선, 없고 use_llm 이면 독립 추출."""
    g = gold_or_none(task_id)
    if g:
        return {**g, "source": "gold"}
    if use_llm:
        return extract_intent(nl)
    return None
