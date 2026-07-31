"""
qant_genbench — "LLM이 우리 IR 문법으로 전략을 만들 수 있는가" 측정.

절차:  코퍼스 -> 문법 계약 -> 프롬프트(계약 + few-shot K개) -> 생성 -> 검증 -> 점수

핵심 설계:
  * few-shot 예시는 과제와 무관하게 고정하지 않고, 요청어와 겹치는 연산자를
    쓰는 알파를 골라 3~5개만 넣는다 (RAG 형태에 가깝게).
  * 과제에 '함정'을 섞는다: 31개 어휘로 표현 불가능한 요청.
    올바른 응답은 없는 연산자를 지어내지 않고 근사하거나 불가를 밝히는 것.
"""
import json, random, re, os
from qant_validate import derive_spec, validate, validate_json_text, spec_summary

random.seed(7)

# ---------------------------------------------------------------- 과제 세트
# tier: easy(단일 연산) / mid(합성) / hard(중첩·조건) / trap(어휘로 불가)
TASKS = [
    dict(id="T01", tier="easy",
         nl="20일 수익률(모멘텀)을 횡단면 순위로 매긴 신호"),
    dict(id="T02", tier="easy",
         nl="60일 종가 변동성이 낮을수록 높은 점수를 주는 신호"),
    dict(id="T03", tier="easy",
         nl="당일 거래량을 10일 평균 거래량으로 나눈 비율"),
    dict(id="T04", tier="mid",
         nl="20일 모멘텀 순위와 60일 변동성 역순위를 더한 결합 신호"),
    dict(id="T05", tier="mid",
         nl="종가와 거래량의 20일 시계열 상관계수를 횡단면 순위로"),
    dict(id="T06", tier="mid",
         nl="섹터 중립화한 vwap 대비 종가 괴리"),
    dict(id="T07", tier="hard",
         nl="거래량이 20일 평균 거래량보다 크면 5일 모멘텀의 반대 방향, "
            "아니면 0을 주는 조건부 신호"),
    dict(id="T08", tier="hard",
         nl="최근 10일 중 고가가 최대였던 날이 며칠 전인지를 순위화하고, "
            "그 값에 20일 수익률 부호를 곱한 신호"),
    dict(id="T09", tier="hard",
         nl="5일 수익률의 횡단면 순위에서 20일 수익률의 횡단면 순위를 뺀 "
            "단기 반전 신호를 섹터 중립화"),
    dict(id="T10", tier="hard",
         nl="14일 RSI가 30 이하인 종목에 매수 신호",
         note="지난 실험에서 where/lt/ts_sum 조합으로 표현 가능 판명 — 함정 아님"),
    # --- 모호: 정답 = 되물음(clarify). 트리 강행도 거절도 감점 ---
    dict(id="A01", tier="ambig",
         nl="거래량 대비 가격으로 뭔가 좋은 신호 하나 만들어줘",
         why="'대비'의 분자/분모와 '좋은'의 방향이 미정 — 명확화 필요"),
    dict(id="A02", tier="ambig",
         nl="요즘 오르는 종목 잡아줘",
         why="'요즘'의 기간과 순위/임계 방식이 미정"),
    # --- 함정: 현재 어휘로 표현 불가 ---
    dict(id="X02", tier="trap",
         nl="종가의 20일 지수이동평균(EMA)",
         why="ts_ewm 계열 연산자가 101 코퍼스에 없음"),
    dict(id="X03", tier="trap",
         nl="시가총액 상위 500종목만 유니버스로 제한한 뒤 모멘텀 순위",
         why="유니버스 필터(selection) 슬롯이 IR 어휘에 없음 — 팩터 표현식의 범위 밖"),
]


# ---------------------------------------------------------------- 프롬프트
def load_labels():
    import os
    for base in (".", "qant_out"):
        p = os.path.join(base, "qant_labels.json")
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))["labels"]
    return None


def pick_examples_labeled(labels, alphas, nl, k=4):
    """라벨 키워드 기반 검색. conflict 라벨은 가중치 하향."""
    nl_low = nl.lower()
    scored = []
    for a in alphas:
        lab = labels.get(a.get("hash_canonical", a.get("hash")))
        if not lab:
            continue
        kw_hits = sum(1 for kw in lab.get("keywords", [])
                      if kw and str(kw).lower() in nl_low)
        cat_hit = 1 if lab.get("category", "") and lab["category"] in nl_low else 0
        score = kw_hits * 2 + cat_hit
        # conflict 하향은 제거됨(지적 ⑤): 부호가 틀려도 키워드·의미는 정확하므로
        # 검색에서 밀어낼 이유가 없다. 방향은 라벨의 direction_sign 이 이미 교정됨.
        if lab.get("label_status") == "unverifiable":
            score *= 0.8
        if a.get("depth", 9) > 8:
            continue
        if score > 0:
            scored.append((score, -a.get("depth", 5), a))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [a for _, _, a in scored[:k]]


def pick_examples(spec, alphas, nl, k=4):
    """요청어 관련 예시 선택. 라벨 인덱스가 있으면 그걸 우선, 없으면 연산자 힌트."""
    labels = load_labels()
    if labels:
        got = pick_examples_labeled(labels, alphas, nl, k)
        if len(got) >= 2:          # 라벨 검색이 충분히 잡으면 그대로
            return got
    return _pick_examples_hint(spec, alphas, nl, k)


def _pick_examples_hint(spec, alphas, nl, k=4):
    """(폴백) 연산자 힌트 기반 선택."""
    hints = []
    if any(w in nl for w in ("순위", "랭크")):      hints += ["cs_rank"]
    if any(w in nl for w in ("변동성", "표준편차")):  hints += ["ts_std"]
    if any(w in nl for w in ("수익률", "모멘텀")):    hints += ["ts_delta", "ts_delay"]
    if "상관" in nl:                                hints += ["ts_corr"]
    if any(w in nl for w in ("섹터", "중립")):       hints += ["grp_demean"]
    if any(w in nl for w in ("조건", "이면", "크면")): hints += ["where", "lt"]
    if "평균" in nl:                                hints += ["ts_sum", "ts_mean"]
    if any(w in nl for w in ("최대", "며칠")):       hints += ["ts_argmax", "ts_max"]

    scored = []
    for a in alphas:
        ops = {o for o, _ in a["ops"]}
        score = len(ops & set(hints))
        # 너무 거대한 예시는 배제 (문맥 오염)
        if a["depth"] > 8:
            continue
        scored.append((score, -a["depth"], a))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    picked = [a for _, _, a in scored[:k]]
    if len(picked) < k:
        picked += [a for a in alphas if a not in picked][:k - len(picked)]
    return picked


SCHEMA_DOC = """\
전략을 아래 '수식 언어'로 표현한다. 함수 호출과 사칙연산만 쓴다.

  예)  cs_rank(ts_delta(close, 20))
       -1 * ts_corr(cs_rank(open), cs_rank(volume), 10)
       (close - open) / (high - low + 0.001)
       where(volume > ts_mean(volume, 20), -1 * ts_delta(close, 5), 0)

규칙:
  1. 아래 목록에 있는 함수만 쓴다. 새 함수를 만들지 않는다.
  2. +window 표시된 시계열 함수는 마지막 인자가 기간(양의 정수)이다.
     예: ts_mean(close, 20) / ts_corr(close, volume, 20)
  3. 미래를 참조하지 않는다. 기간에 음수를 쓰지 않는다.
  4. 축(TS/CS/GRP)은 함수 이름이 결정하므로 따로 쓰지 않는다.
  5. grp_ 함수는 마지막에 그룹 수준을 쓸 수 있다: grp_demean(close, sector)

응답은 반드시 다음 셋 중 하나의 JSON 이다:
  A. 컴파일:   {"type":"expr", "expr":"<수식 한 줄>"}
  B. 되물음:   {"type":"clarify", "question":"<한 문장>",
                "options":["<선택지1>","<선택지2>"]}
     -> 요청이 두 가지 이상으로 해석되거나 핵심 파라미터가 빠졌을 때.
        추측으로 만들지 말 것.
  C. 표현불가: {"type":"unsupported", "reason":"<이유>", "closest":"<대안 또는 null>"}
     -> 아래 함수들의 '조합'으로도 정말 불가능할 때만. 조합으로 가능하면
        반드시 A 로 만들어라. 쉽게 포기하지 말 것.
"""


def build_prompt(spec, alphas, task, k=4):
    ex = pick_examples(spec, alphas, task["nl"], k)
    from qant_dsl import to_dsl
    ex_txt = "\n".join(
        f'  {to_dsl(a.get("ir_canonical") or a["ir"])[:150]}'
        for a in ex)
    # 이름 인덱스 (레시피 사전): 요청에 지표 이름/별칭이 있으면 전개식을 제공.
    # 과제 전용 규칙이 아니라 일반 사전 검색이다 (피드백 §12).
    rec_txt = ""
    try:
        from qant_recipes import find_recipes
        hits = find_recipes(task["nl"])
    except Exception:
        hits = []
    if hits:
        lines = []
        for r in hits:
            pr = ", ".join(f"{k2}={v}" for k2, v in r["default_params"].items())
            lines.append(f'  {r["name"]}({pr}) = {r["expanded_dsl"]}')
            if r.get("notes"):
                lines.append(f'    # {r["notes"][:90]}')
        rec_txt = ("\n--- 지표 레시피 (이름 -> 전개식, 그대로 또는 변형해 사용 가능) ---\n"
                   + "\n".join(lines) + "\n")
    return f"""{SCHEMA_DOC}
{spec_summary(spec)}

--- 참고 예시 (실제 코퍼스에서 발췌) ---
{ex_txt}
{rec_txt}
--- 과제 ---
다음 요청을 위 수식 언어로 표현하라.
JSON 객체 하나만 출력한다. 설명, 코드펜스 금지.

요청: {task['nl']}
"""


# ---------------------------------------------------------------- 채점
def score_dir(gen_dir, spec):
    rows = []
    for t in TASKS:
        p = os.path.join(gen_dir, f"{t['id']}.json")
        if not os.path.exists(p):
            rows.append(dict(id=t["id"], tier=t["tier"], status="MISSING",
                             n_err=0, errs=[], n_warn=0, warns=[], nodes=0))
            continue
        txt = open(p, encoding="utf-8").read()
        # 응답 3형 해석 (+ 구버전 bare tree / {"error"} 하위호환)
        try:
            o = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M))
        except Exception:
            o = None
        rtype = None
        if isinstance(o, dict):
            if o.get("type") == "clarify":
                rtype = "clarify"
            elif o.get("type") == "unsupported" or "error" in o:
                rtype = "unsupported"
            elif o.get("type") == "tree" and isinstance(o.get("ir"), dict):
                txt = json.dumps(o["ir"], ensure_ascii=False)   # 트리만 검증으로
            # bare tree (구버전) 는 그대로 통과
        if rtype == "clarify":
            ok = (t["tier"] == "ambig")
            rows.append(dict(id=t["id"], tier=t["tier"],
                             status="CLARIFY-OK" if ok else "CLARIFY-BAD",
                             n_err=0, errs=[], n_warn=0, warns=[], nodes=0))
            continue
        if rtype == "unsupported":
            ok = (t["tier"] == "trap")
            rows.append(dict(id=t["id"], tier=t["tier"],
                             status="REFUSED-OK" if ok else "REFUSED-BAD",
                             n_err=0, errs=[], n_warn=0, warns=[], nodes=0))
            continue
        if t["tier"] == "ambig":
            # 모호 과제인데 트리를 강행 -> 추측 컴파일 (감점)
            errs, warns, n, _ = validate_json_text(txt, spec)
            rows.append(dict(id=t["id"], tier=t["tier"],
                             status="GUESSED" if not errs else "FAIL",
                             n_err=len(errs), errs=errs[:4],
                             n_warn=len(warns), warns=warns[:4], nodes=n))
            continue
        errs, warns, n, _ = validate_json_text(txt, spec)
        status = "PASS" if not errs else "FAIL"
        if t["tier"] == "trap" and not errs:
            status = "TRAP-SLIPPED"     # 불가능한 걸 통과시킴 = 문제
        rows.append(dict(id=t["id"], tier=t["tier"], status=status,
                         n_err=len(errs), errs=errs[:4],
                         n_warn=len(warns), warns=warns[:4], nodes=n))
    return rows


if __name__ == "__main__":
    import sys
    d = json.load(open("qant_101_ir.json"))
    alphas = d["alphas"]
    spec = derive_spec([a["ir"] for a in alphas])

    if len(sys.argv) > 1 and sys.argv[1] == "prompts":
        os.makedirs("prompts", exist_ok=True)
        for t in TASKS:
            open(f"prompts/{t['id']}.txt", "w", encoding="utf-8").write(
                build_prompt(spec, alphas, t))
        print(f"프롬프트 {len(TASKS)}개 생성 -> prompts/")
    else:
        rows = score_dir(sys.argv[1] if len(sys.argv) > 1 else "generated", spec)
        import collections
        print(f"{'ID':5s} {'tier':6s} {'status':13s} {'err':>3s}  {'노드':>4s}")
        for r in rows:
            print(f"{r['id']:5s} {r['tier']:6s} {r['status']:13s} {r['n_err']:3d}  {r['nodes']:4d}")
            for code, path, msg in r["errs"]:
                print(f"        └ {code:12s} {path[:46]:46s} {msg[:52]}")
            for code, path, msg in r.get("warns", []):
                print(f"        ~ {code:12s} {path[:46]:46s} {msg[:52]}")
        c = collections.Counter(r["status"] for r in rows)
        real = [r for r in rows if r["tier"] != "trap"]
        traps = [r for r in rows if r["tier"] == "trap"]
        print(f"\n일반 과제 {len(real)}개 중 통과 {sum(1 for r in real if r['status']=='PASS')}개")
        print(f"함정 과제 {len(traps)}개 중 올바른 거절 "
              f"{sum(1 for r in traps if r['status']=='REFUSED-OK')}개")
        print("상태 분포:", dict(c))
