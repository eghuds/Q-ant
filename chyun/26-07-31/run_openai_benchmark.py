"""
run_openai_benchmark — "LLM이 우리 AST 문법으로 유효한 전략을 만드는가" 실측.

VS Code 에서:
    set OPENAI_API_KEY=sk-...     (PowerShell 은 $env:OPENAI_API_KEY="sk-...")
    python run_openai_benchmark.py            # 생성 + 3층 검증 리포트
    python run_openai_benchmark.py --dry-run  # API 없이 프롬프트만 생성
    python run_openai_benchmark.py --score-only  # 이미 생성된 출력 재채점

필요 파일 (같은 폴더): qant_ir.py, qant_validate.py, qant_eval.py,
    qant_hygiene.py, qant_genbench.py, qant_101_alphas.py,
    qant_101_ir.json (+ 있으면 qant_out/qant_158_ir.json 자동 병합)

설계 원칙 (지금까지의 피드백 반영):
  * 문법 계약은 코퍼스에서 관측으로 도출 (101+158 병합) — 상상 금지
  * few-shot 은 과제당 관련 예시 4개만 (RAG 형태)
  * 함정 과제 포함 — 없는 연산자를 지어내는지 측정
  * 검증 3층: 문법(E/W) -> 실행 -> 위생(상수/횡단면/경험적 룩어헤드/의도)
  * 경고(W_*)는 실패가 아니라 검토 신호 — 코퍼스 자신도 경고를 냄 (Alpha#7)
  * API 실패가 파이프라인을 죽이지 않게: 타임아웃 60s, 재시도 2회, 과제별 격리
"""
import json, os, sys, time, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter

from qant_validate import derive_spec, validate_json_text
from qant_genbench import TASKS, build_prompt
from qant_eval import make_panel
from qant_hygiene import hygiene

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "qant_out"; OUTDIR.mkdir(exist_ok=True)
GEN = HERE / "generated_openai"; GEN.mkdir(exist_ok=True)
PROMPTS = HERE / "prompts_openai"; PROMPTS.mkdir(exist_ok=True)

MODEL = os.environ.get("QANT_MODEL", "gpt-4o-2024-11-20")

# 의도 검사: 사람 골드(qant_intent.GOLD_INTENT)가 유일한 기준.
# 생성 호출과 같은 곳에서 expect 를 받지 않는다 (오류 상관 방지 — T03 교훈).
from qant_intent import intent_for, properties_for
from qant_dsl import parse as dsl_parse


# ---------------------------------------------------------------- 코퍼스 로드
_versions = {}


def load_corpora():
    alphas = []
    p101 = HERE / "qant_101_ir.json"
    if not p101.exists():
        p101 = OUTDIR / "qant_101_ir.json"
    if p101.exists():
        d = json.loads(p101.read_text(encoding="utf-8"))
        alphas += d["alphas"]; _versions["101"] = d["meta"].get("normalizer_version")
    p158 = OUTDIR / "qant_158_ir.json"
    if p158.exists():
        d = json.loads(p158.read_text(encoding="utf-8"))
        alphas += d["alphas"]; _versions["158"] = d["meta"].get("normalizer_version")
    if not alphas:
        sys.exit("[오류] qant_101_ir.json 이 필요합니다 (qant_101_alphas.py 먼저 실행)")
    # ⑧ 버전 스큐 강제: 구버전 코퍼스를 조용히 소비하지 않는다
    from qant_ir import NORMALIZER_VERSION
    bad = [f for f, v in _versions.items() if v != NORMALIZER_VERSION]
    if bad:
        sys.exit(f"[오류] 코퍼스 정규화 버전 불일치 {bad} (필요: {NORMALIZER_VERSION}). "
                 f"qant_101_alphas.py / qant_158_alphas.py 를 다시 실행하세요.")
    if any("hash_canonical" not in a for a in alphas):
        sys.exit("[오류] 구버전 코퍼스(이중 저장 이전). 코퍼스를 재생성하세요.")
    srcs = Counter(a.get("corpus", "?") for a in alphas)
    print(f"코퍼스: {dict(srcs)}  (예시 풀 {len(alphas)}개)")
    return alphas


# ---------------------------------------------------------------- 생성
def call_openai(prompt):
    from openai import OpenAI
    client = OpenAI(timeout=60, max_retries=0)
    last = None
    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "너는 퀀트 전략 IR 컴파일러다. JSON 객체 하나만 출력한다."},
                    {"role": "user", "content": prompt}],
                max_tokens=2000)
            return r.choices[0].message.content
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenAI 호출 실패: {str(last)[:150]}")


def generate(spec, alphas, dry=False):
    for t in TASKS:
        prompt = build_prompt(spec, alphas, t)
        (PROMPTS / f"{t['id']}.txt").write_text(prompt, encoding="utf-8")
        if dry:
            continue
        out_p = GEN / f"{t['id']}.json"
        if out_p.exists():
            print(f"  {t['id']} 이미 있음 — 건너뜀 (재생성하려면 파일 삭제)")
            continue
        try:
            txt = call_openai(prompt)
            out_p.write_text(txt, encoding="utf-8")
            print(f"  {t['id']} 생성 완료 ({len(txt)}자)")
        except Exception as e:
            print(f"  {t['id']} 실패: {e}")
    if dry:
        print(f"프롬프트 {len(TASKS)}개 -> {PROMPTS}/  (API 호출 없음)")


# ---------------------------------------------------------------- 채점 (3층)
def score(spec):
    df = make_panel()
    rows = []
    for t in TASKS:
        p = GEN / f"{t['id']}.json"
        row = dict(id=t["id"], tier=t["tier"], nl=t["nl"][:34])
        if not p.exists():
            row.update(status="MISSING"); rows.append(row); continue
        txt = p.read_text(encoding="utf-8")

        # 거절 처리 (함정 과제의 정답 경로)
        try:
            obj0 = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M))
        except Exception:
            obj0 = None
        # json_object 모드가 {"ir": {...}} 처럼 감싸는 경우 언래핑
        if isinstance(obj0, dict) and "kind" not in obj0 and "error" not in obj0:
            for key in ("ir", "strategy", "expression", "tree", "result"):
                inner = obj0.get(key)
                if isinstance(inner, dict) and "kind" in inner:
                    obj0 = inner
                    txt = json.dumps(inner, ensure_ascii=False)
                    break
        if isinstance(obj0, dict) and obj0.get("type") == "clarify":
            ok = (t["tier"] == "ambig")
            row.update(status="CLARIFY-OK" if ok else "CLARIFY-BAD",
                       detail=str(obj0.get("question", ""))[:70])
            rows.append(row); continue
        if isinstance(obj0, dict) and (obj0.get("type") == "unsupported" or "error" in obj0):
            ok = (t["tier"] == "trap")
            row.update(status="REFUSED-OK" if ok else "REFUSED-BAD",
                       detail=str(obj0.get("reason", obj0.get("error", "")))[:70])
            rows.append(row); continue
        if isinstance(obj0, dict) and obj0.get("type") == "expr" and obj0.get("expr"):
            try:
                txt = json.dumps(dsl_parse(str(obj0["expr"])), ensure_ascii=False)
                row["expr"] = str(obj0["expr"])[:90]
            except Exception as e:
                row.update(status="FAIL-PARSE", detail=str(e)[:70])
                rows.append(row); continue
        elif isinstance(obj0, dict) and obj0.get("type") == "tree" and \
                isinstance(obj0.get("ir"), dict):
            txt = json.dumps(obj0["ir"], ensure_ascii=False)
        if t["tier"] == "ambig":
            row.update(status="GUESSED",
                       detail="모호 요청에 트리 강행 — clarify 가 정답")
            rows.append(row); continue

        # 1층: 문법
        errs, warns, n, obj = validate_json_text(txt, spec)
        row.update(nodes=n, n_err=len(errs), n_warn=len(warns),
                   errs=[f"{c}:{m[:40]}" for c, _, m in errs[:3]],
                   warns=[f"{c}:{m[:40]}" for c, _, m in warns[:3]])
        if errs:
            row.update(status="FAIL-SYNTAX"); rows.append(row); continue
        if t["tier"] == "trap":
            row.update(status="TRAP-SLIPPED"); rows.append(row); continue

        # 2+3층: 실행 + 위생
        iss, m = hygiene(obj, df, intent_for(t["id"], t["nl"]),
                         properties=properties_for(t["id"]))
        row.update(valid_ratio=m.get("valid_ratio"),
                   intent_corr=m.get("intent_corr"),
                   future_drift=m.get("future_drift"),
                   hygiene=[f"{c}:{msg[:40]}" for c, msg in iss[:3]])
        row.update(status="PASS" if not iss else "FAIL-HYGIENE")
        rows.append(row)
    return rows


def report(rows):
    print(f"\n{'ID':4s} {'tier':5s} {'상태':13s} {'문법W':>5s} {'의도상관':>8s}  비고")
    for r in rows:
        ic = r.get("intent_corr")
        print(f"{r['id']:4s} {r['tier']:5s} {r['status']:13s} "
              f"{r.get('n_warn', 0):5d} {(f'{ic:+.3f}' if ic is not None else '   -   '):>8s}  "
              f"{r['nl']}")
        for e in r.get("errs", []):   print(f"      └ E {e}")
        for w in r.get("warns", []):  print(f"      ~ W {w}")
        for h in r.get("hygiene", []): print(f"      └ H {h}")

    real = [r for r in rows if r["tier"] != "trap"]
    traps = [r for r in rows if r["tier"] == "trap"]
    syn = sum(1 for r in real if r["status"] not in ("FAIL-SYNTAX", "MISSING", "REFUSED-BAD"))
    full = sum(1 for r in real if r["status"] == "PASS")
    tr = sum(1 for r in traps if r["status"] == "REFUSED-OK")
    print(f"\n=== 요약 (모델: {MODEL}) ===")
    print(f"일반 {len(real)}개 — 문법 통과 {syn}  /  전층(위생 포함) 통과 {full}")
    print(f"함정 {len(traps)}개 — 올바른 거절 {tr}")
    print("* W_ 경고는 실패가 아니라 검토 신호. 코퍼스 원본(Alpha#7)도 경고를 낸다.")

    import csv
    with open(OUTDIR / "openai_benchmark_result.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        wtr = csv.writer(f)
        wtr.writerow(["id", "tier", "status", "n_err", "n_warn",
                      "intent_corr", "future_drift", "nl"])
        for r in rows:
            wtr.writerow([r["id"], r["tier"], r["status"], r.get("n_err", ""),
                          r.get("n_warn", ""), r.get("intent_corr", ""),
                          r.get("future_drift", ""), r["nl"]])
    print(f"저장: {OUTDIR/'openai_benchmark_result.csv'}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    score_only = "--score-only" in sys.argv

    alphas = load_corpora()
    spec = derive_spec([a["ir"] for a in alphas])
    print(f"문법 계약: 레지스트리 {len(spec['registry'])}종 / 관측 {len(spec['profile'])}종 / 필드 {len(spec['fields'])}종")

    if not score_only:
        if not dry and not os.environ.get("OPENAI_API_KEY"):
            sys.exit("[오류] OPENAI_API_KEY 환경변수를 설정하세요. "
                     "프롬프트만 보려면 --dry-run")
        generate(spec, alphas, dry=dry)
    if not dry:
        report(score(spec))
