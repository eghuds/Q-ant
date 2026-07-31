"""
qant_label — 코퍼스 금융 라벨링 + 지문 교차검증 + RAG 키워드 인덱스.

목적 (T10 교훈): "RSI" 요청이 왔을 때 sump_14 가 검색되게 하려면
각 트리에 금융적 의미·동의어 키워드가 붙어 있어야 한다.

원칙:
  - LLM 라벨은 '제안'이다. canonical IR 위 메타데이터 층에만 붙는다 (구조 불변).
  - 방향 주장(direction_ref/sign)은 지문(합성 실행)으로 교차검증한다.
  - 3상태 (⑥): verified / unverifiable / conflict
      verified     지문이 주장과 일치 (|corr|>0.15, 부호 일치)
      unverifiable 참조신호가 없는 카테고리 (캔들·미시구조 등) — 라벨은 유지, 신뢰는 낮게
      conflict     지문이 주장과 반대 -> 검토 큐 (RAG 가중치 하향)
  - 같은-모델 상관 위험(⑥ 지적): gpt-4o 라벨을 gpt-4o 생성 검증에 그대로 쓰지
    않는다. 라벨은 '검색'에만 쓰고, '검증'은 골드/지문이 담당한다.

실행:
  python qant_label.py            # 라벨링 + 검증 -> qant_out/qant_labels.json
  python qant_label.py --dry-run  # 프롬프트 1개 미리보기 (API 호출 없음)
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter

from qant_hygiene import fingerprint, best_reference
from qant_eval import make_panel, ev

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "qant_out"; OUTDIR.mkdir(exist_ok=True)
MODEL = os.environ.get("QANT_MODEL", "gpt-4o-2024-11-20")
BATCH = 20

CATEGORIES = ["momentum", "reversal", "volatility", "volume_liquidity", "trend",
              "range_position", "candle_pattern", "correlation", "oscillator",
              "size_value", "sector_relative", "mixed", "other"]
REF_MAP = {  # 카테고리 -> 검증 가능한 참조신호 (없으면 unverifiable)
    "momentum": "mom20", "reversal": "revers", "volatility": "vol20",
    "volume_liquidity": "liq", "size_value": "size",
}

SYS = "\n".join([
    "너는 퀀트 팩터 라벨러다. 각 팩터 수식(S-expression)에 금융 라벨을 붙인다.",
    "표기: 함수형 수식. ts_*=종목내 시계열, cs_*=시점내 횡단면, grp_*=섹터내.",
    '출력: JSON {"items":[...]} — 입력 id 마다 하나.',
    "스키마:",
    '{"id": str,',
    f' "category": {CATEGORIES} 중 하나,',
    ' "meaning": str,          // 한국어 15단어 이내, 무엇을 재는가',
    ' "direction_ref": "mom20"|"mom5"|"vol20"|"size"|"liq"|"revers"|null,',
    ' "direction_sign": "+"|"-"|null,   // 그 참조와 기대되는 상관 부호',
    ' "keywords": [str, ...]   // 검색용 동의어 5개 이내. 한국어+영어 혼용.',
    '                          // 예: RSI류면 ["RSI","오실레이터","과매도","상대강도"]',
    "}",
])


def load_corpus():
    rows = []
    for fn in ("qant_101_ir.json", "qant_158_ir.json"):
        p = OUTDIR / fn
        if not p.exists():
            p = HERE / fn
        if p.exists():
            rows += json.loads(p.read_text(encoding="utf-8"))["alphas"]
    if not rows:
        sys.exit("[오류] 코퍼스 JSON 이 없습니다.")
    return rows


def ask_batch(items):
    from openai import OpenAI
    client = OpenAI(timeout=90, max_retries=0)
    user = "팩터 목록:\n" + json.dumps(items, ensure_ascii=False)
    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}],
                max_tokens=4000)
            o = json.loads(r.choices[0].message.content)
            return o.get("items", [])
        except Exception as e:
            print(f"  재시도 {attempt+1}: {str(e)[:100]}")
            time.sleep(3)
    return []


def verify(row, label, df, thr=0.15):
    """지문 교차검증 -> label 을 제자리 수정하고 상태 문자열 반환.

    지적 ⑤ 반영: conflict 의 절반은 '의미는 맞고 부호만 반대'였고, 그 군의
    |corr| 중앙값이 verified 군보다 높았다(0.727 vs 0.558). 검색에서 밀어낼
    게 아니라 부호를 뒤집어 승격해야 한다.

    상태:
      verified          주장대로 확인됨
      verified_flipped  부호만 반대라 자동 반전 후 확인 (원 주장은 flip_from 에 보존)
      reassigned        ref 배정이 틀려 지문상 최강 참조로 재배정
      unverifiable      어떤 참조와도 유의한 상관 없음 (구조적 속성 검사 대상)
    """
    ref = label.get("direction_ref") or REF_MAP.get(label.get("category"))
    sign = label.get("direction_sign")
    try:
        s = ev(row.get("ir_canonical") or row["ir"], df)
    except Exception as e:
        label["verify_note"] = f"실행 실패: {str(e)[:60]}"
        return "unverifiable", None

    fp = fingerprint(s, df)
    label["fingerprint_top"] = sorted(
        ((k, round(v, 3)) for k, v in fp.items() if v == v),
        key=lambda kv: -abs(kv[1]))[:3]

    if ref and sign in ("+", "-"):
        got = fp.get(ref)
        if got is not None and got == got:
            if (got > thr) if sign == "+" else (got < -thr):
                return "verified", round(float(got), 3)
            if abs(got) >= thr:                     # 부호만 반대 -> 자동 반전
                label["flip_from"] = sign
                label["direction_sign"] = "-" if sign == "+" else "+"
                return "verified_flipped", round(float(got), 3)

    # ref 배정 자체가 틀렸을 수 있다 -> 지문상 최강 참조로 재배정 시도
    bk, bs, _ = best_reference(s, df, min_abs=thr)
    if bk:
        label["reassign_from"] = ref
        label["direction_ref"] = bk
        label["direction_sign"] = bs
        return "reassigned", round(float(fp[bk]), 3)
    return "unverifiable", None


if __name__ == "__main__":
    rows = load_corpus()
    uniq = {}
    for r in rows:                          # canonical 해시 단위로만 라벨 (비용/일관성)
        uniq.setdefault(r["hash_canonical"], r)
    from qant_dsl import to_dsl
    items = [{"id": h, "expr": to_dsl(r["ir_canonical"])[:300]} for h, r in uniq.items()]
    print(f"코퍼스 {len(rows)}개 -> 유니크 트리 {len(items)}개")

    if "--dry-run" in sys.argv:
        print("\n--- 시스템 프롬프트 ---\n" + SYS)
        print("\n--- 배치 예시 (앞 3개) ---")
        print(json.dumps(items[:3], ensure_ascii=False, indent=1))
        sys.exit(0)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[오류] OPENAI_API_KEY 필요. 미리보기는 --dry-run")

    labels = {}
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        out = ask_batch(chunk)
        for o in (out or []):
            if isinstance(o, dict) and o.get("id"):
                labels[o["id"]] = o
        print(f"  {min(i+BATCH, len(items))}/{len(items)}  (누적 라벨 {len(labels)})")

    # 지문 교차검증
    df = make_panel()
    st = Counter()
    for h, r in uniq.items():
        lab = labels.get(h)
        if not lab:
            continue
        status, corr = verify(r, lab, df)
        lab["label_status"] = status
        lab["fingerprint_corr"] = corr
        st[status] += 1
    print(f"\n검증 결과: {dict(st)}")

    out = {"meta": {"model": MODEL, "n_labeled": len(labels),
                    "status_counts": dict(st)},
           "labels": labels}
    (OUTDIR / "qant_labels.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUTDIR/'qant_labels.json'}")
    unv = [h for h, l in labels.items() if l.get("label_status") == "unverifiable"]
    trust = sum(1 for l in labels.values()
                if l.get("label_status") in ("verified", "verified_flipped", "reassigned"))
    print(f"신뢰 가능한 방향 라벨: {trust}/{len(labels)} ({trust/max(1,len(labels)):.0%})")
    if unv:
        print(f"[검토 큐] unverifiable {len(unv)}건 — 참조 상관 없음. "
              f"구조적 속성 검사(qant_hygiene.PROPERTIES) 대상")
