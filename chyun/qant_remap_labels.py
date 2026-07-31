# -*- coding: utf-8 -*-
"""
qant_remap_labels — 정규화 버전 상승 후 라벨 키 이관.

qant_labels.json 은 canonical hash 를 키로 쓰므로 normalizer 가 바뀌면
키가 무효화된다. 라벨 내용(카테고리/의미/키워드/방향)은 트리의 '의미'에
대한 것이라 재작성(구조 보존 변환) 후에도 유효하다 — 재라벨링(API 비용)
없이 alpha_id 조인으로 구→신 해시만 이관한다.

입력:  --old-corpus <구버전 ir.json ...>  --labels <구 labels.json>
       --new-corpus <신버전 ir.json ...>
출력:  qant_out/qant_labels.json (키 이관 + remap_status 표시)

주의: 재작성으로 canonical 이 바뀐 알파는 fingerprint_corr 이 구버전 실행
기준이다. 수치 동일성이 회귀로 보증되는 재작성(ts_returns)이라 값은 그대로
유효하지만, 이관 표시(remap_status=remapped)를 남겨 추적 가능하게 한다.

실행 예:
  python qant_remap_labels.py \
      --old-corpus old/qant_101_ir.json old/qant_158_ir.json \
      --labels old/qant_labels.json \
      --new-corpus qant_out/qant_101_ir.json qant_out/qant_158_ir.json
"""
import argparse
import json
from pathlib import Path

from qant_ir import NORMALIZER_VERSION


def load_alphas(paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for a in d["alphas"]:
            key = (a["corpus"], a["alpha_id"])
            out[key] = a["hash_canonical"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-corpus", nargs="+", required=True)
    ap.add_argument("--new-corpus", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="qant_out/qant_labels.json")
    args = ap.parse_args()

    old_h = load_alphas(args.old_corpus)
    new_h = load_alphas(args.new_corpus)

    # 구 해시 -> 신 해시 (alpha_id 조인). 같은 구 해시가 서로 다른 신 해시로
    # 가면 재작성이 결정론적이지 않다는 뜻이므로 즉시 실패시킨다.
    h_map = {}
    for key, oh in old_h.items():
        nh = new_h.get(key)
        if nh is None:
            continue
        if oh in h_map and h_map[oh] != nh:
            raise SystemExit(f"[오류] 구 해시 {oh} 가 두 신 해시로 갈라짐 "
                             f"({h_map[oh]} vs {nh}) — 재작성 비결정성 의심")
        h_map[oh] = nh

    lab = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    new_labels = {}
    n_remap = n_same = n_orphan = 0
    for oh, rec in lab["labels"].items():
        nh = h_map.get(oh)
        if nh is None:
            rec["remap_status"] = "orphan"      # 신 코퍼스에 대응 알파 없음
            n_orphan += 1
            nh = oh
        elif nh != oh:
            rec["remap_status"] = "remapped"
            rec["id_prev"] = oh
            n_remap += 1
        else:
            rec["remap_status"] = "unchanged"
            n_same += 1
        rec["id"] = nh
        if nh in new_labels:                    # 재작성으로 트리가 합쳐진 경우
            new_labels[nh].setdefault("merged_from", []).append(oh)
            continue
        new_labels[nh] = rec

    lab["labels"] = new_labels
    lab["meta"]["normalizer_version"] = NORMALIZER_VERSION
    lab["meta"]["remap"] = {"remapped": n_remap, "unchanged": n_same,
                            "orphan": n_orphan, "n_out": len(new_labels)}
    Path(args.out).write_text(json.dumps(lab, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"이관 완료: unchanged {n_same} / remapped {n_remap} / orphan {n_orphan}"
          f" -> {args.out} (총 {len(new_labels)}개, {NORMALIZER_VERSION})")


if __name__ == "__main__":
    main()
