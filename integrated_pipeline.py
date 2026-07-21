"""
통합 파이프라인: 텍스트 입력 -> 실제 프로파일 추출 -> 매매 파라미터 계산
=======================================================================

UGRP.py      : 사용자의 자유서술 텍스트를 실제로 OpenAI API에 보내서
                UserProfile(고정 필드 + 동적 파라미터)을 추출한다.
trading_params_mapping.py : 그 UserProfile + 정형 질문 답변을 받아
                실제 매매 파라미터(TradingParams)로 계산한다.

이 파일은 위 두 개를 실제로 이어붙여서, "사용자가 입력한 텍스트가 그대로
최종 파라미터 값에 반영되는지"를 확인할 수 있는 end-to-end 스크립트다.

실행 전 준비:
    - UGRP.py, trading_params_mapping.py, integrated_pipeline.py가
      전부 같은 폴더에 있어야 함
    - pip install openai
    - (PowerShell) $env:OPENAI_API_KEY="sk-여기에실제키"

실행:
    python integrated_pipeline.py
"""

import os

from UGRP import UserProfile, process_user_turn
from trading_params_mapping import StructuredAnswers, build_trading_params


# ---------------------------------------------------------------------------
# 정형 질문 (버튼 선택) - 터미널에서는 숫자로 대체 입력
# ---------------------------------------------------------------------------

def ask_structured_answers() -> StructuredAnswers:
    print("먼저 몇 가지 정형 질문에 답해주세요.\n")

    print("1) 투자 목적이 무엇인가요?")
    print("   [1] 은퇴자금  [2] 단기수익  [3] 자산증식  [4] 학습목적")
    purpose_map = {
        "1": "RETIREMENT", "2": "SHORT_TERM_GAIN",
        "3": "WEALTH_GROWTH", "4": "LEARNING",
    }
    purpose = purpose_map[input("번호 선택> ").strip()]

    print("\n2) 손실이 발생하면 어떤 느낌이 드실 것 같나요?")
    print("   [1] 매우 불안함  [2] 다소 신경쓰임  [3] 감내 가능함")
    loss_map = {
        "1": "VERY_ANXIOUS", "2": "SOMEWHAT_CONCERNED", "3": "CAN_TOLERATE",
    }
    loss_feeling = loss_map[input("번호 선택> ").strip()]

    print("\n3) 투자 기간은 어느 정도를 생각하시나요?")
    print("   [1] 단기(1년 이내)  [2] 중기(1~5년)  [3] 장기(5년 이상)")
    horizon_map = {"1": "SHORT", "2": "MEDIUM", "3": "LONG"}
    time_horizon = horizon_map[input("번호 선택> ").strip()]

    return StructuredAnswers(
        purpose=purpose, loss_feeling=loss_feeling, time_horizon=time_horizon
    )


# ---------------------------------------------------------------------------
# 자유서술 텍스트 수집 (실제 UGRP.process_user_turn 호출 -> API 사용)
# ---------------------------------------------------------------------------

def collect_free_text_profile() -> UserProfile:
    profile = UserProfile()
    print(
        "\n투자 성향을 자유롭게 입력해주세요. "
        "여러 번 나눠 입력하셔도 됩니다.\n"
        "(입력 종료: 빈 줄에서 그냥 Enter)\n"
    )

    while True:
        text = input("입력> ").strip()
        if not text:
            break

        try:
            result = process_user_turn(profile, text)
        except Exception as e:
            print(f"[오류] API 호출 실패: {e}\n다시 입력해주세요.\n")
            continue

        if not result["relevant"]:
            print(f"[알림] {result['message']}\n")
            continue

        print(f"[해석] {result['base_reasoning']}")
        if result["newly_added_dynamic_params"]:
            print(f"[신규 파라미터 발견] {result['newly_added_dynamic_params']}")
        print()

    return profile


# ---------------------------------------------------------------------------
# 전체 실행
# ---------------------------------------------------------------------------

def run_full_pipeline():
    structured = ask_structured_answers()
    profile = collect_free_text_profile()

    print("\n" + "=" * 70)
    print("추출된 프로파일 (UGRP.py 실행 결과)")
    print("=" * 70)
    print(profile.summary())

    params = build_trading_params(structured, profile)

    print("\n" + "=" * 70)
    print("최종 매매 파라미터 (trading_params_mapping.py 실행 결과)")
    print("=" * 70)
    print(params.summary())


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
        print('PowerShell: $env:OPENAI_API_KEY="sk-여기에실제키"')
    else:
        run_full_pipeline()
