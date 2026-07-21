

import json
import os
from dataclasses import dataclass, field
from typing import Literal, Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# 1. 고정(BASE) 파라미터 스키마
#    -> 이전에 정리한 "A. 텍스트 인터뷰에서 직접 추출되는 원시 필드" 표와 동일
# ---------------------------------------------------------------------------

BASE_PROFILE_TOOL = {
    "name": "extract_base_profile",
    "description": (
        "사용자의 자유서술 텍스트에서, 이미 정의된 고정 스키마 필드(대기업 선호, "
        "리스크 성향 힌트, 테마 투자 의도, 선호/배제 섹터)에 해당하는 정보만 추출한다. "
        "이 스키마에 없는 새로운 종류의 신호는 여기서 다루지 않는다 (별도 도구에서 처리)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean",
                "description": "이 텍스트가 투자 성향과 관련된 내용인지 여부"
            },
            "company_size_preference": {
                "type": ["string", "null"],
                "enum": ["LARGE_CAP", "MID_CAP", "SMALL_CAP", None],
                "description": "기업 규모 선호. 명시 안 됐으면 null"
            },
            "company_size_confidence": {"type": "number"},
            "risk_tolerance_hint": {
                "type": ["number", "null"],
                "description": "자유서술에서 유추되는 리스크 성향 (0.0~1.0), 없으면 null"
            },
            "risk_tolerance_confidence": {"type": "number"},
            "thematic_intent": {
                "type": ["string", "null"],
                "enum": ["FUTURE_GROWTH_SECTOR", None]
            },
            "thematic_intent_confidence": {"type": "number"},
            "sector_preference": {
                "type": ["array", "null"],
                "items": {"type": "string"}
            },
            "excluded_sectors": {
                "type": "array",
                "items": {"type": "string"}
            },
            "excluded_sectors_confidence": {"type": "number"},
            "reasoning": {
                "type": "string",
                "description": "왜 이렇게 추출했는지 한 문단 요약"
            }
        },
        "required": [
            "is_relevant", "company_size_preference", "company_size_confidence",
            "risk_tolerance_hint", "risk_tolerance_confidence",
            "thematic_intent", "thematic_intent_confidence",
            "sector_preference", "excluded_sectors", "excluded_sectors_confidence",
            "reasoning"
        ],
        "additionalProperties": False
    }
}

BASE_PROFILE_SYSTEM_PROMPT = """당신은 투자자 온보딩 인터뷰에서 사용자의 자유서술
텍스트를 분석하는 전문가입니다. 아래 고정된 필드에 해당하는 정보만 추출하세요.

## 규칙
1. 텍스트에 명시적으로 나오지 않은 내용을 추측해서 채우지 마세요.
   확실하지 않으면 null로 두고 confidence를 낮게 주세요.
2. 텍스트가 투자와 무관하면 is_relevant를 false로 하고 나머지 필드는
   모두 null 또는 빈 값으로 반환하세요.
3. 이 스키마에 없는 새로운 종류의 정보(예: ESG 선호, 특정 인물 관련 언급 등)는
   여기서 채우려 하지 말고, 해당 필드가 없다면 그냥 비워두세요.
"""


# ---------------------------------------------------------------------------
# 2. 동적(DYNAMIC) 파라미터 - 고정 스키마에 없는 새 신호를 "정의"하며 추출
# ---------------------------------------------------------------------------

DYNAMIC_PARAM_TOOL = {
    "name": "extract_dynamic_parameters",
    "description": (
        "사용자의 자유서술 텍스트에서, 이미 존재하는 고정 스키마 필드나 기존에 "
        "발견된 동적 파라미터로는 표현할 수 없는 새로운 투자 성향 신호를 찾아 "
        "새로운 파라미터로 정의하여 추출한다."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "new_parameters": {
                "type": "array",
                "description": "새로 정의된 파라미터 목록. 없으면 빈 배열.",
                "items": {
                    "type": "object",
                    "properties": {
                        "parameter_name": {
                            "type": "string",
                            "description": "snake_case 파라미터 이름 (예: esg_preference)"
                        },
                        "description": {
                            "type": "string",
                            "description": "이 파라미터가 무엇을 나타내는지"
                        },
                        "dtype": {
                            "type": "string",
                            "enum": ["float", "int", "bool", "categorical", "string", "list"]
                        },
                        "value": {
                            "type": "string",
                            "description": "추출된 값 (문자열로 직렬화, 예: '0.7', 'true', 'ESG_HIGH')"
                        },
                        "value_range": {
                            "type": "string",
                            "description": "이 파라미터가 가질 수 있는 현실적인 값의 범위"
                        },
                        "confidence": {"type": "number"},
                        "rationale": {
                            "type": "string",
                            "description": "텍스트의 어떤 부분에서 왜 이렇게 추출했는지"
                        }
                    },
                    "required": [
                        "parameter_name", "description", "dtype", "value",
                        "value_range", "confidence", "rationale"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["new_parameters"],
        "additionalProperties": False
    }
}

DYNAMIC_PARAM_SYSTEM_PROMPT = """당신은 투자자 온보딩 시스템의 파라미터 설계자입니다.
사용자의 자유서술 텍스트를 읽고, 이미 존재하는 파라미터들(고정 스키마 + 이전에
발견된 동적 파라미터)로는 표현되지 않는 "새로운 종류"의 투자 성향 신호가 있는지
찾아내세요.

## 규칙
1. 이미 존재하는 파라미터와 의미가 같거나 매우 유사하면 새로 만들지 마세요
   (아래 '이미 존재하는 파라미터 목록'을 반드시 확인하세요).
2. 텍스트에 명시적 근거가 없는 파라미터는 만들지 마세요. 추측성 파라미터 생성 금지.
3. 파라미터 하나는 하나의 명확한 의미 단위만 담아야 합니다
   (여러 의도를 한 파라미터에 억지로 합치지 마세요).
4. 새로 만들 파라미터가 없으면 new_parameters를 빈 배열로 반환하세요.
   무리하게 아무거나 만들어내지 마세요.
5. value는 dtype에 맞는 문자열로 직렬화하세요 (float면 "0.7", bool이면 "true"/"false").
"""


# ---------------------------------------------------------------------------
# 3. 데이터 구조
# ---------------------------------------------------------------------------

@dataclass
class DynamicParameter:
    name: str
    description: str
    dtype: str
    value: str
    value_range: str
    confidence: float
    rationale: str
    source_text: str = field(repr=False, default="")


@dataclass
class UserProfile:
    """온보딩 전체 대화에 걸쳐 누적되는 사용자 성향 프로파일.
    base_fields는 고정 스키마, dynamic_parameters는 대화가 진행되며
    계속 늘어날 수 있는 확장 파라미터."""
    base_fields: dict[str, Any] = field(default_factory=dict)
    dynamic_parameters: dict[str, DynamicParameter] = field(default_factory=dict)

    def known_dynamic_names(self) -> list[str]:
        return list(self.dynamic_parameters.keys())

    def summary(self) -> str:
        lines = ["[고정 필드]"]
        for k, v in self.base_fields.items():
            if not k.endswith("_confidence") and k != "reasoning":
                lines.append(f"  - {k}: {v}")
        if self.dynamic_parameters:
            lines.append("[동적 확장 필드]")
            for name, p in self.dynamic_parameters.items():
                lines.append(f"  - {name} ({p.dtype}): {p.value}  <- {p.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. OpenAI API 호출 (strict function calling으로 구조화 출력)
# ---------------------------------------------------------------------------

def _call_tool(system_prompt: str, user_prompt: str, tool: dict, model: str) -> dict:
    """tool(위에서 정의한 BASE_PROFILE_TOOL / DYNAMIC_PARAM_TOOL)을
    OpenAI function calling 스펙으로 감싸서 강제 호출하고, 반환된
    JSON 인자를 dict로 파싱해 돌려준다."""
    client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
                "strict": True,
            },
        }],
        # 반드시 이 함수를 호출하도록 강제 -> 자유 텍스트 응답 가능성을 차단
        tool_choice={"type": "function", "function": {"name": tool["name"]}},
    )

    message = response.choices[0].message

    if not message.tool_calls:
        raise RuntimeError(
            f"모델이 tool_calls를 반환하지 않았습니다. "
            f"content={message.content!r}"
        )

    call = message.tool_calls[0]
    return json.loads(call.function.arguments)


def extract_base_profile(user_text: str, model: str = "gpt-4o") -> dict:
    return _call_tool(BASE_PROFILE_SYSTEM_PROMPT, user_text, BASE_PROFILE_TOOL, model)


def extract_dynamic_parameters(
    user_text: str,
    known_base_field_names: list[str],
    known_dynamic_param_names: list[str],
    model: str = "gpt-4o",
) -> dict:
    context = (
        f"이미 존재하는 고정 스키마 필드: {known_base_field_names}\n"
        f"이미 발견된 동적 파라미터: {known_dynamic_param_names if known_dynamic_param_names else '(없음)'}\n\n"
        f"사용자 텍스트: \"{user_text}\""
    )
    return _call_tool(DYNAMIC_PARAM_SYSTEM_PROMPT, context, DYNAMIC_PARAM_TOOL, model)



# ---------------------------------------------------------------------------
# 5. 병합 로직 - 프로파일에 누적 반영
# ---------------------------------------------------------------------------

BASE_FIELD_NAMES = [
    "is_relevant", "company_size_preference", "risk_tolerance_hint",
    "thematic_intent", "sector_preference", "excluded_sectors",
]

RESERVED_NAMES = set(BASE_FIELD_NAMES)  # 동적 파라미터가 이 이름들과 겹치면 안 됨


def merge_base_fields(profile: UserProfile, base_result: dict) -> None:
    """새로 추출된 고정 필드 값을 프로파일에 병합. confidence가 더 높은 경우에만
    기존 값을 덮어써서, 낮은 확신도의 새 추출이 기존 확실한 값을 훼손하지 않게 함."""
    if not base_result.get("is_relevant", True):
        return

    field_confidence_pairs = [
        ("company_size_preference", "company_size_confidence"),
        ("risk_tolerance_hint", "risk_tolerance_confidence"),
        ("thematic_intent", "thematic_intent_confidence"),
    ]
    for field_name, conf_name in field_confidence_pairs:
        new_val = base_result.get(field_name)
        new_conf = base_result.get(conf_name, 0.0)
        if new_val is None:
            continue
        existing_conf = profile.base_fields.get(conf_name, -1.0)
        if new_conf >= existing_conf:
            profile.base_fields[field_name] = new_val
            profile.base_fields[conf_name] = new_conf

    # 리스트형 필드는 병합(합집합)
    for list_field in ("sector_preference", "excluded_sectors"):
        new_list = base_result.get(list_field) or []
        existing_list = profile.base_fields.get(list_field) or []
        profile.base_fields[list_field] = sorted(set(existing_list) | set(new_list))


def merge_dynamic_parameters(
    profile: UserProfile, dynamic_result: dict, source_text: str
) -> list[str]:
    """새로 발견된 동적 파라미터를 프로파일에 추가. 이름 충돌(고정 필드와 겹침,
    또는 이미 존재하는 동적 파라미터와 이름이 같음)은 걸러낸다.
    반환값: 실제로 새로 추가된 파라미터 이름 목록."""
    added = []
    for np in dynamic_result.get("new_parameters", []):
        name = np["parameter_name"]

        if name in RESERVED_NAMES:
            print(f"[경고] '{name}'은 고정 스키마 필드명과 충돌하여 무시됨")
            continue

        if name in profile.dynamic_parameters:
            # 이미 있는 동적 파라미터면, confidence 더 높은 쪽으로 갱신
            existing = profile.dynamic_parameters[name]
            if np["confidence"] > existing.confidence:
                profile.dynamic_parameters[name] = DynamicParameter(
                    name=name, description=np["description"], dtype=np["dtype"],
                    value=np["value"], value_range=np["value_range"],
                    confidence=np["confidence"], rationale=np["rationale"],
                    source_text=source_text,
                )
            continue

        profile.dynamic_parameters[name] = DynamicParameter(
            name=name, description=np["description"], dtype=np["dtype"],
            value=np["value"], value_range=np["value_range"],
            confidence=np["confidence"], rationale=np["rationale"],
            source_text=source_text,
        )
        added.append(name)

    return added


# ---------------------------------------------------------------------------
# 6. 전체 파이프라인 - 한 턴의 사용자 입력을 프로파일에 반영
# ---------------------------------------------------------------------------

def process_user_turn(
    profile: UserProfile, user_text: str, model: str = "gpt-4o"
) -> dict:
    """사용자 발화 한 번을 받아 base + dynamic 추출을 모두 수행하고
    프로파일에 병합한다. 결과 요약(dict)을 반환."""
    base_result = extract_base_profile(user_text, model=model)

    if not base_result.get("is_relevant", True):
        return {
            "relevant": False,
            "message": "투자와 무관한 입력으로 판단되어 분석을 건너뜁니다.",
        }

    merge_base_fields(profile, base_result)

    dynamic_result = extract_dynamic_parameters(
        user_text,
        known_base_field_names=BASE_FIELD_NAMES,
        known_dynamic_param_names=profile.known_dynamic_names(),
        model=model,
    )
    newly_added = merge_dynamic_parameters(profile, dynamic_result, source_text=user_text)

    return {
        "relevant": True,
        "base_reasoning": base_result.get("reasoning", ""),
        "newly_added_dynamic_params": newly_added,
    }


# ---------------------------------------------------------------------------
# 7. 데모
# ---------------------------------------------------------------------------

def demo():
    profile = UserProfile()

    turns = [
        # 1턴 - 고정 스키마로 대부분 커버되는 발화 (이전 대화의 김민준님 예시)
        "저는 대기업 위주로 안정적으로 가고 싶은데, 그래도 10년 뒤에 유망할 것 "
        "같은 섹터에는 어느 정도 비중을 두고 싶어요. 담배 관련 회사는 빼주세요.",

        # 2턴 - 고정 스키마에 없는 새로운 신호 (동적 파라미터 생성 기대)
        "그리고 저는 ESG 등급이 높은 회사를 선호해요. 환경 문제를 자주 일으키는 "
        "회사는 피하고 싶습니다.",

        # 3턴 - 또 다른 새로운 신호 + 이미 나온 동적 파라미터와 겹치는지 테스트
        "배당을 꾸준히 주는 회사면 더 좋을 것 같아요. 그리고 ESG는 저한테 꽤 "
        "중요한 기준이에요.",
    ]

    for i, text in enumerate(turns, start=1):
        print("=" * 70)
        print(f"[{i}번째 발화] {text}")
        print("=" * 70)
        result = process_user_turn(profile, text)
        print(result)
        print()
        print(profile.summary())
        print()


def interactive():
    """터미널에서 직접 텍스트를 입력하며 프로파일이 어떻게 누적되는지 테스트하는 모드."""
    profile = UserProfile()
    print("투자 성향을 자유롭게 입력하세요. (종료: 'q', 현재 프로파일 보기: 's')\n")

    while True:
        user_text = input("입력> ").strip()

        if user_text.lower() == "q":
            print("종료합니다.")
            break

        if user_text.lower() == "s":
            print()
            print(profile.summary())
            print()
            continue

        if not user_text:
            continue

        try:
            result = process_user_turn(profile, user_text)
        except Exception as e:
            print(f"[오류] API 호출 중 문제가 발생했습니다: {e}\n")
            continue

        print()
        if not result["relevant"]:
            print(f"[알림] {result['message']}")
        else:
            print(f"[해석] {result['base_reasoning']}")
            if result["newly_added_dynamic_params"]:
                print(f"[신규 파라미터 발견] {result['newly_added_dynamic_params']}")
        print()
        print(profile.summary())
        print()


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
        print('PowerShell: $env:OPENAI_API_KEY="sk-여기에실제키"')
        print("Mac/Linux : export OPENAI_API_KEY=sk-여기에실제키")
    else:
        mode = input("실행 모드를 선택하세요 [1] 데모(자동 3턴) [2] 직접 입력: ").strip()
        if mode == "2":
            interactive()
        else:
            demo()
