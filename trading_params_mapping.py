"""
UserProfile -> 실제 매매 파라미터(TradingParams) 매핑
======================================================

이전 단계(user_profile_parameter_extraction.py)에서 만든 UserProfile
(고정 필드 + 동적 파라미터)을, HFT/ML/LLM 에이전트가 실제로 소비할 수 있는
숫자/카테고리 파라미터로 변환한다.

이 모듈은 API 호출이 없는 순수 계산 로직이다 (LLM 없이 규칙 기반 매핑).

입력 두 가지를 합쳐서 처리한다:
  1. StructuredAnswers - 온보딩 정형 질문(버튼 선택) 결과
     (purpose, loss_feeling, time_horizon)
  2. UserProfile - 자유서술 텍스트에서 추출된 고정+동적 파라미터

출력:
  TradingParams - 에이전트/실행 레이어가 실제로 참조하는 최종 파라미터 세트
"""

from dataclasses import dataclass, field
from typing import Any, Literal

# UGRP.py(=user_profile_parameter_extraction.py 동일 내용) 파일에서 정의한 클래스를 재사용
# 주의: 이 import가 동작하려면 UGRP.py가 이 파일과 "같은 폴더"에 있어야 함
from UGRP import UserProfile, DynamicParameter


# ---------------------------------------------------------------------------
# 1. 입력 - 정형 질문 결과
# ---------------------------------------------------------------------------

@dataclass
class StructuredAnswers:
    purpose: Literal["RETIREMENT", "SHORT_TERM_GAIN", "WEALTH_GROWTH", "LEARNING"]
    loss_feeling: Literal["VERY_ANXIOUS", "SOMEWHAT_CONCERNED", "CAN_TOLERATE"]
    time_horizon: Literal["SHORT", "MEDIUM", "LONG"]


# ---------------------------------------------------------------------------
# 2. 출력 - 최종 매매 파라미터
# ---------------------------------------------------------------------------

@dataclass
class TradingParams:
    risk_tolerance: float
    time_horizon: str
    loss_aversion: float
    min_confidence_to_act: float
    max_position_size_pct: float
    stop_loss_pct: float
    min_holding_period_days: int
    max_leverage: float
    portfolio_allocation: dict[str, float]
    excluded_sectors: list[str]
    agent_weights: dict[str, float]
    # 동적으로 발견된 파라미터들 (esg_preference 등) - 이름 -> 파싱된 값
    extra_constraints: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"risk_tolerance          : {self.risk_tolerance:.2f}",
            f"time_horizon            : {self.time_horizon}",
            f"loss_aversion           : {self.loss_aversion:.2f}",
            f"min_confidence_to_act   : {self.min_confidence_to_act:.2f}",
            f"max_position_size_pct   : {self.max_position_size_pct * 100:.1f}%",
            f"stop_loss_pct           : {self.stop_loss_pct * 100:.1f}%",
            f"min_holding_period_days : {self.min_holding_period_days}",
            f"max_leverage            : {self.max_leverage:.1f}x",
            f"portfolio_allocation    : {self.portfolio_allocation}",
            f"excluded_sectors        : {self.excluded_sectors}",
            f"agent_weights           : "
            f"{{HFT: {self.agent_weights['HFT']:.2f}, "
            f"ML: {self.agent_weights['ML']:.2f}, "
            f"LLM: {self.agent_weights['LLM']:.2f}}}",
        ]
        if self.extra_constraints:
            lines.append(f"extra_constraints       : {self.extra_constraints}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. 보조 함수
# ---------------------------------------------------------------------------

def linear_map(x: float, low: float, high: float,
                out_low: float = 0.0, out_high: float = 1.0) -> float:
    """x를 [low, high] 구간에서 [out_low, out_high] 구간으로 선형 변환.
    범위를 벗어나면 clamp."""
    if high == low:
        return out_low
    t = (x - low) / (high - low)
    t = max(0.0, min(1.0, t))
    return out_low + t * (out_high - out_low)


LOSS_FEELING_TO_AVERSION = {
    "VERY_ANXIOUS": 0.85,
    "SOMEWHAT_CONCERNED": 0.5,
    "CAN_TOLERATE": 0.2,
}

TIME_HORIZON_TO_HOLDING_DAYS = {
    "SHORT": 5,
    "MEDIUM": 180,
    "LONG": 720,
}


def parse_dynamic_value(param: DynamicParameter) -> Any:
    """DynamicParameter.value(문자열)를 dtype에 맞는 파이썬 값으로 변환."""
    raw = param.value
    try:
        if param.dtype == "bool":
            return raw.strip().lower() in ("true", "1", "yes")
        if param.dtype == "float":
            return float(raw)
        if param.dtype == "int":
            return int(float(raw))
        if param.dtype == "list":
            # 아주 단순한 리스트 파싱 (예: "['A','B']" 형태를 가정하지 않고
            # 콤마 구분 문자열도 처리)
            cleaned = raw.strip().strip("[]")
            return [s.strip().strip("'\"") for s in cleaned.split(",") if s.strip()]
        return raw  # string, categorical은 그대로
    except (ValueError, TypeError):
        # 파싱 실패 시 원본 문자열 그대로 반환 (매매 파라미터에 잘못된 타입이
        # 섞이는 것보다, 그대로 두고 상위 레이어에서 로그로 확인하는 게 안전)
        return raw


# ---------------------------------------------------------------------------
# 4. 핵심 매핑 로직
# ---------------------------------------------------------------------------

def compute_risk_tolerance(structured: StructuredAnswers, profile: UserProfile) -> float:
    """정형 질문(loss_feeling)과 자유서술(risk_tolerance_hint)을 결합.
    자유서술 쪽 confidence가 낮으면 정형 질문 비중을 더 크게 준다."""
    structured_score = 1.0 - LOSS_FEELING_TO_AVERSION[structured.loss_feeling]
    # (loss_feeling이 곧 risk_tolerance의 역이라고 가정 - 손실회피 높음=리스크성향 낮음)

    hint = profile.base_fields.get("risk_tolerance_hint")
    hint_conf = profile.base_fields.get("risk_tolerance_confidence", 0.0)

    if hint is None:
        return structured_score

    # confidence를 가중치로 사용한 가중평균
    weight_hint = hint_conf  # 0~1
    weight_structured = 1.0 - weight_hint
    return weight_structured * structured_score + weight_hint * hint


def compute_portfolio_allocation(profile: UserProfile) -> dict[str, float]:
    """company_size_preference + thematic_intent를 목표 비중으로 변환.
    실제 서비스라면 이 비율(80/20 등)은 온보딩 '확인/조정' 단계에서
    사용자가 직접 선택한 값을 받아야 하지만, 여기서는 기본값으로 대체."""
    size_pref = profile.base_fields.get("company_size_preference")
    thematic = profile.base_fields.get("thematic_intent")

    if size_pref and thematic == "FUTURE_GROWTH_SECTOR":
        return {size_pref: 0.8, "GROWTH_THEME_AUTO": 0.2}
    if size_pref:
        return {size_pref: 1.0}
    if thematic == "FUTURE_GROWTH_SECTOR":
        return {"GROWTH_THEME_AUTO": 1.0}
    return {"UNRESTRICTED": 1.0}  # 규모/테마 선호가 전혀 없으면 제약 없음


def compute_agent_weights(structured: StructuredAnswers, profile: UserProfile,
                            risk_tolerance: float) -> dict[str, float]:
    """이전에 설계한 compute_agent_weights 로직을 그대로 코드화."""
    weights = {"HFT": 0.0, "ML": 0.0, "LLM": 0.0}

    if structured.time_horizon == "SHORT":
        weights["HFT"] += 0.5
    elif structured.time_horizon == "MEDIUM":
        weights["HFT"] += 0.15
    else:  # LONG
        weights["HFT"] += 0.05

    weights["ML"] += linear_map(risk_tolerance, low=0.0, high=1.0,
                                  out_low=0.2, out_high=0.5)

    # 자유서술에서 발견된 "복잡한 의도"의 개수 - 배제섹터, 테마투자,
    # 그리고 동적으로 발견된 파라미터 전부가 신호에 포함됨
    complexity_signal = 0
    if profile.base_fields.get("excluded_sectors"):
        complexity_signal += 1
    if profile.base_fields.get("thematic_intent"):
        complexity_signal += 1
    complexity_signal += len(profile.dynamic_parameters)

    weights["LLM"] += linear_map(complexity_signal, low=0, high=4,
                                   out_low=0.2, out_high=0.5)

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def build_trading_params(
    structured: StructuredAnswers,
    profile: UserProfile,
) -> TradingParams:
    """전체 매핑 파이프라인의 진입점."""
    risk_tolerance = compute_risk_tolerance(structured, profile)
    loss_aversion = LOSS_FEELING_TO_AVERSION[structured.loss_feeling]

    # 확신 임계값 - 손실회피가 강할수록(=loss_aversion 높을수록) 더 확실할 때만 행동
    min_confidence_to_act = linear_map(loss_aversion, low=0.0, high=1.0,
                                         out_low=0.55, out_high=0.85)

    # 1회 최대 포지션 비중 / 손절폭 - 리스크 성향에 비례
    max_position_size_pct = linear_map(risk_tolerance, low=0.0, high=1.0,
                                         out_low=0.02, out_high=0.15)
    stop_loss_pct = linear_map(risk_tolerance, low=0.0, high=1.0,
                                 out_low=0.02, out_high=0.10)

    min_holding_period_days = TIME_HORIZON_TO_HOLDING_DAYS[structured.time_horizon]

    max_leverage = linear_map(risk_tolerance, low=0.0, high=1.0,
                                out_low=1.0, out_high=3.0)

    portfolio_allocation = compute_portfolio_allocation(profile)
    excluded_sectors = profile.base_fields.get("excluded_sectors", [])

    agent_weights = compute_agent_weights(structured, profile, risk_tolerance)

    extra_constraints = {
        name: parse_dynamic_value(p)
        for name, p in profile.dynamic_parameters.items()
    }

    return TradingParams(
        risk_tolerance=risk_tolerance,
        time_horizon=structured.time_horizon,
        loss_aversion=loss_aversion,
        min_confidence_to_act=min_confidence_to_act,
        max_position_size_pct=max_position_size_pct,
        stop_loss_pct=stop_loss_pct,
        min_holding_period_days=min_holding_period_days,
        max_leverage=max_leverage,
        portfolio_allocation=portfolio_allocation,
        excluded_sectors=excluded_sectors,
        agent_weights=agent_weights,
        extra_constraints=extra_constraints,
    )


# ---------------------------------------------------------------------------
# 5. 데모 (API 호출 없이, 이전 대화의 김민준님 케이스를 그대로 재현)
# ---------------------------------------------------------------------------

def demo():
    # 온보딩 정형 질문 답변 (이전 대화의 김민준님 케이스)
    structured = StructuredAnswers(
        purpose="WEALTH_GROWTH",
        loss_feeling="SOMEWHAT_CONCERNED",
        time_horizon="MEDIUM",
    )

    # 자유서술에서 추출됐다고 가정한 UserProfile (이전 대화의 추출 결과를 그대로 사용)
    profile = UserProfile()
    profile.base_fields = {
        "company_size_preference": "LARGE_CAP",
        "company_size_confidence": 0.92,
        "risk_tolerance_hint": 0.4,
        "risk_tolerance_confidence": 0.5,
        "thematic_intent": "FUTURE_GROWTH_SECTOR",
        "thematic_intent_confidence": 0.8,
        "sector_preference": [],
        "excluded_sectors": ["TOBACCO"],
    }
    profile.dynamic_parameters = {
        "esg_preference": DynamicParameter(
            name="esg_preference", description="ESG 등급이 높은 기업 선호",
            dtype="bool", value="true", value_range="true/false",
            confidence=0.9, rationale="명시적 언급",
        ),
        "dividend_preference": DynamicParameter(
            name="dividend_preference", description="꾸준한 배당 지급 기업 선호",
            dtype="bool", value="true", value_range="true/false",
            confidence=0.8, rationale="명시적 언급",
        ),
    }

    params = build_trading_params(structured, profile)

    print("=" * 70)
    print("UserProfile -> TradingParams 매핑 결과")
    print("=" * 70)
    print(params.summary())


if __name__ == "__main__":
    demo()
