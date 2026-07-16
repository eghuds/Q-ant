"""
에이전트 규격과 예시 구현.

규격
-------------------
- 에이전트는 파이썬 클래스 인스턴스다. "내부 메모리"는 self 속성으로 구현한다.
- Runner는 매 시점 observation(dict)을 주고 행동(str)을 받는다:
      action = agent.act(observation)
- 행동은 "BUY" / "SELL" / "HOLD" 세 가지 문자열 중 하나.
- 백테스트 시작 전 Runner가 agent.reset()을 호출해 내부 상태를 초기화한다.

ML 파이프라인 담당이 만들 에이전트도 BaseAgent만 상속하면
Runner에 그대로 꽂힌다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

ACTIONS = ("BUY", "SELL", "HOLD")


class BaseAgent(ABC):
    """모든 에이전트가 따라야 하는 최소 규격."""

    def reset(self) -> None:
        """백테스트 시작 전 내부 메모리 초기화. 필요 없으면 그대로 둬도 된다."""

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> str:
        """한 시점의 데이터(observation)를 받아 'BUY'/'SELL'/'HOLD' 반환.

        observation은 Dataset의 한 행(row)을 dict로 만든 것:
        {'datetime': ..., 'open': ..., 'close': ..., 'feat_ret_1': ..., ...}
        """
        ...


# ----------------------------------------------------------------- 예시들 << 뭐임 이건..
class BuyAndHoldAgent(BaseAgent):
    """첫 시점에 사서 끝까지 보유. 파이프라인 동작 검증 및 성과 비교 기준선."""

    def reset(self):
        self._bought = False

    def act(self, observation):
        if not self._bought:
            self._bought = True
            return "BUY"
        return "HOLD"


class MovingAverageCrossAgent(BaseAgent):
    """단기/장기 이동평균 골든/데드크로스 전략.

    '내부 메모리'(최근 가격 큐)를 사용하는 예시.
    피쳐 없이 close만 있으면 어떤 데이터셋에서도 동작한다.
    """

    def __init__(self, short: int = 5, long: int = 20):
        if short >= long:
            raise ValueError("short는 long보다 작아야 합니다.")
        self.short, self.long = short, long

    def reset(self):
        self._prices: deque = deque(maxlen=self.long)

    def act(self, observation):
        self._prices.append(float(observation["close"]))
        if len(self._prices) < self.long:
            return "HOLD"  # 이동평균을 계산할 만큼 데이터가 쌓일 때까지 대기
        prices = list(self._prices)
        ma_s = sum(prices[-self.short:]) / self.short
        ma_l = sum(prices) / self.long
        if ma_s > ma_l:
            return "BUY"
        if ma_s < ma_l:
            return "SELL"
        return "HOLD"


class MLModelAgent(BaseAgent):
    """학습된 ML 모델을 감싸는 어댑터.

    predict(X) 메서드를 가진 모델(사이킷런 스타일)이면 무엇이든 꽂을 수 있다.
    모델의 예측(1=상승, 0=하락)을 매매 행동으로 변환한다.

    Parameters
    ----------
    model : predict(2D array-like) -> 1D array 를 지원하는 객체
    feature_columns : 모델이 학습에 사용한 피쳐 컬럼 이름 순서
    buy_threshold / sell_threshold :
        predict_proba가 있으면 확률 기반으로 매매 강도를 조절할 수 있다.
        없으면 predict 결과 1→BUY, 0→SELL로 단순 매핑.
    """

    def __init__(self, model, feature_columns: list[str],
                 buy_threshold: float = 0.55, sell_threshold: float = 0.45):
        self.model = model
        self.feature_columns = list(feature_columns)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def act(self, observation):
        x = [[float(observation[c]) for c in self.feature_columns]]

        # 확률 예측을 지원하면 임계값 기반으로 판단 (애매하면 HOLD)
        if hasattr(self.model, "predict_proba"):
            p_up = float(self.model.predict_proba(x)[0][-1])
            if p_up >= self.buy_threshold:
                return "BUY"
            if p_up <= self.sell_threshold:
                return "SELL"
            return "HOLD"

        pred = int(self.model.predict(x)[0])
        return "BUY" if pred == 1 else "SELL"
