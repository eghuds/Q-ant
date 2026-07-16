
"""
backtesting_utils
=================
백테스팅 SDK:

    from backtesting_utils import Dataset, Runner

    data = Dataset(type='ML', subset='test-sample')
    runner = Runner(data)
    agent = ...  # BaseAgent를 상속한 인스턴스
    result = runner.submit(agent)

공개 API는 이 파일에서 export하는 것들이 전부다.
내부 구현(모듈 구조)은 자유롭게 바뀔 수 있으니
반드시 `from backtesting_utils import ...` 형태로 사용할 것.
"""

from .dataset import Dataset
from .runner import Runner, BacktestResult
from .agent import BaseAgent, BuyAndHoldAgent, MovingAverageCrossAgent, MLModelAgent
from . import metrics

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Runner",
    "BacktestResult",
    "BaseAgent",
    "BuyAndHoldAgent",
    "MovingAverageCrossAgent",
    "MLModelAgent",
    "metrics",
]
