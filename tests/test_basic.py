"""기본 동작 검증. 실행: pytest tests/ -v  (pip install pytest 필요)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from backtesting_utils import Dataset, Runner, BaseAgent, BuyAndHoldAgent
from backtesting_utils.metrics import max_drawdown, total_return


def make_df(n=100, price_path=None):
    """테스트용 미니 데이터프레임."""
    dt = pd.date_range("2026-01-01 09:00", periods=n, freq="1min")
    close = price_path if price_path is not None else np.linspace(100, 110, n)
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "datetime": dt,
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1000.0, 
        "feat_x": np.arange(n, dtype=float),
        "target": 1,
    })


def test_dataset_wraps_dataframe():
    ds = Dataset(type="ML", subset="unit", df=make_df())
    assert len(ds) == 100
    assert ds.feature_columns == ["feat_x"]
    assert ds.has_target
    assert list(ds.X().columns) == ["feat_x"]


def test_dataset_sorts_by_datetime():
    df = make_df().iloc[::-1]  # 역순으로 넣어도
    ds = Dataset(type="ML", subset="unit", df=df)
    assert ds.df["datetime"].is_monotonic_increasing  # 시간순으로 정렬


def test_buy_and_hold_gains_on_rising_market():
    """가격이 100→110 오르는 시장에서 바이앤홀드는 이익이어야 한다."""
    ds = Dataset(type="ML", subset="unit", df=make_df())
    result = Runner(ds, fee_rate=0.0).submit(BuyAndHoldAgent())
    assert result.metrics["total_return"] > 0.08  # 수수료 0이면 ~+10%


def test_invalid_action_raises():
    class BadAgent(BaseAgent):
        def act(self, obs):
            return "PANIC"

    ds = Dataset(type="ML", subset="unit", df=make_df(10))
    with pytest.raises(ValueError):
        Runner(ds).submit(BadAgent())


def test_next_open_execution_delays_fill():
    """next_open 모드: t시점 판단은 t+1 시가에 체결되어야 한다."""
    class BuyOnceAgent(BaseAgent):
        def reset(self):
            self.done = False
        def act(self, obs):
            if not self.done:
                self.done = True
                return "BUY"
            return "HOLD"

    close = np.array([100.0, 105.0, 105.0, 105.0])
    ds = Dataset(type="ML", subset="unit", df=make_df(4, price_path=close))
    result = Runner(ds, fee_rate=0.0, execution="next_open").submit(BuyOnceAgent())
    # 첫 판단(t=0)이 t=1 시가 105에 체결됐다면 매수가는 105
    assert result.trade_log.iloc[0]["price"] == 105.0


def test_mdd_and_return_math():
    eq = [100, 120, 90, 130]
    assert total_return(eq) == pytest.approx(0.30)
    assert max_drawdown(eq) == pytest.approx((120 - 90) / 120)
