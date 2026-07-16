"""
Runner: 백테스팅 실행 루프.

동작 방식
---------
1. dataset을 시간순으로 한 시점씩 순회하며 에이전트에게 observation을 준다.
2. 에이전트의 행동(BUY/SELL/HOLD)을 가상 계좌에 반영한다.
   - 룩어헤드 방지: 시점 t의 데이터를 보고 낸 주문은 t+1의 시가(open)에 체결.
     (t 종가를 보고 t 종가에 사는 것은 현실에서 불가능하므로)
   - execution='same_close' 옵션으로 단순 모드(그 시점 종가 체결)도 선택 가능.
3. 수수료를 반영하고, 매 시점 총자산을 기록한다.
4. 끝나면 지표(수익률, MDD, 샤프 등)와 거래 로그를 담은 BacktestResult 반환.

단순화 가정 (v0.1)
------------------
- 단일 종목, 전량 매수/전량 매도 (포지션 사이징 없음)
- 공매도 없음, 슬리피지 없음
- 이 가정들은 이후 버전에서 옵션으로 확장 예정
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import metrics as M
from .agent import ACTIONS, BaseAgent
from .dataset import Dataset


@dataclass
class BacktestResult:
    """runner.submit()의 반환값. result.metrics / result.trade_log 로 접근."""

    metrics: dict
    trade_log: pd.DataFrame
    equity_curve: list[float] = field(repr=False)
    equity_index: list = field(repr=False, default_factory=list)

    def equity_series(self) -> pd.Series:
        """총자산 곡선을 pandas Series로 (시각화용)."""
        return pd.Series(self.equity_curve, index=self.equity_index, name="equity")

    def __repr__(self):
        m = self.metrics
        return (
            f"BacktestResult(total_return={m['total_return']:+.2%}, "
            f"mdd={m['mdd']:.2%}, sharpe={m['sharpe']:.2f}, "
            f"n_trades={m['n_trades']}, win_rate={m['win_rate']:.1%})"
        )


class Runner:
    """Dataset을 받아두었다가 에이전트를 넣으면 백테스트를 실행한다.

    Parameters
    ----------
    dataset : Dataset
    initial_cash : 초기 자본 (기본 1천만 원)
    fee_rate : 편도 수수료+세금 비율 (기본 0.015% = 국내주식 온라인 수수료 수준)
    execution : 'next_open'(기본, 다음 봉 시가 체결) 또는 'same_close'
    periods_per_year : 샤프비율 연율화 계수. 일봉=252, 1분봉이면 252*381 권장
    """

    def __init__(
        self,
        dataset: Dataset,
        initial_cash: float = 10_000_000,
        fee_rate: float = 0.00015,
        execution: str = "next_open",
        periods_per_year: int = 252,
    ):
        if execution not in ("next_open", "same_close"):
            raise ValueError("execution은 'next_open' 또는 'same_close'")
        self.dataset = dataset
        self.initial_cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.execution = execution
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    def submit(self, agent: BaseAgent) -> BacktestResult:
        agent.reset()

        cash = self.initial_cash
        shares = 0
        avg_buy_price = 0.0  # 승률 계산용 평균 매수가

        pending_action: str | None = None  # next_open 모드에서 다음 봉에 체결할 주문
        trades: list[dict] = []
        equity_curve: list[float] = []
        equity_index: list = []

        for obs in self.dataset.iter_steps():
            open_p = float(obs.get("open", obs["close"]))
            close_p = float(obs["close"])

            # ---- (1) 직전 시점에 낸 주문을 이번 봉 시가에 체결 (next_open 모드)
            if self.execution == "next_open" and pending_action:
                cash, shares, avg_buy_price = self._execute(
                    pending_action, open_p, cash, shares, avg_buy_price,
                    obs["datetime"], trades,
                )
                pending_action = None

            # ---- (2) 에이전트 판단
            action = agent.act(obs)
            if action not in ACTIONS:
                raise ValueError(
                    f"에이전트가 잘못된 행동을 반환: {action!r} (가능: {ACTIONS})"
                )

            # ---- (3) 체결 처리
            if self.execution == "same_close":
                cash, shares, avg_buy_price = self._execute(
                    action, close_p, cash, shares, avg_buy_price,
                    obs["datetime"], trades,
                )
            else:  # next_open: 이번 판단은 다음 봉 시가에 체결되도록 보류
                if action != "HOLD":
                    pending_action = action

            # ---- (4) 이번 시점 총자산 기록 (종가 평가)
            equity_curve.append(cash + shares * close_p)
            equity_index.append(obs["datetime"])

        trade_log = pd.DataFrame(
            trades, columns=["datetime", "action", "price", "qty", "pnl"]
        )
        result_metrics = M.summarize(
            equity_curve, trade_log, periods_per_year=self.periods_per_year
        )
        return BacktestResult(
            metrics=result_metrics,
            trade_log=trade_log,
            equity_curve=equity_curve,
            equity_index=equity_index,
        )

    # ------------------------------------------------------------------
    def _execute(self, action, price, cash, shares, avg_buy_price, dt, trades):
        """주문 1건을 계좌에 반영. (전량 매수 / 전량 매도)"""
        if action == "BUY" and shares == 0 and price > 0:
            qty = int(cash // (price * (1 + self.fee_rate)))
            if qty > 0:
                cost = qty * price * (1 + self.fee_rate)
                cash -= cost
                shares += qty
                avg_buy_price = price
                trades.append(
                    {"datetime": dt, "action": "BUY", "price": price,
                     "qty": qty, "pnl": None}
                )
        elif action == "SELL" and shares > 0:
            proceeds = shares * price * (1 - self.fee_rate)
            pnl = proceeds - shares * avg_buy_price * (1 + self.fee_rate)
            cash += proceeds
            trades.append(
                {"datetime": dt, "action": "SELL", "price": price,
                 "qty": shares, "pnl": pnl}
            )
            shares = 0
            avg_buy_price = 0.0
        # HOLD 또는 체결 불가 조건이면 아무것도 하지 않음
        return cash, shares, avg_buy_price
