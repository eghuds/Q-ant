"""성과 지표 계산 함수 모음. 모두 equity_curve(매 시점 총자산 리스트)를 입력으로 받는다."""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(equity_curve: list[float]) -> float:
    """총수익률. 0.05 = +5%"""
    if len(equity_curve) < 2:
        return 0.0
    return equity_curve[-1] / equity_curve[0] - 1.0


def max_drawdown(equity_curve: list[float]) -> float:
    """최대 낙폭(MDD). 고점 대비 최대 하락 비율. 0.2 = -20%"""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    return float(drawdown.max())


def sharpe_ratio(equity_curve: list[float], periods_per_year: int = 252) -> float:
    """샤프비율(무위험수익률 0 가정). 시점 단위가 하루가 아니면
    periods_per_year를 조정할 것 (예: 1분봉 ≈ 252일 * 381봉)."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 3:
        return 0.0
    rets = np.diff(eq) / eq[:-1]
    std = rets.std(ddof=1)
    if std == 0:
        return 0.0
    return float(rets.mean() / std * np.sqrt(periods_per_year))


def win_rate(trade_log: pd.DataFrame) -> float:
    """매도 시점 기준 이익 거래 비율. trade_log는 Runner가 만든 형식을 가정."""
    if trade_log.empty:
        return 0.0
    sells = trade_log[trade_log["action"] == "SELL"]
    if sells.empty or "pnl" not in sells.columns:
        return 0.0
    return float((sells["pnl"] > 0).mean())


def summarize(equity_curve: list[float], trade_log: pd.DataFrame,
              periods_per_year: int = 252) -> dict:
    """주요 지표를 한 번에 dict로 반환."""
    return {
        "total_return": total_return(equity_curve),
        "mdd": max_drawdown(equity_curve),
        "sharpe": sharpe_ratio(equity_curve, periods_per_year),
        "n_trades": int(len(trade_log)),
        "win_rate": win_rate(trade_log),
    }
