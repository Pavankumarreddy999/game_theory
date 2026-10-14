"""
Shared backtesting engine for directional strategies (position in {-1, 0, 1}).
Pairs Trading and Market Making have their own P&L logic (see their modules)
since they don't fit the single-asset directional pattern.
"""

import numpy as np
import pandas as pd


def backtest_signal(df: pd.DataFrame, signal: pd.Series, price_col: str = "Adj Close",
                    cost_bps: float = 0.0) -> pd.DataFrame:
    """
    signal.shift(1) is used so today's position is based on yesterday's
    decision (avoids look-ahead bias -- you can't trade on a signal you
    compute using today's own closing price before today closes).
    """
    returns = df[price_col].pct_change()
    pos = signal.shift(1).fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat_returns = pos * returns - turnover * cost_bps / 10_000
    cumulative = (1 + strat_returns.fillna(0)).cumprod()
    return pd.DataFrame({
        "returns": returns,
        "signal": signal,
        "strategy_returns": strat_returns,
        "cumulative": cumulative,
    })


def performance_metrics(strategy_returns: pd.Series, periods_per_year: int = 252) -> dict:
    r = strategy_returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return {"total_return": 0, "annualized_return": 0, "annualized_vol": 0,
                "sharpe": 0, "max_drawdown": 0}

    total_return = (1 + r).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(r)) - 1
    annualized_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0

    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def pairs_performance(pairs_df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    ret_a = pairs_df["a"].pct_change()
    ret_b = pairs_df["b"].pct_change()
    strat_returns = pairs_df["pos_a"].shift(1).fillna(0) * ret_a + \
                    pairs_df["pos_b"].shift(1).fillna(0) * ret_b
    return performance_metrics(strat_returns, periods_per_year)
