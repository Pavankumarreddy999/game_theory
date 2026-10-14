"""
Pairs Trading (Statistical Arbitrage).
Requires two price series (same industry, expected to move together).

Steps:
1. Regress price_a on price_b to find the hedge ratio (beta).
2. Compute the spread = price_a - beta * price_b.
3. Check cointegration with the Augmented Dickey-Fuller test (informational).
4. Convert the spread into a z-score.
5. Trade: short A/long B when z > entry_z, long A/short B when z < -entry_z,
   close when |z| < exit_z.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    X = add_constant(price_b.values)
    model = OLS(price_a.values, X).fit()
    beta = model.params[1]
    return beta


def check_cointegration(spread: pd.Series) -> float:
    """Returns the ADF test p-value. p < 0.05 suggests the spread is stationary
    (mean-reverting), i.e. the pair is likely cointegrated."""
    result = adfuller(spread.dropna())
    return result[1]  # p-value


def generate_pairs_signals(
    price_a: pd.Series,
    price_b: pd.Series,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
):
    """
    Returns a DataFrame with the spread, z-score, and positions for A and B.
    Uses a rolling window so the hedge ratio and z-score adapt over time
    (avoids look-ahead bias — only past data is used at each point).
    """
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    beta_series = pd.Series(index=df.index, dtype=float)
    spread = pd.Series(index=df.index, dtype=float)

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]
        beta = compute_hedge_ratio(window["a"], window["b"])
        beta_series.iloc[i] = beta
        spread.iloc[i] = df["a"].iloc[i] - beta * df["b"].iloc[i]

    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    zscore = (spread - rolling_mean) / rolling_std

    pos_a = pd.Series(0, index=df.index)
    pos_b = pd.Series(0, index=df.index)

    position_state = 0  # 0 = flat, 1 = long A/short B, -1 = short A/long B
    for i in range(len(df)):
        z = zscore.iloc[i]
        if pd.isna(z):
            continue
        if position_state == 0:
            if z > entry_z:
                position_state = -1  # spread too high -> short A, long B
            elif z < -entry_z:
                position_state = 1  # spread too low -> long A, short B
        else:
            if abs(z) < exit_z:
                position_state = 0
        pos_a.iloc[i] = position_state
        pos_b.iloc[i] = -position_state

    result = df.copy()
    result["beta"] = beta_series
    result["spread"] = spread
    result["zscore"] = zscore
    result["pos_a"] = pos_a
    result["pos_b"] = pos_b
    return result
