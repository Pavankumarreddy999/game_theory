"""
Kalman Filter Trend Estimator.
Manual implementation of the standard scalar Kalman filter (no external
Kalman library needed) exactly matching the equations in the strategy guide:

Prediction:  x_hat_minus = x_hat_prev ;  P_minus = P_prev + Q
Update:      K = P_minus / (P_minus + R)
             x_hat = x_hat_minus + K * (y - x_hat_minus)
             P = (1 - K) * P_minus

Trading rule: buy (long) when price is meaningfully BELOW the filter's fair
value estimate, sell/short when meaningfully ABOVE.
"""

import numpy as np
import pandas as pd


def kalman_filter_estimate(prices: pd.Series, Q: float = 1e-5, R: float = 1e-2) -> pd.Series:
    n = len(prices)
    x_hat = np.zeros(n)
    P = np.zeros(n)

    x_hat[0] = prices.iloc[0]
    P[0] = 1.0

    for t in range(1, n):
        # Prediction step
        x_hat_minus = x_hat[t - 1]
        P_minus = P[t - 1] + Q

        # Update step
        K = P_minus / (P_minus + R)
        x_hat[t] = x_hat_minus + K * (prices.iloc[t] - x_hat_minus)
        P[t] = (1 - K) * P_minus

    return pd.Series(x_hat, index=prices.index)


def kalman_trend_signal(df: pd.DataFrame, price_col: str = "Adj Close",
                         entry_threshold: float = 0.01) -> pd.Series:
    """
    entry_threshold: fraction (e.g. 0.01 = 1%) price must deviate from the
    filter's fair-value estimate before taking a position.
    """
    prices = df[price_col]
    fair_value = kalman_filter_estimate(prices)
    deviation = (prices - fair_value) / fair_value

    signal = pd.Series(0, index=df.index)
    signal[deviation < -entry_threshold] = 1   # price below fair value -> buy
    signal[deviation > entry_threshold] = -1   # price above fair value -> sell/short
    return signal
