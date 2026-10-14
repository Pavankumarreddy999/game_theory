"""
Regime-Switching Strategy using a Hidden Markov Model (HMM).

Fits a 2-state Gaussian HMM on daily returns:
  State 0 and State 1 each have their own mean and volatility.
  hmmlearn's fit() runs the Baum-Welch algorithm automatically.
  hmmlearn's predict() runs the Viterbi algorithm to infer the most likely
  state on each day.

Trading rule (kept simple and explainable):
  - Identify which state has the LOWER volatility -> call it "calm"
  - Identify which state has the HIGHER volatility -> call it "volatile"
  - In the "calm" state: trend-follow (buy if recent return positive, else sell)
  - In the "volatile" state: stay flat (sit out -- avoid whipsaw/choppy losses)
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_regime_model(returns: pd.Series, n_states: int = 2, seed: int = 42):
    X = returns.dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                         n_iter=1000, random_state=seed)
    model.fit(X)
    states = model.predict(X)
    return model, states


def regime_switching_signal(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.Series:
    prices = df[price_col]
    returns = prices.pct_change()

    valid_idx = returns.dropna().index
    model, states = fit_regime_model(returns.loc[valid_idx])

    # Identify which fitted state has the higher volatility ("volatile" regime)
    variances = model.covars_.flatten()
    volatile_state = int(np.argmax(variances))

    state_series = pd.Series(states, index=valid_idx)

    signal = pd.Series(0, index=df.index)
    trend = np.sign(returns.loc[valid_idx].rolling(5).mean().fillna(0))

    for date in valid_idx:
        if state_series.loc[date] == volatile_state:
            signal.loc[date] = 0  # choppy/volatile -> stay flat
        else:
            signal.loc[date] = trend.loc[date]  # calm -> trend-follow

    return signal
