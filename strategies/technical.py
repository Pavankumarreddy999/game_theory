import numpy as np
import pandas as pd


def ma_crossover_signal(df, price_col="Adj Close", short=20, long=50):
    p = df[price_col]
    s, l = p.rolling(short).mean(), p.rolling(long).mean()
    sig = pd.Series(0, index=df.index)
    sig[s > l] = 1
    sig[s < l] = -1
    return sig


def compute_rsi(prices, window=14):
    d = prices.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def rsi_signal(df, price_col="Adj Close", window=14, low=30, high=70, exit_level=50):
    rsi = compute_rsi(df[price_col], window).values
    state, out = 0, []
    for v in rsi:
        if np.isnan(v):
            out.append(0)
            continue
        if state == 0:
            if v < low:
                state = 1
            elif v > high:
                state = -1
        elif state == 1 and v > exit_level:
            state = 0
        elif state == -1 and v < exit_level:
            state = 0
        out.append(state)
    return pd.Series(out, index=df.index)


def turtle_breakout_signal(df, price_col="Adj Close", entry=20, exit_=10):
    p = df[price_col]
    eh = p.rolling(entry).max().shift(1).values
    el = p.rolling(entry).min().shift(1).values
    xh = p.rolling(exit_).max().shift(1).values
    xl = p.rolling(exit_).min().shift(1).values
    state, out = 0, []
    for i, x in enumerate(p.values):
        if state == 0:
            if x > eh[i]:
                state = 1
            elif x < el[i]:
                state = -1
        elif state == 1 and x < xl[i]:
            state = 0
        elif state == -1 and x > xh[i]:
            state = 0
        out.append(state)
    return pd.Series(out, index=df.index)
