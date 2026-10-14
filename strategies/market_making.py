"""
Avellaneda-Stoikov Market Making (simplified for daily OHLCV data).

Since we don't have live order-book data, we approximate:
  - mid price 's' = Adj Close
  - volatility 'sigma' = rolling std of daily returns
  - the day's high-low range approximates whether our quotes would have been
    "filled" that day (a simplification, stated explicitly as a limitation)

Formulas (from the strategy guide):
  reservation price: r = s - q * gamma * sigma^2 * (T - t)
  optimal spread:    delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
  bid = r - delta/2
  ask = r + delta/2

Inventory q changes by +1 when the bid is filled (we bought) and -1 when the
ask is filled (we sold). P&L = spread earned on fills + mark-to-market change
on any inventory still held.
"""

import numpy as np
import pandas as pd


def avellaneda_stoikov_backtest(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    gamma: float = 0.1,
    k: float = 1.5,
    horizon_days: int = 1,
    vol_window: int = 20,
    max_inventory: int = 5,
):
    prices = df[price_col]
    returns = prices.pct_change()
    sigma = returns.rolling(vol_window).std()

    adj_factor = df[price_col] / df["Close"]  # put High/Low on the adjusted-price scale
    q = 0
    inventory = []
    cash = 0.0
    pnl = []

    for i in range(len(df)):
        s = prices.iloc[i]
        sig = sigma.iloc[i] if not np.isnan(sigma.iloc[i]) else 0.0
        T_minus_t = horizon_days  # simplistic: treat each day as its own horizon

        r = s - q * gamma * (sig ** 2) * T_minus_t
        delta = gamma * (sig ** 2) * T_minus_t + (2 / gamma) * np.log(1 + gamma / k)
        bid = r - delta / 2
        ask = r + delta / 2

        day_low = df["Low"].iloc[i] * adj_factor.iloc[i]
        day_high = df["High"].iloc[i] * adj_factor.iloc[i]

        # Simplified fill logic: if the day's range crosses our quote, assume a fill
        if day_low <= bid and q < max_inventory:
            q += 1
            cash -= bid
        if day_high >= ask and q > -max_inventory:
            q -= 1
            cash += ask

        mark_to_market = q * s
        pnl.append(cash + mark_to_market)
        inventory.append(q)

    result = pd.DataFrame({"pnl": pnl, "inventory": inventory}, index=df.index)
    return result
