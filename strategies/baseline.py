"""
Baseline strategies: Buy-and-Hold and Random Trader.
Both follow the common interface: generate_signal(df) -> pd.Series of positions.
Position values: 1 = long (holding), 0 = flat, -1 = short.
"""

import numpy as np
import pandas as pd


def buy_and_hold_signal(df: pd.DataFrame) -> pd.Series:
    """Always long, from the very first day."""
    return pd.Series(1, index=df.index)


def random_trader_signal(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Randomly long or flat each day (50/50)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([1, 0], size=len(df)), index=df.index)
