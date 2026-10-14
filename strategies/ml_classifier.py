"""
Machine-Learning Direction Classifier.
Trains a Random Forest on simple technical features to predict whether
tomorrow's close will be higher than today's.

Features used (all computed from data available up to today -- no look-ahead):
  - 1-day return
  - 5-day rolling return
  - 5-day rolling volatility (std of returns)
  - Volume change

Label: 1 if next day's close > today's close, else 0.

IMPORTANT: uses a simple chronological train/test split (train on the first
70% of the data, test on the last 30%) to avoid overfitting / look-ahead bias.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["ret_1d"] = df[price_col].pct_change()
    feat["ret_5d"] = df[price_col].pct_change(5)
    feat["vol_5d"] = feat["ret_1d"].rolling(5).std()
    feat["vol_change"] = df["Volume"].pct_change()
    feat["label"] = (df[price_col].shift(-1) > df[price_col]).astype(int)
    return feat.dropna()


def ml_classifier_signal(df: pd.DataFrame, price_col: str = "Adj Close",
                          train_frac: float = 0.7, threshold: float = 0.5) -> pd.Series:
    feat = build_features(df, price_col)
    split = int(len(feat) * train_frac)
    train, test = feat.iloc[:split], feat.iloc[split:]

    feature_cols = ["ret_1d", "ret_5d", "vol_5d", "vol_change"]
    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(train[feature_cols], train["label"])

    probs = model.predict_proba(test[feature_cols])[:, 1]
    test_signal = pd.Series(np.where(probs > threshold, 1, 0), index=test.index)

    # No trading during the training period (model hasn't been evaluated there yet)
    signal = pd.Series(0, index=df.index)
    signal.loc[test_signal.index] = test_signal
    return signal
