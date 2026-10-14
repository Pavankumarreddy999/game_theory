import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategies.baseline import buy_and_hold_signal, random_trader_signal
from strategies.kalman_trend import kalman_trend_signal
from strategies.technical import ma_crossover_signal, rsi_signal, turtle_breakout_signal
from strategies.pairs_trading import generate_pairs_signals
from backtest import backtest_signal, performance_metrics


def make_synthetic_data(n=1000, seed=1, drift=0.0003, vol=0.02):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    df = pd.DataFrame({"Date": dates, "Open": prices, "High": prices, "Low": prices,
                       "Close": prices, "Adj Close": prices,
                       "Volume": rng.integers(1_000_000, 5_000_000, n)})
    return df.set_index("Date")


def load_data(path):
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date")


def evaluate(df, signals, pairs_df, start, cost_bps):
    rows, curves = {}, {}
    for name, sig in signals.items():
        r = backtest_signal(df, sig, cost_bps=cost_bps)["strategy_returns"].loc[start:]
        rows[name] = performance_metrics(r)
        curves[name] = (1 + r.fillna(0)).cumprod()

    ra, rb = pairs_df["a"].pct_change(), pairs_df["b"].pct_change()
    pa, pb = pairs_df["pos_a"].shift(1).fillna(0), pairs_df["pos_b"].shift(1).fillna(0)
    turnover = pa.diff().abs().fillna(pa.abs()) + pb.diff().abs().fillna(pb.abs())
    r = (pa * ra + pb * rb - turnover * cost_bps / 10_000).loc[start:]
    rows["Pairs Trading"] = performance_metrics(r)
    curves["Pairs Trading"] = (1 + r.fillna(0)).cumprod()
    return pd.DataFrame(rows).T, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_a", default=None)
    ap.add_argument("--file_b", default=None)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--oos_frac", type=float, default=0.7)
    args = ap.parse_args()

    df_a = load_data(args.file_a) if args.file_a else make_synthetic_data(seed=1)
    df_b = load_data(args.file_b) if args.file_b else make_synthetic_data(seed=2)
    print("Asset A:", args.file_a or "SYNTHETIC", "| Asset B:", args.file_b or "SYNTHETIC")

    start = df_a.index[int(len(df_a) * args.oos_frac)]
    print(f"Evaluation window: {start.date()} -> {df_a.index[-1].date()} "
          f"(same for every strategy), cost = {args.cost_bps} bps per position change")

    signals = {
        "Buy & Hold": buy_and_hold_signal(df_a),
        "Random Trader": random_trader_signal(df_a),
        "MA Crossover": ma_crossover_signal(df_a),
        "RSI Mean-Reversion": rsi_signal(df_a),
        "Turtle Breakout": turtle_breakout_signal(df_a),
        "Kalman Trend": kalman_trend_signal(df_a),
    }
    pairs_df = generate_pairs_signals(df_a["Adj Close"], df_b["Adj Close"])

    board, curves = evaluate(df_a, signals, pairs_df, start, args.cost_bps)
    board = board.sort_values("sharpe", ascending=False)
    print("\n=== Leaderboard (out-of-sample, net of costs) ===")
    print(board.round(4))

    sens = {c: evaluate(df_a, signals, pairs_df, start, c)[0]["sharpe"] for c in [0, 5, 10, 20]}
    print("\n=== Sharpe vs transaction cost (bps) ===")
    print(pd.DataFrame(sens).round(3))

    plt.figure(figsize=(10, 6))
    for name, c in curves.items():
        plt.plot(c.index, c.values, label=name)
    plt.legend()
    plt.title("Solo backtest, out-of-sample window (net of costs)")
    plt.ylabel("Cumulative return (1.0 = start)")
    plt.tight_layout()
    plt.savefig("results/tournament_results.png", dpi=150)
    board.to_csv("results/tournament_leaderboard.csv")
    print("\nSaved results/tournament_results.png and results/tournament_leaderboard.csv")


if __name__ == "__main__":
    main()
