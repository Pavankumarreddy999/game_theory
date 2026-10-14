"""
Phase 2: multi-agent market. All strategies trade together; their net order flow
moves the price. lam = price impact per unit of net flow (lam=0 -> solo/no impact).

USAGE:
    python simulate_market.py --industry alcohol
    python simulate_market.py --industry all --seeds 30
"""
import argparse
import numpy as np
import pandas as pd

AGENTS = ["Buy & Hold", "Random Trader", "MA Crossover",
          "RSI Mean-Reversion", "Turtle Breakout", "Kalman Trend"]
A = len(AGENTS)


def load_calibration(industry, path="results/calibration_by_industry.csv"):
    row = pd.read_csv(path, index_col="industry").loc[industry]
    vol, jumps, jsize = row["ann_vol"], row["jumps_per_year"], row["mean_jump_size"]
    diff_var = max(vol ** 2 - jumps * jsize ** 2, 0.5 * vol ** 2)
    return {"sigma": np.sqrt(diff_var / 252), "jump_prob": jumps / 252, "jump_size": jsize}


def value_path(n, cal, rng, drift=0.0):
    eps = rng.normal(0, cal["sigma"], n)
    jumps = ((rng.random(n) < cal["jump_prob"]) * rng.choice([-1, 1], n)
             * rng.exponential(cal["jump_size"], n))
    return np.log(100.0) + np.cumsum(drift / 252 + eps + jumps)


class Agents:
    """Incremental versions of the six strategies. decide(hist) sees prices up to now."""

    def __init__(self, rng):
        self.rng = rng
        self.ag = self.al = None
        self.rsi_state = 0
        self.turtle = 0
        self.kx = None
        self.kP = 1.0

    def decide(self, hist):
        n, y = len(hist), hist[-1]
        pos = np.zeros(A)
        pos[0] = 1.0                                   # Buy & Hold
        pos[1] = self.rng.choice([0, 1])               # Random Trader

        if n >= 50:                                    # MA Crossover
            s, l = hist[-20:].mean(), hist[-50:].mean()
            pos[2] = 1 if s > l else (-1 if s < l else 0)

        if n >= 2:                                     # RSI Mean-Reversion
            d = hist[-1] - hist[-2]
            g, ls = max(d, 0.0), max(-d, 0.0)
            if self.ag is None:
                self.ag, self.al = g, ls
            else:
                a = 1 / 14
                self.ag = (1 - a) * self.ag + a * g
                self.al = (1 - a) * self.al + a * ls
            rsi = (100 if self.ag > 0 else 50) if self.al == 0 else 100 - 100 / (1 + self.ag / self.al)
            if self.rsi_state == 0:
                self.rsi_state = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
            elif self.rsi_state == 1 and rsi > 50:
                self.rsi_state = 0
            elif self.rsi_state == -1 and rsi < 50:
                self.rsi_state = 0
            pos[3] = self.rsi_state

        if n >= 21:                                    # Turtle Breakout
            prev20, prev10 = hist[-21:-1], hist[-11:-1]
            if self.turtle == 0:
                self.turtle = 1 if y > prev20.max() else (-1 if y < prev20.min() else 0)
            elif self.turtle == 1 and y < prev10.min():
                self.turtle = 0
            elif self.turtle == -1 and y > prev10.max():
                self.turtle = 0
            pos[4] = self.turtle

        Q, R = 1e-5, 1e-2                              # Kalman Trend
        if self.kx is None:
            self.kx = y
        else:
            Pm = self.kP + Q
            K = Pm / (Pm + R)
            self.kx = self.kx + K * (y - self.kx)
            self.kP = (1 - K) * Pm
        dev = (y - self.kx) / self.kx
        pos[5] = 1 if dev < -0.01 else (-1 if dev > 0.01 else 0)
        return pos


def simulate(cal, n, lam, rho, cost_bps, seed, drift=0.0):
    rng = np.random.default_rng(seed)
    logV = value_path(n, cal, rng, drift)
    agents = Agents(np.random.default_rng(seed + 10_000))
    w = np.full(A, 1.0 / A)                            # equal capital shares
    p = np.empty(n)
    p[0] = np.exp(logV[0])
    rets = np.zeros((n, A))
    pos_prev, dev = np.zeros(A), 0.0
    for t in range(1, n):
        pos = agents.decide(p[:t])                     # decide on yesterday's price
        flow = w @ (pos - pos_prev)                    # net buying pressure today
        dev = rho * dev + lam * flow                   # impact decays, flow pushes price
        p[t] = np.exp(logV[t] + dev)
        rets[t] = pos_prev * (p[t] / p[t - 1] - 1) - np.abs(pos - pos_prev) * cost_bps / 1e4
        pos_prev = pos
    return rets, p


def run(industry, lams, days, seeds, rho, cost_bps, burn=100):
    cal = load_calibration(industry)
    sharpe = {lam: [] for lam in lams}
    mvol = {lam: [] for lam in lams}
    for lam in lams:
        for s in range(seeds):
            rets, p = simulate(cal, days, lam, rho, cost_bps, seed=s)
            r = rets[burn:]
            sharpe[lam].append(r.mean(0) / (r.std(0) + 1e-12) * np.sqrt(252))
            mvol[lam].append(np.std(np.diff(np.log(p[burn:]))) * np.sqrt(252))
    table = pd.DataFrame({lam: np.mean(sharpe[lam], axis=0) for lam in lams}, index=AGENTS)
    ranks = table.rank(ascending=False).astype(int)
    vols = pd.Series({lam: np.mean(mvol[lam]) for lam in lams})
    return table, ranks, vols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default="alcohol",
                    help="alcohol | tech | auto | banking | all")
    ap.add_argument("--lams", default="0,0.02,0.05,0.1")
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    industries = ["alcohol", "tech", "auto", "banking"] if args.industry == "all" else [args.industry]

    pd.set_option("display.width", 200)
    for ind in industries:
        table, ranks, vols = run(ind, lams, args.days, args.seeds, args.rho, args.cost_bps)
        print(f"\n===== {ind.upper()}  ({args.seeds} seeds x {args.days} days, rho={args.rho}) =====")
        print("Mean Sharpe by strategy (columns = lam; 0 = solo/no impact, higher = more crowding)")
        print(table.round(3))
        print("\nRank (1 = best)")
        print(ranks)
        print("\nMarket annualized volatility by lam")
        print(vols.round(4).to_string())
        out = table.copy()
        out.columns = [f"sharpe_lam_{c}" for c in out.columns]
        out.to_csv(f"results/phase2_{ind}.csv")
    print("\nSaved results/phase2_<industry>.csv")


if __name__ == "__main__":
    main()
