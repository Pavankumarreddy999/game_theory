"""
Sweep crowding strength (lam) x friction rule, with and without Buy & Hold.
Reports mean +- standard error across seeds.

USAGE:
    python sweep_evolution.py --industries alcohol,tech --seeds 10
"""
import argparse
import numpy as np
import pandas as pd
import simulate_evolution as se
from simulate_market import load_calibration

_orig_update = se.update


def make_update(drop_bh):
    def f(*a, **k):
        w = _orig_update(*a, **k)
        if drop_bh:                      # remove the passive benchmark from the competition
            w[0] = 1e-9
            w = w / w.sum()
        return w
    return f


def fmt(x):
    return f"{np.mean(x):.3f}±{np.std(x, ddof=1) / np.sqrt(len(x)):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industries", default="alcohol,tech")
    ap.add_argument("--lams", default="0,0.05,0.1,0.2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--epoch_days", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    burn = args.epochs // 3
    rows = []
    for ind in args.industries.split(","):
        cal = load_calibration(ind)
        for drop in [False, True]:
            se.update = make_update(drop)
            for lam in lams:
                for name, (mode, cap, beta) in se.CONFIGS.items():
                    hhi, churn, vol = [], [], []
                    for s in range(args.seeds):
                        W, p = se.run_one(cal, mode, cap, beta, lam, args.rho, args.cost_bps,
                                          args.epochs, args.epoch_days, s)
                        fin = W[-10:].mean(0)
                        hhi.append((fin ** 2).sum())          # 1/6=0.17 even, 1.0 = one winner
                        churn.append(np.abs(np.diff(W[burn:], axis=0)).sum(1).mean() / 2)
                        vol.append(np.std(np.diff(np.log(p[burn * args.epoch_days:]))) * np.sqrt(252))
                    rows.append({"industry": ind, "buy&hold": "out" if drop else "in",
                                 "lam": lam, "rule": name,
                                 "concentration": fmt(hhi), "churn": fmt(churn), "mkt_vol": fmt(vol)})
                print(f"done {ind} bh={'out' if drop else 'in'} lam={lam}", flush=True)
    se.update = _orig_update
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(out.to_string(index=False))
    out.to_csv("results/phase3_sweep.csv", index=False)
    print("\nSaved results/phase3_sweep.csv")


if __name__ == "__main__":
    main()