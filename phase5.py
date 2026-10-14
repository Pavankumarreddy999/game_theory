"""
Phase 5: (1) ESS / invasion test, (2) beta x stickiness-cap sweeps.
lam = k * daily sigma, so impact is the same *relative* size in every industry.

USAGE (quick test):  python phase5.py --industry alcohol --seeds 3 --epochs 150
USAGE (full):        python phase5.py --industry alcohol --seeds 8 --epochs 300 --bh out
                     python phase5.py --industry tech
"""
import argparse
import numpy as np
import pandas as pd
import simulate_evolution as se
from simulate_market import Agents, A, AGENTS, load_calibration, value_path

_orig_update = se.update


def make_update(drop_bh):
    def f(*a, **k):
        w = _orig_update(*a, **k)
        if drop_bh:
            w[0] = 1e-9
            w = w / w.sum()
        return w
    return f


def run_one_w(cal, mode, cap, beta, lam, rho, cost_bps, epochs, E, seed, w0):
    """Same as se.run_one but starts from capital shares w0 (used for invasion)."""
    rng = np.random.default_rng(seed)
    n = epochs * E
    logV = value_path(n, cal, rng)
    agents = Agents(np.random.default_rng(seed + 10_000))
    wrng = np.random.default_rng(seed + 20_000)
    w = w0 / w0.sum()
    p = np.empty(n)
    p[0] = np.exp(logV[0])
    pos_prev, dev = np.zeros(A), 0.0
    W = np.zeros((epochs, A))
    ep_ret = np.zeros(A)
    for t in range(1, n):
        pos = agents.decide(p[:t])
        flow = w @ (pos - pos_prev)
        dev = rho * dev + lam * flow
        p[t] = np.exp(logV[t] + dev)
        ep_ret += pos_prev * (p[t] / p[t - 1] - 1) - np.abs(pos - pos_prev) * cost_bps / 1e4
        pos_prev = pos
        if (t + 1) % E == 0:
            e = (t + 1) // E - 1
            W[e] = w
            w = se.update(w, ep_ret, mode, cap, beta, wrng)
            ep_ret[:] = 0
    return W, p


def vol_of(p, burn, E):
    return np.std(np.diff(np.log(p[burn * E:]))) * np.sqrt(252)


def mse(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def ess_test(cal, lam, a):
    rows = []
    for name, (mode, cap, beta) in se.CONFIGS.items():
        fin = [se.run_one(cal, mode, cap, beta, lam, a.rho, a.cost_bps, a.epochs, a.epoch_days, s)[0][-10:].mean(0)
               for s in range(a.seeds)]
        w_star = np.mean(fin, axis=0)
        w_star = w_star / w_star.sum()
        for i in range(A):
            if a.bh == "out" and i == 0:
                continue
            w0 = w_star.copy()
            w0[i] = 0.0
            w0 = w0 / w0.sum() * (1 - a.eps)
            w0[i] = a.eps
            end = []
            for s in range(a.seeds):
                W, _ = run_one_w(cal, mode, cap, beta, lam, a.rho, a.cost_bps, a.epochs,
                                 a.epoch_days, 1000 + s, w0)
                end.append(W[-10:, i].mean())
            m, se_ = mse(end)
            stable = abs(m - w_star[i]) <= max(0.03, 2 * se_)
            rows.append({"rule": name, "strategy": AGENTS[i], "eq_share": round(w_star[i], 3),
                         "start": a.eps, "end_share": f"{m:.3f}±{se_:.3f}",
                         "verdict": "stable" if stable else "drifts"})
        print(f"ESS done: {name}", flush=True)
    return pd.DataFrame(rows)


def dial_sweep(cal, lam, a):
    base = np.mean([vol_of(se.run_one(cal, "replicator", 1.0, None, 0.0, a.rho, a.cost_bps,
                                      a.epochs, a.epoch_days, s)[1], a.epochs // 3, a.epoch_days)
                    for s in range(a.seeds)])
    rows = []
    for cap in [0.005, 0.01, 0.02, 0.05, 1.0]:
        for beta in [0.5, 1.0, 2.0, 4.0, 8.0]:
            vol, hhi, churn = [], [], []
            for s in range(a.seeds):
                W, p = se.run_one(cal, "fermi", cap, beta, lam, a.rho, a.cost_bps,
                                  a.epochs, a.epoch_days, s)
                b = a.epochs // 3
                vol.append(vol_of(p, b, a.epoch_days))
                hhi.append((W[-10:].mean(0) ** 2).sum())
                churn.append(np.abs(np.diff(W[b:], axis=0)).sum(1).mean() / 2)
            rows.append({"cap": cap, "beta": beta, "amp": np.mean(vol) / base,
                         "concentration": np.mean(hhi), "churn": np.mean(churn)})
        print(f"dials done: cap={cap}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default="alcohol")
    ap.add_argument("--k", type=float, default=7.0)
    ap.add_argument("--bh", default="out", help="in | out")
    ap.add_argument("--part", default="both", help="ess | dials | both")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--epoch_days", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    a = ap.parse_args()

    cal = load_calibration(a.industry)
    lam = a.k * cal["sigma"]
    se.update = make_update(a.bh == "out")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(f"{a.industry}: k={a.k} -> lam={lam:.3f}, Buy&Hold {a.bh}")

    if a.part in ("ess", "both"):
        ess = ess_test(cal, lam, a)
        print("\n=== ESS / invasion test (eps invader; 'stable' = returns to its equilibrium share) ===")
        print(ess.to_string(index=False))
        ess.to_csv(f"results/phase5_ess_{a.industry}.csv", index=False)
    if a.part in ("dials", "both"):
        d = dial_sweep(cal, lam, a)
        for col in ["amp", "concentration", "churn"]:
            print(f"\n=== {col} (rows = stickiness cap, columns = noise beta) ===")
            print(d.pivot(index="cap", columns="beta", values=col).round(3))
        d.to_csv(f"results/phase5_dials_{a.industry}.csv", index=False)
    se.update = _orig_update


if __name__ == "__main__":
    main()