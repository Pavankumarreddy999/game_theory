"""
Phase 3: capital reallocation on top of the Phase 2 market.
Each epoch: run E days with current capital shares w, score strategies,
then reallocate capital by one of four rules (classic / sticky / noisy / both).

USAGE:
    python simulate_evolution.py --industry alcohol
    python simulate_evolution.py --industry all --seeds 20
"""
import argparse
import numpy as np
import pandas as pd
from simulate_market import AGENTS, A, Agents, load_calibration, value_path

CONFIGS = {
    "classic":       ("replicator", 1.00, None),
    "sticky (2%)":   ("replicator", 0.02, None),
    "noisy (Fermi)": ("fermi",      1.00, 2.0),
    "sticky+noisy":  ("fermi",      0.02, 2.0),
}


def update(w, f, mode, cap, beta, rng, eta=0.5, frac=0.25, floor=1e-3):
    z = (f - f.mean()) / (f.std() + 1e-12)          # standardize payoffs
    delta = np.zeros(A)
    if mode == "replicator":
        delta = eta * w * (z - w @ z)               # grow if above average payoff
    else:                                           # Fermi / logit pairwise switching
        for i in range(A):
            j = rng.choice([k for k in range(A) if k != i])
            prob = 1 / (1 + np.exp(-np.clip(beta * (z[j] - z[i]), -50, 50)))
            if rng.random() < prob:
                amt = frac * w[i]
                delta[i] -= amt
                delta[j] += amt
    moved = np.abs(delta).sum() / 2                 # stickiness: cap total capital moved
    if moved > cap:
        delta *= cap / moved
    new = np.maximum(w + delta, floor)
    return new / new.sum()


def run_one(cal, mode, cap, beta, lam, rho, cost_bps, epochs, E, seed):
    rng = np.random.default_rng(seed)
    n = epochs * E
    logV = value_path(n, cal, rng)
    agents = Agents(np.random.default_rng(seed + 10_000))
    wrng = np.random.default_rng(seed + 20_000)
    w = np.full(A, 1.0 / A)
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
            w = update(w, ep_ret, mode, cap, beta, wrng)
            ep_ret[:] = 0
    return W, p


def run(industry, lam, rho, cost_bps, epochs, E, seeds):
    cal = load_calibration(industry)
    burn = epochs // 3
    rows = []
    for name, (mode, cap, beta) in CONFIGS.items():
        fin, wob, vol = [], [], []
        for s in range(seeds):
            W, p = run_one(cal, mode, cap, beta, lam, rho, cost_bps, epochs, E, s)
            fin.append(W[-10:].mean(0))
            wob.append(np.abs(np.diff(W[burn:], axis=0)).sum(1).mean() / 2)        # >0 and large = cycling/instability
            vol.append(np.std(np.diff(np.log(p[burn * E:]))) * np.sqrt(252))
        row = dict(zip(AGENTS, np.mean(fin, axis=0)))
        row["wobble"] = np.mean(wob)
        row["mkt_vol"] = np.mean(vol)
        rows.append(pd.Series(row, name=name))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default="alcohol")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--epoch_days", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    inds = ["alcohol", "tech", "auto", "banking"] if args.industry == "all" else [args.industry]
    pd.set_option("display.width", 220)
    for ind in inds:
        out = run(ind, args.lam, args.rho, args.cost_bps, args.epochs, args.epoch_days, args.seeds)
        print(f"\n===== {ind.upper()}  lam={args.lam}, {args.seeds} seeds, "
              f"{args.epochs} epochs x {args.epoch_days} days =====")
        print("Final capital share per strategy | wobble = std of shares over time | mkt_vol = annualized price vol")
        print(out.round(3))
        out.to_csv(f"results/phase3_{ind}.csv")
    print("\nSaved results/phase3_<industry>.csv")


if __name__ == "__main__":
    main()
