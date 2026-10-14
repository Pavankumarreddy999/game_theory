"""
Phase 4: fair cross-industry comparison + cycle detection.
 - lam = k * daily diffusion sigma of that industry (same *relative* impact everywhere)
 - cycle detectors on the capital-share series:
     peak_ratio : strongest spectral peak (periods 4..N/4 epochs) / mean power in that band
                  (white noise gives roughly 4-6; much higher = a real cycle)
     period     : period in epochs of that peak
     switches   : changes of the leading strategy per 100 epochs
USAGE:
    python sweep_v2.py --industries alcohol,tech,auto,banking --seeds 8
    python sweep_v2.py --industries alcohol,tech --seeds 3 --epochs 150   (quick test)
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
        if drop_bh:
            w[0] = 1e-9
            w = w / w.sum()
        return w
    return f


def cycle_stats(W, burn):
    X = W[burn:]
    N = len(X)
    t = np.arange(N)
    X = X - X.mean(0)
    coef = np.polyfit(t, X, 1)                       # remove linear drift per strategy
    X = X - (np.outer(t, coef[0]) + coef[1])
    P = (np.abs(np.fft.rfft(X, axis=0)) ** 2).sum(1)
    ks = np.arange(len(P))
    band = (ks >= 4) & (ks <= N // 4)                # periods from N/4 down to 4 epochs
    pb, kb = P[band], ks[band]
    peak_ratio = pb.max() / (pb.mean() + 1e-18)
    period = N / kb[pb.argmax()]
    leader = (X + 0).argmax(1)
    switches = (np.diff(leader) != 0).sum() / N * 100
    return peak_ratio, period, switches


def fmt(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f}±{x.std(ddof=1) / np.sqrt(len(x)):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industries", default="alcohol,tech,auto,banking")
    ap.add_argument("--ks", default="0,3,7,14", help="lam = k * daily sigma")
    ap.add_argument("--bh", default="out", help="in | out | both")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--epoch_days", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    args = ap.parse_args()
    ks = [float(x) for x in args.ks.split(",")]
    burn = args.epochs // 3
    bh_modes = [False, True] if args.bh == "both" else [args.bh == "out"]
    rows, base = [], {}
    for ind in args.industries.split(","):
        cal = load_calibration(ind)
        for drop in bh_modes:
            se.update = make_update(drop)
            for k in ks:
                lam = k * cal["sigma"]
                for name, (mode, cap, beta) in se.CONFIGS.items():
                    vol, hhi, pk, per, sw = [], [], [], [], []
                    for s in range(args.seeds):
                        W, p = se.run_one(cal, mode, cap, beta, lam, args.rho, args.cost_bps,
                                          args.epochs, args.epoch_days, s)
                        vol.append(np.std(np.diff(np.log(p[burn * args.epoch_days:]))) * np.sqrt(252))
                        hhi.append((W[-10:].mean(0) ** 2).sum())
                        a, b, c = cycle_stats(W, burn)
                        pk.append(a); per.append(b); sw.append(c)
                    key = (ind, drop, name)
                    if k == ks[0]:
                        base[key] = np.mean(vol)
                    rows.append({"industry": ind, "bh": "out" if drop else "in", "k": k,
                                 "lam": round(lam, 3), "rule": name,
                                 "mkt_vol": fmt(vol), "amp": round(np.mean(vol) / base[key], 2),
                                 "concentration": fmt(hhi), "peak_ratio": fmt(pk),
                                 "period": round(float(np.median(per)), 1),
                                 "switches/100ep": fmt(sw)})
                print(f"done {ind} bh={'out' if drop else 'in'} k={k}", flush=True)
    se.update = _orig_update
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 1000)
    print(out.to_string(index=False))
    out.to_csv("results/phase4_sweep.csv", index=False)
    print("\nSaved results/phase4_sweep.csv")


if __name__ == "__main__":
    main()
    