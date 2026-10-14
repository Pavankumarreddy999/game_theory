"""
Phase 6: industry-specific momentum / mean-reversion in the true-value process.
phi (AR(1) of the diffusion shocks) is calibrated so the 5-day variance ratio matches
each industry's measured value; --scales multiplies phi (1 = real data, >1 = stress test).

USAGE (quick test): python phase6.py --industries alcohol,tech --seeds 3 --epochs 150 --solo_seeds 8
USAGE (full):       python phase6.py --industries alcohol,tech,auto,banking
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import lfilter
import simulate_market as sm
import simulate_evolution as se
from simulate_market import A, AGENTS, load_calibration

_orig_update = se.update
_orig_vp = sm.value_path


def variance_ratio(phi, q=5):
    k = np.arange(1, q)
    return 1 + 2 * np.sum((1 - k / q) * phi ** k)


def calibrate_phi(target):
    try:
        return brentq(lambda p: variance_ratio(p) - target, -0.9, 0.9)
    except ValueError:
        return 0.0


def make_path(phi):
    def f(n, cal, rng, drift=0.0):
        e = rng.normal(0, cal["sigma"] * np.sqrt(1 - phi ** 2), n)
        x = lfilter([1.0], [1.0, -phi], e)
        jumps = ((rng.random(n) < cal["jump_prob"]) * rng.choice([-1, 1], n)
                 * rng.exponential(cal["jump_size"], n))
        return np.log(100.0) + np.cumsum(drift / 252 + x + jumps)
    return f


def drop_bh_update(*a, **k):
    w = _orig_update(*a, **k)
    w[0] = 1e-9
    return w / w.sum()


def fmt(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f}±{x.std(ddof=1) / np.sqrt(len(x)):.3f}"


def solo_sharpe(cal, a, burn=100):
    out = []
    for s in range(a.solo_seeds):
        rets, _ = sm.simulate(cal, a.solo_days, 0.0, a.rho, a.cost_bps, seed=s)
        r = rets[burn:]
        out.append(r.mean(0) / (r.std(0) + 1e-12) * np.sqrt(252))
    return np.array(out)


def vol_of(p, burn, E):
    return np.std(np.diff(np.log(p[burn * E:]))) * np.sqrt(252)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--industries", default="alcohol,tech,auto,banking")
    ap.add_argument("--scales", default="0,1,5,10")
    ap.add_argument("--k", type=float, default=7.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--epoch_days", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--solo_seeds", type=int, default=20)
    ap.add_argument("--solo_days", type=int, default=2000)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(",")]
    burn = a.epochs // 3
    cal_tab = pd.read_csv("results/calibration_by_industry.csv", index_col="industry")
    solo_rows, evo_rows = [], []

    for ind in a.industries.split(","):
        cal = load_calibration(ind)
        phi0 = calibrate_phi(cal_tab.loc[ind, "variance_ratio_5d"])
        lam = a.k * cal["sigma"]
        for sc in scales:
            phi = float(np.clip(phi0 * sc, -0.9, 0.9))
            f = make_path(phi)
            sm.value_path = f
            se.value_path = f

            # 1) mechanism check: solo Sharpe with no price impact, all agents
            S = solo_sharpe(cal, a)
            solo_rows.append({"industry": ind, "scale": sc, "phi": round(phi, 3),
                              "VR5": round(variance_ratio(phi), 3),
                              "MA(mom)": fmt(S[:, 2]), "Turtle(mom)": fmt(S[:, 4]),
                              "RSI(rev)": fmt(S[:, 3]), "Kalman(rev)": fmt(S[:, 5])})

            # 2) evolution with Buy & Hold removed
            se.update = drop_bh_update
            base = np.mean([vol_of(se.run_one(cal, "replicator", 1.0, None, 0.0, a.rho, a.cost_bps,
                                              a.epochs, a.epoch_days, s)[1], burn, a.epoch_days)
                            for s in range(a.seeds)])
            for name, (mode, cap, beta) in se.CONFIGS.items():
                mom, rev, hhi, vol = [], [], [], []
                for s in range(a.seeds):
                    W, p = se.run_one(cal, mode, cap, beta, lam, a.rho, a.cost_bps,
                                      a.epochs, a.epoch_days, s)
                    fin = W[-10:].mean(0)
                    mom.append(fin[2] + fin[4])
                    rev.append(fin[3] + fin[5])
                    hhi.append((fin ** 2).sum())
                    vol.append(vol_of(p, burn, a.epoch_days))
                evo_rows.append({"industry": ind, "scale": sc, "phi": round(phi, 3), "rule": name,
                                 "momentum_share": fmt(mom), "reversion_share": fmt(rev),
                                 "amp": round(np.mean(vol) / base, 2), "concentration": fmt(hhi)})
            se.update = _orig_update
            print(f"done {ind} scale={sc} (phi={phi:.3f})", flush=True)

    sm.value_path = _orig_vp
    se.value_path = _orig_vp
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 1000)
    solo, evo = pd.DataFrame(solo_rows), pd.DataFrame(evo_rows)
    print("\n=== Mechanism check: solo Sharpe, no price impact "
          "(momentum should rise with phi, reversion should fall) ===")
    print(solo.to_string(index=False))
    print("\n=== Evolution (Buy & Hold out): capital in momentum vs reversion strategies ===")
    print(evo.to_string(index=False))
    solo.to_csv("results/phase6_solo.csv", index=False)
    evo.to_csv("results/phase6_evolution.csv", index=False)
    print("\nSaved results/phase6_solo.csv and results/phase6_evolution.csv")


if __name__ == "__main__":
    main()
    