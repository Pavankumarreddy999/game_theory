"""
Builds report figures/tables from results/*.csv into results/figures/.
USAGE: python make_figures.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results", "results/figures"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
RULES = ["classic", "sticky (2%)", "noisy (Fermi)", "sticky+noisy"]


def num(s):
    """'0.123±0.004' -> 0.123"""
    return float(str(s).split("±")[0])


def have(name):
    ok = os.path.exists(f"{R}/{name}")
    if not ok:
        print(f"skip: {R}/{name} not found")
    return ok


# ---- Table 1: calibration ----
if have("calibration_by_industry.csv"):
    c = pd.read_csv(f"{R}/calibration_by_industry.csv", index_col="industry")
    t = c[["ann_vol", "jumps_per_year", "mean_jump_size", "autocorr_lag1", "variance_ratio_5d"]].round(3)
    t.to_csv(f"{F}/table1_calibration.csv")
    print("\nTable 1: industry calibration\n", t)

# ---- Fig 1: the twist (solo vs interactive) ----
if have("tournament_leaderboard.csv") and have("phase2_alcohol.csv"):
    solo = pd.read_csv(f"{R}/tournament_leaderboard.csv", index_col=0)["sharpe"].sort_values()
    sim = pd.read_csv(f"{R}/phase2_alcohol.csv", index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(solo.index, solo.values, color=["#c0392b" if v < 0 else "#27ae60" for v in solo.values])
    ax[0].set_title("A. Solo backtest (real Diageo data, out-of-sample)")
    ax[0].set_xlabel("Sharpe ratio")
    lam_cols = sorted(sim.columns, key=lambda s: float(s.split("_")[-1]))
    for name in sim.index:
        ax[1].plot([float(c.split("_")[-1]) for c in lam_cols], sim.loc[name, lam_cols].values,
                   marker="o", label=name)
    ax[1].set_title("B. Same strategies trading together (simulated)")
    ax[1].set_xlabel("price impact lam (0 = no interaction)")
    ax[1].set_ylabel("Sharpe ratio")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{F}/fig1_twist.png", dpi=150)
    plt.close()
    print("saved fig1_twist.png")

# ---- Fig 2: crowding amplification vs k ----
if have("phase4_sweep.csv"):
    d = pd.read_csv(f"{R}/phase4_sweep.csv")
    d = d[d["bh"] == "out"]
    inds = list(d["industry"].unique())
    fig, ax = plt.subplots(1, len(inds), figsize=(4.2 * len(inds), 4), sharey=True, squeeze=False)
    for a_, ind in zip(ax[0], inds):
        for rule in RULES:
            s = d[(d["industry"] == ind) & (d["rule"] == rule)].sort_values("k")
            a_.plot(s["k"], s["amp"], marker="o", label=rule)
        a_.set_title(ind)
        a_.set_xlabel("crowding strength k (lam = k x daily sigma)")
    ax[0][0].set_ylabel("price-vol amplification (1 = none)")
    ax[0][0].legend(fontsize=8)
    plt.suptitle("Crowding amplifies price volatility; stickiness damps it (Buy & Hold excluded)")
    plt.tight_layout()
    plt.savefig(f"{F}/fig2_amplification.png", dpi=150)
    plt.close()
    print("saved fig2_amplification.png")

# ---- Fig 3: stickiness cap x noise beta heatmaps ----
files = sorted(glob.glob(f"{R}/phase5_dials_*.csv"))
if files:
    fig, ax = plt.subplots(1, len(files), figsize=(5 * len(files), 4), squeeze=False)
    vmin = min(pd.read_csv(f)["amp"].min() for f in files)
    vmax = max(pd.read_csv(f)["amp"].max() for f in files)
    for a_, f in zip(ax[0], files):
        ind = os.path.basename(f).replace("phase5_dials_", "").replace(".csv", "")
        p = pd.read_csv(f).pivot(index="cap", columns="beta", values="amp").sort_index()
        im = a_.imshow(p.values, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        a_.set_xticks(range(len(p.columns)))
        a_.set_xticklabels(p.columns)
        a_.set_yticks(range(len(p.index)))
        a_.set_yticklabels([f"{x:.3f}" for x in p.index])
        a_.set_xlabel("noise beta (small = noisier)")
        a_.set_ylabel("stickiness cap (share of capital per epoch)")
        a_.set_title(ind)
        a_.grid(False)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                a_.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax[0].tolist(), label="price-vol amplification")
    plt.savefig(f"{F}/fig3_dials.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved fig3_dials.png")

# ---- Fig 4: does industry character matter? (phase 6) ----
if have("phase6_evolution.csv"):
    e = pd.read_csv(f"{R}/phase6_evolution.csv")
    e["rev"] = e["reversion_share"].map(num)
    e["conc"] = e["concentration"].map(num)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for ind in e["industry"].unique():
        for k_, (rule, a_) in enumerate([("classic", ax[0]), ("sticky (2%)", ax[1]), ("noisy (Fermi)", ax[2])]):
            s = e[(e["industry"] == ind) & (e["rule"] == rule)].sort_values("scale")
            a_.plot(s["scale"], s["rev"], marker="o", label=ind)
            a_.set_title(f"capital in RSI+Kalman, {rule}")
            a_.set_xlabel("momentum/reversion stress scale (1 = real data)")
            a_.set_ylim(0, 1.05)
    ax[0].set_ylabel("final capital share")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(f"{F}/fig4_industry_character.png", dpi=150)
    plt.close()
    print("saved fig4_industry_character.png")

    t2 = e[(e["scale"] == 1) & (e["rule"].isin(RULES))][["industry", "rule", "momentum_share",
                                                          "reversion_share", "amp", "concentration"]]
    t2.to_csv(f"{F}/table2_real_calibration_results.csv", index=False)
    print("\nTable 2: results at real-data calibration (scale = 1)\n", t2.to_string(index=False))

print(f"\nDone. Files in {F}/")