"""
Phase 0: download 16 tickers, validate, and build the industry calibration table.
Raw CSVs -> data/raw/ (untouched). Calibration tables -> results/.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

START = "2015-01-01"
JUMP_SIGMAS = 3.0
ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RAW.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "alcohol": {"Diageo": "DEO", "AB_InBev": "BUD", "Pernod_Ricard": "RI.PA", "United_Spirits": "UNITDSPR.NS"},
    "tech": {"Apple": "AAPL", "Intel": "INTC", "Samsung": "005930.KS", "Sony": "SONY"},
    "auto": {"Tesla": "TSLA", "Toyota": "TM", "Tata_Motors": "TMPV.NS", "Ford": "F"},
    "banking": {"JPMorgan": "JPM", "BofA": "BAC", "HDFC_Bank": "HDFCBANK.NS", "ICICI_Bank": "ICICIBANK.NS"},
}


def download(ticker, tries=3):
    for attempt in range(tries):
        try:
            df = yf.Ticker(ticker).history(start=START, auto_adjust=False)
            if len(df) > 0:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df.index.name = "Date"
                return df
        except Exception as e:
            print(f"   {ticker} attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return None


def company_stats(adj):
    r = adj.pct_change().dropna()
    sd = r.std()
    jumps = r[r.abs() > JUMP_SIGMAS * sd]
    years = len(r) / 252
    r5 = adj.pct_change(5).dropna()
    return {
        "n_days": len(r),
        "ann_return": (1 + r).prod() ** (252 / len(r)) - 1,
        "ann_vol": sd * np.sqrt(252),
        "jumps_per_year": len(jumps) / years,
        "mean_jump_size": jumps.abs().mean() if len(jumps) else 0.0,
        "autocorr_lag1": r.autocorr(1),
        "variance_ratio_5d": r5.var() / (5 * r.var()),  # >1 momentum, <1 mean-reversion
    }


# ---- 1. Download and save raw ----
data, rows = {}, []
for industry, names in TICKERS.items():
    for name, tkr in names.items():
        print(f"Downloading {industry}/{name} ({tkr}) ...")
        df = download(tkr)
        if df is None:
            print(f"   FAILED: {tkr} -- skipping")
            continue
        df.to_csv(RAW / f"{industry}_{name}.csv")
        data[(industry, name)] = df
        rows.append({"industry": industry, "company": name, "ticker": tkr,
                     "rows": len(df), "first": df.index[0].date(),
                     "last": df.index[-1].date(),
                     "missing_adj_close": int(df["Adj Close"].isna().sum())})

val = pd.DataFrame(rows)
print("\n=== Data validation ===")
print(val.to_string(index=False))
val.to_csv(RES / "data_validation.csv", index=False)

if not data:
    raise SystemExit("No data downloaded -- check your internet connection.")

# ---- 2. Common period ----
common_start = max(d.index[0] for d in data.values())
common_end = min(d.index[-1] for d in data.values())
print(f"\nCommon period: {common_start.date()} -> {common_end.date()}")
if common_start > pd.Timestamp("2016-01-01"):
    latest = val.sort_values("first").iloc[-1]
    print(f"WARNING: {latest['company']} ({latest['ticker']}) starts late and shortens the window for everyone.")

# ---- 3. Calibration on the common period ----
cal = []
for (industry, name), df in data.items():
    adj = df["Adj Close"].loc[common_start:common_end].dropna()
    s = company_stats(adj)
    s.update({"industry": industry, "company": name})
    cal.append(s)

cal = pd.DataFrame(cal).set_index(["industry", "company"])
cal.to_csv(RES / "calibration_by_company.csv")

by_industry = cal.groupby("industry").mean(numeric_only=True)
by_industry.to_csv(RES / "calibration_by_industry.csv")

pd.set_option("display.width", 200)
print("\n=== Calibration by company ===")
print(cal.round(4))
print("\n=== Calibration by industry (the Phase 0 table) ===")
print(by_industry.round(4))
print("\nSaved: data/raw/*.csv, results/data_validation.csv, calibration_by_company.csv, calibration_by_industry.csv")
