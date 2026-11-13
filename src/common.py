"""Shared helpers used by every step: loading the data, splitting it, scoring forecasts.

GMAK = ONS series "Overseas visits to the UK: earnings", in GBP millions per month,
not seasonally adjusted, in current (nominal) prices.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "gmak.xlsx"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

LAST_PRE_COVID = "2020-02-01"   # we stop here: travel collapsed from March 2020
N_TEST = 24                     # hold out the last 24 pre-COVID months for testing
MAXITER = 500                   # attempts the SARIMA optimiser gets to settle (the default, 50, is too few for bigger models)


def load_gmak(pre_covid_only=True):
    """Return GMAK as a monthly pandas Series indexed by date."""
    df = pd.read_excel(DATA_FILE, sheet_name="GMAK")
    y = df.set_index("Date")["GMAK"].astype(float)
    y.index = pd.DatetimeIndex(y.index, freq="MS")      # "MS" = month-start frequency
    if pre_covid_only:
        y = y.loc[:LAST_PRE_COVID]
    return y


def train_test_split(y, n_test=N_TEST):
    """Everything except the last n_test months is training; the last n_test months are the test."""
    return y.iloc[:-n_test], y.iloc[-n_test:]


def scores(actual, forecast):
    """Three standard forecast errors. Lower is better for all three."""
    err = np.asarray(actual) - np.asarray(forecast)
    return {
        "MAE": float(np.mean(np.abs(err))),                              # average miss, in GBP million
        "RMSE": float(np.sqrt(np.mean(err ** 2))),                       # like MAE but punishes big misses more
        "MAPE_%": float(np.mean(np.abs(err / np.asarray(actual))) * 100),  # average miss as a % of actual
    }


def seasonal_naive(train, n_ahead):
    """Baseline: predict each future month with the same month one year earlier."""
    last_year = train.iloc[-12:].to_numpy()
    values = np.tile(last_year, n_ahead // 12 + 1)[:n_ahead]
    index = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=n_ahead, freq="MS")
    return pd.Series(values, index=index)


def seasonal_naive_drift(train, n_ahead):
    """Baseline 2: seasonal naive, plus the average yearly growth added once for each year ahead."""
    base = seasonal_naive(train, n_ahead)
    drift = (train - train.shift(12)).dropna().mean()   # average amount a month beat the same month a year earlier
    years_ahead = np.arange(n_ahead) // 12 + 1          # 1 for months 1-12, 2 for months 13-24, and so on
    return base + drift * years_ahead
