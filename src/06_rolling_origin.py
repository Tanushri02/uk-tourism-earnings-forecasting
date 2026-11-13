"""Step 6: repeat the hold-out test many times (rolling origin) so one lucky or unlucky window cannot mislead us.

An "origin" is the last month of training data. For each origin we fit every model on data up to that month,
forecast the next `horizon` months, and score the forecasts against what actually happened. Then we average.

Three designs are run:
  A (main)      origins Feb 2010 ... Feb 2018, 24-month horizon -> 9 windows, which overlap
  B (robust 1)  origins Feb 2010, 2012, ... 2018, 24-month horizon -> 5 windows, no overlap
  C (robust 2)  origins Feb 2010 ... Feb 2019, 12-month horizon -> 10 windows, no overlap
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import (FIG_DIR, MAXITER, RESULTS_DIR, load_gmak, plt, scores, seasonal_naive,
                    seasonal_naive_drift)

y = load_gmak()
not_converged = []        # remembers any SARIMA fit that did not settle


def fit_sarima(data, order, seasonal_order):
    fit = sm.tsa.SARIMAX(data, order=order, seasonal_order=seasonal_order).fit(disp=False, maxiter=MAXITER)
    if not fit.mle_retvals["converged"]:
        not_converged.append((order, seasonal_order))
    return fit


def sarima_airline_log(train, n):
    fit = fit_sarima(np.log(train), (0, 1, 1), (0, 1, 1, 12))
    return np.exp(fit.forecast(n))


def sarima_complex_raw(train, n):
    return fit_sarima(train, (2, 1, 2), (2, 1, 2, 12)).forecast(n)


def holt_winters(train, n):
    return ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit().forecast(n)


MODELS = {
    "Seasonal naive": seasonal_naive,
    "Seasonal naive + drift": seasonal_naive_drift,
    "SARIMA(0,1,1)(0,1,1)12 log": sarima_airline_log,
    "SARIMA(2,1,2)(2,1,2)12 raw": sarima_complex_raw,
    "Holt-Winters add/mul": holt_winters,
}


def rolling_origin(origin_years, horizon):
    """Score every model in every window. Returns one row per (window, model)."""
    rows = []
    for year in origin_years:
        end = pd.Timestamp(f"{year}-02-01")
        train = y.loc[:end]
        test = y.loc[end + pd.offsets.MonthBegin(1): end + pd.offsets.MonthBegin(horizon)]
        assert len(test) == horizon, "test window runs past the last pre-COVID month"
        for name, model in MODELS.items():
            row = scores(test, model(train, horizon))
            row.update(model=name, origin=year)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(results):
    """Average scores per model, plus how many windows each model won (lowest RMSE)."""
    table = results.groupby("model")[["MAE", "RMSE", "MAPE_%"]].mean()
    wins = results.loc[results.groupby("origin")["RMSE"].idxmin(), "model"].value_counts()
    table["windows_won"] = wins.reindex(table.index).fillna(0).astype(int)
    return table.round(2).sort_values("RMSE")


designs = {
    "A: 9 overlapping windows, 24 months": rolling_origin(range(2010, 2019), 24),
    "B: 5 non-overlapping windows, 24 months": rolling_origin([2010, 2012, 2014, 2016, 2018], 24),
    "C: 10 non-overlapping windows, 12 months": rolling_origin(range(2010, 2020), 12),
}

summaries = []
for name, res in designs.items():
    s = summarise(res)
    print(f"\nDesign {name} (average over windows, best first):")
    print(s)
    summaries.append(s.assign(design=name))
pd.concat(summaries).to_csv(RESULTS_DIR / "06_rolling_summary.csv")

main = designs["A: 9 overlapping windows, 24 months"]
main.to_csv(RESULTS_DIR / "06_rolling_windows_A.csv", index=False)
print(f"\nSARIMA fits that did not fully converge: {len(not_converged)}")

# Chart: error in each window, for the main design
pivot = main.pivot(index="origin", columns="model", values="MAPE_%")
fig, ax = plt.subplots(figsize=(10, 5))
pivot.plot(ax=ax, marker="o")
ax.set_xlabel("Forecast origin (training ends February of this year)")
ax.set_ylabel("MAPE over the next 24 months (%)")
ax.set_title("GMAK: forecast error in each of the nine rolling windows")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_rolling_origin.png", dpi=150)
print("Saved results/06_rolling_summary.csv, results/06_rolling_windows_A.csv, figures/06_rolling_origin.png")
