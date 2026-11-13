"""Step 1: a fair test. Fit on data up to Feb 2018, forecast the next 24 months, compare with what really happened.

Models compared:
  - seasonal naive (baseline: "same month last year")
  - seasonal naive + drift (tougher baseline: adds the average yearly growth)
  - SARIMA(0,1,1)(0,1,1)12 on raw data    (the classic "airline" model)
  - SARIMA(0,1,1)(0,1,1)12 on log data    (same model after taking logs)
  - SARIMA(2,1,2)(2,1,2)12 on raw data    (the more complex model chosen in the original coursework)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (FIG_DIR, MAXITER, RESULTS_DIR, load_gmak, plt, scores, seasonal_naive,
                    seasonal_naive_drift, train_test_split)

y = load_gmak()
train, test = train_test_split(y)
print(f"Train: {train.index[0]:%b %Y} to {train.index[-1]:%b %Y} ({len(train)} months)")
print(f"Test : {test.index[0]:%b %Y} to {test.index[-1]:%b %Y} ({len(test)} months)")


def sarima_forecast(series, order, seasonal_order, n_ahead, use_log=False):
    """Fit a SARIMA model and forecast n_ahead months. If use_log, model log(series) and convert back."""
    data = np.log(series) if use_log else series
    fit = sm.tsa.SARIMAX(data, order=order, seasonal_order=seasonal_order).fit(disp=False, maxiter=MAXITER)
    if not fit.mle_retvals["converged"]:
        print(f"  Note: SARIMA{order}{seasonal_order} did not fully converge, so its settings may not be optimal")
    fc = fit.forecast(n_ahead)
    return np.exp(fc) if use_log else fc


forecasts = {
    "Seasonal naive": seasonal_naive(train, len(test)),
    "Seasonal naive + drift": seasonal_naive_drift(train, len(test)),
    "SARIMA(0,1,1)(0,1,1)12 raw": sarima_forecast(train, (0, 1, 1), (0, 1, 1, 12), len(test)),
    "SARIMA(0,1,1)(0,1,1)12 log": sarima_forecast(train, (0, 1, 1), (0, 1, 1, 12), len(test), use_log=True),
    "SARIMA(2,1,2)(2,1,2)12 raw": sarima_forecast(train, (2, 1, 2), (2, 1, 2, 12), len(test)),
}

table = pd.DataFrame({name: scores(test, fc) for name, fc in forecasts.items()}).T.round(2)
print("\nHold-out scores (lower is better):")
print(table.sort_values("RMSE"))
table.to_csv(RESULTS_DIR / "01_holdout_scores.csv")

# Chart: last 5 years of training, then the test period with every forecast
fig, ax = plt.subplots(figsize=(11, 5))
train.iloc[-60:].plot(ax=ax, color="grey", label="Training data (last 5 years)")
test.plot(ax=ax, color="black", linewidth=2, label="Actual (test)")
for name, fc in forecasts.items():
    fc.plot(ax=ax, label=name, alpha=0.8)
ax.set_title("GMAK: 24-month hold-out forecasts vs actual")
ax.set_ylabel("Earnings from overseas visits (GBP million)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "01_holdout_forecasts.png", dpi=150)
print("\nSaved figures/01_holdout_forecasts.png and results/01_holdout_scores.csv")
