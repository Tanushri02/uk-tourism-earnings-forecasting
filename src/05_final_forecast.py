"""Step 5: refit on ALL pre-COVID data and forecast to December 2023, then compare with what actually happened.

The forecast answers: "what would earnings have looked like if the pre-2020 pattern had simply continued?"
We have real data up to Sep 2022, so we can measure the gap. This is an indicative gap, not a proven
effect of COVID: a forecast cannot account for other changes (exchange rates, Brexit, prices).
Both forecast and actuals are in current (nominal) prices.
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import FIG_DIR, MAXITER, RESULTS_DIR, load_gmak, plt

warnings.filterwarnings("ignore")

history = load_gmak()                               # Jan 1980 to Feb 2020
actual_all = load_gmak(pre_covid_only=False)         # Jan 1980 to Sep 2022 (includes COVID months)
horizon = len(pd.date_range("2020-03-01", "2023-12-01", freq="MS"))       # 46 months

with open(RESULTS_DIR / "best_sarima.json") as f:
    spec = json.load(f)

sarima_fit = sm.tsa.SARIMAX(np.log(history), order=tuple(spec["order"]),
                            seasonal_order=tuple(spec["seasonal_order"])).fit(disp=False, maxiter=MAXITER)
pred = sarima_fit.get_forecast(horizon)
sarima_fc = np.exp(pred.predicted_mean)
ci = np.exp(pred.conf_int(alpha=0.05))               # 95% interval, converted back from log scale
lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]

hw_fit = ExponentialSmoothing(history, trend="add", seasonal="mul", seasonal_periods=12).fit()
hw_fc = hw_fit.forecast(horizon)

out = pd.DataFrame({"sarima_forecast": sarima_fc, "sarima_lower95": lower, "sarima_upper95": upper,
                    "holt_winters_forecast": hw_fc})
out["actual"] = actual_all.reindex(out.index)         # blank after Sep 2022
out.round(0).to_csv(RESULTS_DIR / "05_final_forecast.csv")

# How far below the "no-COVID" forecast was actual spending, Mar 2020 to Sep 2022?
seen = out.dropna(subset=["actual"])
print(f"Months with real data after the forecast start: {len(seen)} ({seen.index[0]:%b %Y} to {seen.index[-1]:%b %Y})")
for col in ["sarima_forecast", "holt_winters_forecast"]:
    gap = seen[col].sum() - seen["actual"].sum()
    print(f"{col}: forecast {seen[col].sum():,.0f}, actual {seen['actual'].sum():,.0f} "
          f"-> shortfall about GBP {gap:,.0f} million ({gap / seen[col].sum() * 100:.0f}% below forecast)")

# Forecast for the months with no real data in our file (Oct 2022 to Dec 2023)
future = out[out["actual"].isna()]
print(f"\nSARIMA forecast total for Oct 2022 to Dec 2023: GBP {future['sarima_forecast'].sum():,.0f} million")

fig, ax = plt.subplots(figsize=(12, 5))
history.loc["2012":].plot(ax=ax, color="grey", label="History (to Feb 2020)")
actual_all.loc["2020-03":].plot(ax=ax, color="black", linewidth=2, label="Actual (Mar 2020 to Sep 2022)")
sarima_fc.plot(ax=ax, color="tab:green", label="SARIMA forecast (no-COVID scenario)")
ax.fill_between(out.index, lower, upper, color="tab:green", alpha=0.2, label="95% interval")
hw_fc.plot(ax=ax, color="tab:orange", linestyle="--", label="Holt-Winters forecast")
ax.set_title("GMAK: pre-COVID pattern projected to Dec 2023 vs what actually happened")
ax.set_ylabel("Earnings from overseas visits (GBP million, current prices)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "05_final_forecast.png", dpi=150)
print("Saved results/05_final_forecast.csv and figures/05_final_forecast.png")

# ---------------------------------------------------------------------------------------------
# Year by year: how far below the projection was actual spending, and how did it recover?
# 2020 covers Mar-Dec (10 months) and 2022 covers Jan-Sep (9 months), because that is the data we have.
yearly = seen.groupby(seen.index.year).agg(
    months=("actual", "count"), actual=("actual", "sum"),
    sarima_forecast=("sarima_forecast", "sum"), holt_winters_forecast=("holt_winters_forecast", "sum"))
yearly["actual_as_%_of_sarima"] = yearly["actual"] / yearly["sarima_forecast"] * 100
yearly["actual_as_%_of_holt_winters"] = yearly["actual"] / yearly["holt_winters_forecast"] * 100
yearly.round(1).to_csv(RESULTS_DIR / "05_yearly_gap.csv")
print("\nActual spending as a share of the projection, by year (GBP million):")
print(yearly.round(1).to_string())

# Recovery chart: actual spending as a percentage of what the pre-COVID pattern implied for the same month
share = seen["actual"] / seen["sarima_forecast"] * 100
low_month = share.idxmin()
print(f"\nLowest point: {low_month:%b %Y}, actual was {share.min():.0f}% of the projection")
print(f"Latest month with data: {share.index[-1]:%b %Y}, actual was {share.iloc[-1]:.0f}% of the projection")

fig, ax = plt.subplots(figsize=(11, 4.5))
share.plot(ax=ax, color="black", linewidth=2)
ax.axhline(100, color="tab:green", linestyle="--", label="Pre-COVID pattern (100%)")
ax.set_ylim(0, 120)
ax.set_ylabel("Actual spending as % of the projection")
ax.set_title("Overseas visitor spending as a share of the pre-COVID pattern, month by month")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "05_recovery_share.png", dpi=150)
print("Saved results/05_yearly_gap.csv and figures/05_recovery_share.png")
