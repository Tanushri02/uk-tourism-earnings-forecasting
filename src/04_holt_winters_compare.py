"""Step 4: Holt-Winters (exponential smoothing) on the SAME train/test split, then one comparison table.

Holt-Winters keeps three running estimates: the level, the trend and the seasonal pattern, and updates
each one with every new month. Candidates:
  - additive trend + multiplicative seasonality (raw data)
  - damped trend (trend flattens out over time) + multiplicative seasonality (raw data)
  - additive trend + additive seasonality on log data (a log turns multiplicative effects into additive ones)
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import FIG_DIR, RESULTS_DIR, load_gmak, plt, scores, train_test_split

warnings.filterwarnings("ignore")

y = load_gmak()
train, test = train_test_split(y)
n = len(test)

hw_mul = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
hw_damped = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12).fit()
hw_log = ExponentialSmoothing(np.log(train), trend="add", seasonal="add", seasonal_periods=12).fit()

hw_forecasts = {
    "Holt-Winters add trend, mul seasonal": hw_mul.forecast(n),
    "Holt-Winters damped trend, mul seasonal": hw_damped.forecast(n),
    "Holt-Winters add/add on log": np.exp(hw_log.forecast(n)),
}
hw_table = pd.DataFrame({name: scores(test, fc) for name, fc in hw_forecasts.items()}).T

# Combine with step 1 (naive and SARIMA models)
step1 = pd.read_csv(RESULTS_DIR / "01_holdout_scores.csv", index_col=0)
combined = pd.concat([step1, hw_table]).round(2).sort_values("RMSE")
combined.to_csv(RESULTS_DIR / "04_final_comparison.csv")
print("All models on the same 24 hold-out months (Mar 2018 to Feb 2020), best first:")
print(combined)

naive_rmse = combined.loc["Seasonal naive", "RMSE"]
best_name = combined.index[0]
gain = (1 - combined.iloc[0]["RMSE"] / naive_rmse) * 100
print(f"\nBest by RMSE: {best_name}, {gain:.0f}% lower RMSE than the seasonal naive baseline")

fig, ax = plt.subplots(figsize=(9, 4.5))
combined["MAPE_%"].sort_values(ascending=False).plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("MAPE on 24 hold-out months (%), lower is better")
ax.set_title("GMAK: how far off was each model on months it had not seen?")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_model_comparison.png", dpi=150)
print("Saved results/04_final_comparison.csv and figures/04_model_comparison.png")
