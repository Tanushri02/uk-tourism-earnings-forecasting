"""Step 3: do the model's mistakes (residuals) look like random noise?

A good model leaves residuals with no pattern. If they still contain structure, the model missed something.
We check this on the training data for the model chosen in step 2.
"""
import json
import warnings

import numpy as np
import statsmodels.api as sm

from common import FIG_DIR, MAXITER, RESULTS_DIR, load_gmak, plt, train_test_split

warnings.filterwarnings("ignore")

with open(RESULTS_DIR / "best_sarima.json") as f:
    spec = json.load(f)
order, seasonal = tuple(spec["order"]), tuple(spec["seasonal_order"])

train, _ = train_test_split(load_gmak())
fit = sm.tsa.SARIMAX(np.log(train), order=order, seasonal_order=seasonal).fit(disp=False, maxiter=MAXITER)
print(f"Model: SARIMA{order}{seasonal} on log(GMAK), fitted to {len(train)} training months\n")

# statsmodels drops the first few start-up residuals automatically in these tests
lb = fit.test_serial_correlation("ljungbox", lags=[12, 24])      # shape: (1 test, 3 stats, n lags)
print("Ljung-Box (are residuals independent? p > 0.05 suggests yes)")
for i, lag in enumerate([12, 24]):
    print(f"  lag {lag}: statistic = {lb[0][0][i]:.1f}, p-value = {lb[0][1][i]:.3f}")

jb = fit.test_normality("jarquebera")
print(f"\nJarque-Bera (are residuals roughly normal? p > 0.05 suggests yes)\n  p-value = {jb[0][1]:.3f}, "
      f"skew = {jb[0][2]:.2f}, kurtosis = {jb[0][3]:.2f}")

het = fit.test_heteroskedasticity("breakvar")
print(f"\nHeteroskedasticity (is the error size steady over time? p > 0.05 suggests yes)\n  p-value = {het[0][1]:.3f}")

fig = fit.plot_diagnostics(figsize=(12, 8))
fig.tight_layout()
fig.savefig(FIG_DIR / "03_residual_diagnostics.png", dpi=150)
print("\nSaved figures/03_residual_diagnostics.png")
