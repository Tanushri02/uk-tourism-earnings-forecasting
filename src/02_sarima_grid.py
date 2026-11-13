"""Step 2: choose SARIMA orders in a principled way, using the TRAINING data only.

We work on log(GMAK): the seasonal swings grow as the series grows, and a log makes them steadier.
We try every combination of small orders and rank them by AIC (lower = better trade-off between
fit and complexity). The test months are only used afterwards, to check the winner.
"""
import itertools
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import MAXITER, RESULTS_DIR, load_gmak, scores, train_test_split

warnings.filterwarnings("ignore")

y = load_gmak()
train, test = train_test_split(y)
log_train = np.log(train)

rows = []
for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    order, seasonal = (p, 1, q), (P, 1, Q, 12)       # d = 1 and D = 1 are fixed (see the stationarity tests)
    try:
        fit = sm.tsa.SARIMAX(log_train, order=order, seasonal_order=seasonal).fit(disp=False, maxiter=MAXITER)
        rows.append({"order": order, "seasonal_order": seasonal, "AIC": fit.aic, "BIC": fit.bic,
                     "converged": bool(fit.mle_retvals["converged"])})
    except Exception as err:                          # a few combinations may fail to fit; skip them
        print("skipped", order, seasonal, err)

grid = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
grid.to_csv(RESULTS_DIR / "02_sarima_grid.csv", index=False)
print(f"Fits that did not fully converge: {int((~grid['converged']).sum())} of {len(grid)}")
print("Top 5 by AIC (log scale, training data):")
print(grid.head(5).round(1).to_string())

# Check the winner on the 24 test months
best = grid.iloc[0]
best_fit = sm.tsa.SARIMAX(log_train, order=best["order"], seasonal_order=best["seasonal_order"]).fit(disp=False, maxiter=MAXITER)
test_fc = np.exp(best_fit.forecast(len(test)))
print(f"\nBest by AIC: SARIMA{best['order']}{best['seasonal_order']} on log data")
print("Hold-out scores:", {k: round(v, 2) for k, v in scores(test, test_fc).items()})

with open(RESULTS_DIR / "best_sarima.json", "w") as f:
    json.dump({"order": list(best["order"]), "seasonal_order": list(best["seasonal_order"])}, f)
print("Saved results/02_sarima_grid.csv and results/best_sarima.json")
