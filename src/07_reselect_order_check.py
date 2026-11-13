"""Step 7 (optional, about 3 minutes): does choosing the SARIMA order WITHOUT peeking change anything?

In step 2 the order was chosen using data up to Feb 2018. For earlier rolling windows that is a small peek at
the future. Here we repeat the 36-model AIC search inside each window, using only data up to that window's
origin, and compare with simply using the airline model every time.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import MAXITER, RESULTS_DIR, load_gmak, scores

warnings.filterwarnings("ignore")      # the search fits 36 models per window; we only need the winners
y = load_gmak()
HORIZON = 24

rows = []
for year in range(2010, 2019):
    end = pd.Timestamp(f"{year}-02-01")
    log_train = np.log(y.loc[:end])
    test = y.loc[end + pd.offsets.MonthBegin(1): end + pd.offsets.MonthBegin(HORIZON)]

    best = None
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
        order, seasonal = (p, 1, q), (P, 1, Q, 12)
        aic = sm.tsa.SARIMAX(log_train, order=order, seasonal_order=seasonal).fit(disp=False, maxiter=MAXITER).aic
        if best is None or aic < best[0]:
            best = (aic, order, seasonal)

    def mape(order, seasonal):
        fit = sm.tsa.SARIMAX(log_train, order=order, seasonal_order=seasonal).fit(disp=False, maxiter=MAXITER)
        return scores(test, np.exp(fit.forecast(HORIZON)))["MAPE_%"]

    rows.append({"origin": year, "chosen_order": str(best[1]) + str(best[2]),
                 "MAPE_chosen_%": mape(best[1], best[2]), "MAPE_airline_%": mape((0, 1, 1), (0, 1, 1, 12))})
    print(f"Feb {year}: chose {rows[-1]['chosen_order']}", flush=True)

out = pd.DataFrame(rows).round(2)
out.to_csv(RESULTS_DIR / "07_reselect_order_check.csv", index=False)
print(out.to_string(index=False))
print("\nAverage MAPE: re-selected each window =", round(out["MAPE_chosen_%"].mean(), 2),
      "| always the airline model =", round(out["MAPE_airline_%"].mean(), 2))
