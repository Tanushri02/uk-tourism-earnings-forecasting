# Method notes: the ideas behind each step, in plain English

These notes explain the choices in this repository without assuming a statistics background. Rupee examples are illustrations only, not project data.

**Reading guide.** If you are new to forecasting, start with sections 1, 2 and 8. Sections 3 to 7 cover the modelling choices in more depth.

## 1. Hold-out and rolling origin

A **hold-out test** hides the most recent months, fits each model on the earlier data, forecasts the hidden months and compares the forecasts with what really happened. It measures how a model does on months it has never seen, which is what a real forecast faces.

One window can mislead. It is small, so chance can favour one model, and the stretch of time it covers has its own conditions, such as an unusually strong season. A **rolling origin** repeats the test. The "origin" is the last month of training data. Window 1 trains to February 2010 and tests the next 24 months. Window 2 trains to February 2011, and so on up to February 2018. Every model is refitted in every window and the errors are averaged, as with a student who sits nine tests instead of one.

## 2. Baselines

A baseline is the simplest sensible forecast. **Seasonal naive** predicts each month as the same month a year earlier. **Seasonal naive + drift** adds the average yearly growth, once for each year ahead. The copied value gets older the further ahead you forecast, so growth accumulates. If June last year was ₹1,00,000 and Junes rise ₹6,000 a year on average, next June is about ₹1,06,000 and the June after is about ₹1,12,000, because both start from the same old June. In the code, `np.arange(n) // 12 + 1` gives 1 for the first twelve months and 2 for the next twelve.

## 3. The airline model, SARIMA(0,1,1)(0,1,1)12

Notation: (p, d, q)(P, D, Q, m).

- `d = 1`: work with month-to-month changes, which removes the trend.
- `D = 1`: compare each month with the same month a year earlier, which removes the yearly pattern.
- `q = 1` and `Q = 1`: the model corrects itself using last month's forecast error and the error from 12 months ago.
- `p = 0` and `P = 0`: no direct dependence on past values. The search found they were not needed.
- `m = 12`: the season is 12 months long.

With no correction terms, both differences together say: this month equals last month plus the month-to-month change a year ago. Equivalently, it equals the same month last year plus last month's year-on-year change. The "dials" are the model's parameters: two correction strengths and the noise level, against eight correction strengths for SARIMA(2,1,2)(2,1,2)12.

The name comes from the classic Box and Jenkins example on monthly airline passengers. It is a standard default for monthly data with a trend and yearly seasonality, usually fitted on logs.

## 4. Why logs, and converting back

The seasonal swings in the raw series widen as earnings grow (visible in `figures/01_holdout_forecasts.png`). A logarithm turns percentage changes into equal-sized steps. A shop whose July sales are always 40% above January swings by ₹40,000 at a ₹1,00,000 level and ₹2,00,000 at a ₹5,00,000 level, but both swings equal log(1.4) in logs. Models like steady swings, so logs help, and `np.exp` converts forecasts back to £ million.

**Median vs mean.** Converting back with `np.exp` gives the *median* forecast, not the mean. The mean is slightly higher, and the gap grows with the forecast horizon because uncertainty compounds. For one month's value the median is a fine answer. For a **total** across months, such as the gap in the README, means are the right quantity, because means add up and medians do not.

A rupee example: daily sales are ₹1,000 on 60% of days and ₹4,000 on 40%. One day's median is ₹1,000 and its mean is ₹2,200. Over two days the totals are ₹2,000 (36%), ₹5,000 (48%) and ₹8,000 (16%). The sum of the two daily medians is ₹2,000, the median of the two-day total is ₹5,000, and the expected total is ₹4,400, exactly twice the daily mean. In this data the skew is small, so the difference is small, but totals reported here come from median forecasts.

## 5. Choosing the SARIMA order: AIC and BIC

AIC = 2 × (number of parameters) − 2 × (log-likelihood). Log-likelihood measures how plausible the data are under the model, so a better fit lowers AIC and every extra parameter adds 2. Lower is better. Models within about 2 points of each other are essentially tied, and the simpler one is preferred. BIC charges more per parameter (about 6.1 here, since ln 458 ≈ 6.1), so it penalises complexity harder. Only the airline model was within 2 points of the best by BIC.

The search uses training data only, so the test months stay an honest exam. Raw and log models are compared by forecast errors on the test months, not by AIC, because AIC is not comparable across different transformations of the data.

## 6. Convergence

A SARIMA fit searches for its parameters by trial and error, like tuning radio dials until the signal stops improving. "Failed to converge" means the search ran out of attempts before settling. The default of 50 attempts was too few for the larger model, so the code allows 500. The one fit that still did not converge, SARIMA(1,1,0)(0,1,0)12, is far from the top of the ranking (details in the README).

## 7. Prediction intervals

The shaded band in `figures/05_final_forecast.png` is a 95% prediction interval: the range in which the model expects the actual value to fall, if its assumptions hold. Intervals are computed on the log scale, where they are symmetric, and converted back, which stretches the upper side more than the lower. The percentage width depends only on how far ahead you are, so the band fans out over time. Summer bands are wider in pounds only because summer levels are higher.

## 8. What the post-2020 comparison is and is not

The models are fitted only on data up to February 2020 and projected forward. The gap is the difference between that projection and actual earnings from March 2020 to September 2022. It answers "how far below the old pattern was spending?" It does not isolate COVID from everything else that changed.

It is also in current prices, meaning each year's own pounds, not adjusted for inflation (also called nominal terms). Comparing across years therefore mixes real changes in spending with price rises, and a gap measured in inflation-adjusted pounds would differ. A rupee example: a thali costs ₹100 in 2019 and ₹120 in 2022, so ₹1,000 and ₹1,200 both buy 10 thalis. Spending grew 20% in current prices and 0% in 2019 prices.
