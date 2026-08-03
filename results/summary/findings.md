# Findings summary

Sample 2015-01-05 to 2025-12-31. Out-of-sample 2019-01-02 to 2025-12-31, 1760 sessions.

## Table 4.1. Descriptive statistics, full sample

| Series              |    N |   Mean |   Std. dev. |      Min |      Max |   Skewness |   Kurtosis |   JB p-value |
|:--------------------|-----:|-------:|------------:|---------:|---------:|-----------:|-----------:|-------------:|
| Return (%)          | 2765 | 0.0501 |      1.1233 | -11.5875 |   9.9869 |    -0.5829 |    17.5353 |            0 |
| Squared return (%²) | 2765 | 1.2639 |      5.1151 |   0.0001 | 134.271  |    15.2987 |   308.592  |            0 |

## Table 4.2. Out-of-sample forecast accuracy, 1,760 sessions

| Model            |   QLIKE |   Rank (QLIKE) |   RMSE |    MAE |     R2 |
|:-----------------|--------:|---------------:|-------:|-------:|-------:|
| GARCH(1,1)       |  1.49   |              3 | 5.2826 | 1.6028 | 0.2835 |
| EGARCH(1,1)      |  1.4715 |              1 | 5.3582 | 1.5036 | 0.2629 |
| GJR-GARCH(1,1,1) |  1.4746 |              2 | 5.3587 | 1.6028 | 0.2628 |
| LSTM (QLIKE)     |  1.645  |              4 | 6.1288 | 1.4883 | 0.0357 |
| LSTM (MSE)       |  1.6862 |              5 | 6.1453 | 1.4803 | 0.0304 |

## Table 4.3. Average QLIKE by market regime

| Model            |   Calm Bull |   COVID Crash |   Recovery Rally |   Rate-Hike Cycle |   Post-Hike Normalisation |   Full sample |
|:-----------------|------------:|--------------:|-----------------:|------------------:|--------------------------:|--------------:|
| GARCH(1,1)       |       1.5   |         1.55  |            1.385 |             1.411 |                     1.567 |         1.49  |
| EGARCH(1,1)      |       1.413 |         1.635 |            1.472 |             1.389 |                     1.51  |         1.472 |
| GJR-GARCH(1,1,1) |       1.442 |         1.372 |            1.45  |             1.43  |                     1.522 |         1.475 |
| LSTM (QLIKE)     |       1.369 |         7.072 |            1.471 |             1.306 |                     1.599 |         1.645 |
| LSTM (MSE)       |       1.421 |         8.116 |            1.468 |             1.364 |                     1.589 |         1.686 |

## Table 4.4. Diebold-Mariano tests against GARCH(1,1)

| Scope                   | Model vs GARCH(1,1)   |   DM statistic |   p-value | Verdict             |
|:------------------------|:----------------------|---------------:|----------:|:--------------------|
| Full sample             | EGARCH(1,1)           |         -0.754 |    0.4511 | No difference       |
| Full sample             | GJR-GARCH(1,1,1)      |         -0.797 |    0.4255 | No difference       |
| Full sample             | LSTM (QLIKE)          |          1.597 |    0.1104 | No difference       |
| Full sample             | LSTM (MSE)            |          1.786 |    0.0742 | No difference       |
| Calm Bull               | LSTM (QLIKE)          |         -3.962 |    0.0001 | LSTM (QLIKE) better |
| COVID Crash             | LSTM (QLIKE)          |          2.852 |    0.0063 | GARCH(1,1) better   |
| Recovery Rally          | LSTM (QLIKE)          |          0.685 |    0.4938 | No difference       |
| Rate-Hike Cycle         | LSTM (QLIKE)          |         -3.73  |    0.0002 | LSTM (QLIKE) better |
| Post-Hike Normalisation | LSTM (QLIKE)          |          0.482 |    0.63   | No difference       |

## Table 4.5. 95% Value-at-Risk coverage

| Model            |   Breaches |   Expected |   Kupiec p |   Joint p | Passes   |
|:-----------------|-----------:|-----------:|-----------:|----------:|:---------|
| GARCH(1,1)       |        106 |         88 |     0.0561 |    0.1562 | Yes      |
| EGARCH(1,1)      |        115 |         88 |     0.0047 |    0.0181 | No       |
| GJR-GARCH(1,1,1) |        113 |         88 |     0.0087 |    0.0205 | No       |
| LSTM (QLIKE)     |        129 |         88 |     0      |    0.0001 | No       |
| LSTM (MSE)       |        132 |         88 |     0      |    0      | No       |

## Table 4.6. Forecast ceiling and computational cost

| Model            |   Highest forecast (ann. %) |   Share of worst day |   Refits |   Runtime (s) |
|:-----------------|----------------------------:|---------------------:|---------:|--------------:|
| GARCH(1,1)       |                     132.078 |                0.516 |     1760 |        17.415 |
| EGARCH(1,1)      |                     109.259 |                0.353 |     1760 |        31.35  |
| GJR-GARCH(1,1,1) |                     151.203 |                0.676 |     1760 |        21.28  |
| LSTM (QLIKE)     |                      33.322 |                0.033 |       14 |        50.605 |
| LSTM (MSE)       |                      34.586 |                0.035 |       14 |        59.752 |
