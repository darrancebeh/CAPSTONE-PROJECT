# Volatility Forecasting: GARCH-family vs LSTM

Backtesting framework for the capstone project *Comparative Analysis of
GARCH-family Econometric Models vs LSTM Models for Volatility Forecasting: A
Proxy-Robust Approach in US Equities*.

The system ingests daily OHLCV data for the SPDR S&P 500 ETF (SPY), engineers
volatility proxies and predictors, fits three GARCH-family models and one LSTM
architecture under two loss functions, produces one-day-ahead variance
forecasts through an expanding-window backtest, and scores them with a
proxy-robust loss function, formal significance tests and Value-at-Risk
coverage tests.

---

## 1. Quick start

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_pipeline.py
```

```bash
python scripts/summarise.py && python scripts/make_figures.py
```

The full backtest takes roughly three minutes of model time on a laptop CPU.
No GPU is required. Everything downstream reads the cached forecast panel, so
the expensive step runs once.

---

## 2. Directory structure

```
FYP/
├── config/
│   └── config.yaml              Every tunable setting: sample window, split
│                                date, regime boundaries, model hyperparameters
├── data/
│   ├── raw/
│   │   └── SPY_HistoricalOHLCV.parquet    Source data (2,766 daily bars)
│   └── processed/               Derived features and proxies, written by the
│                                pipeline
│
├── src/volforecast/             The package. All reusable logic lives here.
│   ├── config.py                Typed configuration objects loaded from YAML
│   ├── pipeline.py              End-to-end orchestration
│   │
│   ├── data/
│   │   ├── calendar.py          Reconstructed NYSE session calendar
│   │   ├── loader.py            Ingestion, validation, repair, audit report
│   │   └── features.py          Returns, proxy, 15 predictors, sequence builder
│   │
│   ├── models/
│   │   ├── base.py              The VolatilityModel contract (fit / predict)
│   │   ├── garch.py             GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1,1)
│   │   ├── lstm.py              Network, training loop, walk-forward wrapper
│   │   └── losses.py            QLIKE and MSE as differentiable objectives
│   │
│   ├── backtest/
│   │   ├── engine.py            Expanding-window engine + look-ahead audit
│   │   ├── regimes.py           Regime segmentation
│   │   └── cache.py             Parquet and JSON persistence
│   │
│   ├── evaluation/
│   │   ├── metrics.py           QLIKE, RMSE, MAE, R², Mincer-Zarnowitz
│   │   ├── diebold_mariano.py   Equal-accuracy test with HAC variance
│   │   ├── var_backtest.py      Kupiec, Christoffersen, conditional coverage
│   │   └── scorecard.py         Assembles the full audit table set
│   │
│   └── utils/                   Logging and seeding
│
├── scripts/                     Command-line entry points (see Section 4)
├── tests/                       95 tests
└── results/                     All output (see Section 5)
```

---

## 3. Workflow

The pipeline runs in five stages. Each stage depends only on the one before,
and the boundary between them is a file on disk, so any stage can be re-run
without repeating the ones before it.

```
  data/raw/SPY_HistoricalOHLCV.parquet
              │
    ┌─────────▼─────────┐
    │ 1. INGEST         │  loader.py
    │                   │  Validate against the NYSE calendar, reject
    │                   │  malformed bars, forward-fill genuine gaps
    └─────────┬─────────┘
              │  clean OHLCV frame + data quality report
    ┌─────────▼─────────┐
    │ 2. ENGINEER       │  features.py
    │                   │  Log returns, squared-return proxy (floored),
    │                   │  15 predictors, all measurable at close of day t
    └─────────┬─────────┘
              │  Dataset(prices, returns, proxy, features)
    ┌─────────▼─────────┐
    │ 3. BACKTEST       │  engine.py
    │                   │  For each out-of-sample date, compute the
    │                   │  information boundary, refit if due, forecast
    └─────────┬─────────┘
              │  results/forecasts/*.parquet
    ┌─────────▼─────────┐
    │ 4. EVALUATE       │  scorecard.py
    │                   │  QLIKE/RMSE/MAE/R², Diebold-Mariano, VaR coverage,
    │                   │  segmented by regime
    └─────────┬─────────┘
              │  results/tables/*.csv
    ┌─────────▼─────────┐
    │ 5. PRESENT        │  summarise.py, make_figures.py
    └───────────────────┘
```

### The information boundary

This is the core of the design. Every model implements exactly two methods:

```python
fit(dataset, train_end)        # estimate using data up to and including train_end
predict(dataset, target_date)  # forecast target_date, formed at the previous close
```

The **engine**, not the model, decides what `train_end` is. For a forecast
dated `t` it is the last session strictly before `t`. Keeping the loop outside
the models is what makes the comparison fair: no model can widen its own
information set, because it never chooses its own boundary.

This is verified rather than asserted. `WalkForwardBacktester.audit_no_lookahead`
re-estimates each model at sampled checkpoints against a dataset that has been
physically truncated at the boundary, and checks the forecast is unchanged. The
audit runs as part of the standard pipeline and raises if any forecast moves.
The committed run passes at **exactly zero difference** for all five models.

### Refit cadence

| Model | Refits | Between refits |
| --- | --- | --- |
| GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1,1) | every session (1,760) | n/a |
| LSTM (QLIKE), LSTM (MSE) | every 126 sessions (14) | weights frozen; features still update daily |

Re-training a network 1,760 times is not practical, so the LSTM is retrained
twice a year and filters forward in between. Because that asymmetry could be
argued to favour the econometric models, `scripts/sensitivity.py` repeats the
comparison at cadences of 21, 63, 126 and 252 sessions. GARCH(1,1) retains the
lower QLIKE at every cadence, so the conclusion does not depend on the schedule.

---

## 4. Scripts

Run all of these from the project root.

### `run_pipeline.py` — the main entry point

```bash
python scripts/run_pipeline.py                    # full backtest, ~3 minutes
python scripts/run_pipeline.py --tables-only      # rebuild tables from cache
python scripts/run_pipeline.py --models garch lstm_qlike
python scripts/run_pipeline.py --no-audit         # skip the look-ahead check
```

Loads the data, engineers features, runs the walk-forward for every model,
executes the look-ahead audit, evaluates, and writes everything to `results/`.
Prints the accuracy table, regime table, Diebold-Mariano tests and VaR coverage
to the console.

`--tables-only` re-derives every table from the cached forecasts without
re-estimating anything. Use this after changing how results are presented.

### `summarise.py` — the report tables

```bash
python scripts/summarise.py
```

Condenses the seventeen audit tables down to the six used in the report and
writes them to `results/summary/` as CSV plus a single `findings.md`. Clears
previous output first, so a renamed table cannot leave a stale file behind.

### `make_figures.py` — the report figures

```bash
python scripts/make_figures.py
```

Renders the six figures used in Chapter 4 to `results/figures/`, numbered in
order of appearance in the report.

### `tune_lstm.py` — architecture selection

```bash
python scripts/tune_lstm.py --trials 40 --seeds 2
```

Random search over forty candidate configurations, each trained on data through
December 2017 and scored by QLIKE on calendar year 2018. That year sits inside
the initial estimation window and ends before the first out-of-sample forecast
date, so nothing from the evaluation period informs the choice. Each candidate
is evaluated under two seeds and averaged.

Writes `results/tables/lstm_tuning.csv` and prints the selected configuration
ready to paste into `config.yaml`.

### `sensitivity.py` — robustness to the retraining schedule

```bash
python scripts/sensitivity.py --cadences 21 63 126 252
```

Re-runs only the neural models at each cadence and re-tests each against the
cached GARCH forecasts. Answers the "you refit GARCH daily but the LSTM twice a
year" objection. Takes roughly seven minutes.

### `clean.py` — reset to a fresh checkout

```bash
python scripts/clean.py --dry-run     # list what would be removed
python scripts/clean.py               # remove it
python scripts/clean.py --caches      # also clear __pycache__ and .pytest_cache
```

Deletes every generated artefact: `data/processed/`, `results/forecasts/`,
`results/tables/`, `results/summary/`, `results/figures/` and the run logs.
Roughly 39 files and 3 MB.

The raw input under `data/raw/` is never touched. The script resolves every
target path and aborts before deleting anything if one of them overlaps the raw
data directory, so a mistake in `config.yaml` cannot destroy the one input that
cannot be regenerated.

### `ceiling_check.py` — is the forecast ceiling architecture-specific?

```bash
python scripts/ceiling_check.py --top 5
```

Retrains the five best-scoring architectures from the tuning stage on data
through 2019 and reports the highest variance each one forecast during the
COVID-19 crash. Tests whether the ceiling is a property of the model family or
of the one configuration that was selected.

---

## 5. Observable outputs

Everything below is produced by the commands in Section 4.

### `results/forecasts/` — the cached run

| File | Contents |
| --- | --- |
| `evaluation_panel.parquet` | One row per out-of-sample day: realised proxy, return, regime label, and each model's forecast. **This is the file everything downstream reads.** |
| `forecasts.parquet` | The forecast panel alone |
| `fit_log.parquet` | Estimated parameters at every refit (ω, α, β, γ per date) |
| `run_metadata.json` | Configuration, regime spans, per-model runtime, data quality report, look-ahead audit result |

### `results/summary/` — the six report tables

| File | Report table |
| --- | --- |
| `table_1_descriptive.csv` | Table 4.1 — descriptive statistics, skewness, kurtosis, Jarque-Bera |
| `table_2_accuracy.csv` | Table 4.2 — QLIKE, rank, RMSE, MAE, R² |
| `table_3_regime_qlike.csv` | Table 4.3 — QLIKE by market regime |
| `table_4_significance.csv` | Table 4.4 — Diebold-Mariano vs GARCH(1,1) |
| `table_5_var_coverage.csv` | Table 4.5 — 95% VaR coverage |
| `table_6_capacity_and_cost.csv` | forecast ceiling and computational cost |
| `findings.md` | All six rendered as one Markdown document |

### `results/tables/` — the full audit trail

Seventeen CSVs kept for verification rather than presentation: per-regime
detail, the full pairwise DM matrix, MSE-based rankings, the look-ahead audit
record, the tuning search results, the cadence sensitivity, and the forecast
ceiling check.

### `results/figures/`

| File | Shows |
| --- | --- |
| `figure_4_1_price_and_volatility.png` | SPY price and realised volatility, 2015–2025 |
| `figure_4_2_regime_context.png` | Realised volatility with the five regimes shaded |
| `figure_4_3_regime_qlike.png` | QLIKE heatmap by model and regime |
| `figure_4_4_cumulative_qlike.png` | Loss accumulated relative to GARCH(1,1) |
| `figure_4_5_var_breaches.png` | Returns against the 95% VaR threshold |
| `figure_4_6_forecast_overlay.png` | Forecasts against the realised proxy |

### Console output

`run_pipeline.py` prints four tables: out-of-sample accuracy, average QLIKE by
regime, Diebold-Mariano tests, and VaR coverage. This is the fastest way to
confirm a run reproduced.

---

## 6. Configuration

All settings live in `config/config.yaml`. The ones that change results:

| Key | Current | Effect |
| --- | --- | --- |
| `split.initial_train_end` | `2018-12-31` | Where out-of-sample begins |
| `data.min_variance_proxy` | `1.0e-4` | Floor on the squared-return proxy so `log()` in QLIKE stays defined |
| `features.sequence_length` | `10` | LSTM lookback, selected by `tune_lstm.py` |
| `models.lstm.*` | h=32, 1 layer | Architecture, selected by `tune_lstm.py` |
| `backtest.refit_every` | GARCH 1, LSTM 126 | Refit cadence per model |
| `regimes` | five windows | Regime boundaries, fixed from documented events |

### The proxy floor

QLIKE contains `log(proxy)`, which is undefined on the eight sessions where SPY
closed exactly unchanged. The proxy is floored once at source at `1e-4` squared
percentage points, equivalent to a one basis point move. Every model and the
evaluator see the identical target. The floor binds on 36 of 2,765 sessions.

---

## 7. Tests

```bash
python -m pytest tests/ -q
```

95 tests, roughly 15 seconds, covering 88 per cent of statements in the package.

| File | Tests | Covers |
| --- | --- | --- |
| `test_data_pipeline.py` | 20 | Calendar edge cases, ingestion repairs, feature causality, sequence alignment |
| `test_models.py` | 22 | GARCH estimation, LSTM determinism, information boundary, walk-forward engine, regimes |
| `test_pipeline.py` | 12 | End-to-end run on synthetic data: orchestration, scorecard assembly, persistence, cache round-trip, model registry |
| `test_diebold_mariano.py` | 11 | HAC variance, known significant and degenerate cases, antisymmetry |
| `test_var_backtest.py` | 10 | Kupiec and Christoffersen against hand-computed likelihood ratios |
| `test_metrics.py` | 10 | QLIKE, Mincer-Zarnowitz, out-of-sample R² |
| `test_losses.py` | 8 | QLIKE gradient, asymmetry, agreement between training and evaluation implementations |

Three tests are worth knowing about because they defend the project's central
claims:

- **`test_features_are_causal`** — rebuilds the feature matrix from a truncated
  price series and asserts every earlier row is unchanged.
- **`test_future_rows_cannot_affect_earlier_sequences`** — corrupts the tail of
  the feature frame and asserts training sequences for earlier dates are
  bit-identical.
- **`test_refitting_the_same_window_is_deterministic`** — fits the LSTM twice on
  the same window and asserts the forecasts match to 1e-12. The per-refit seed
  is a function of the training-window end date rather than a counter, which is
  what makes the look-ahead audit meaningful.

---

## 8. Reproducibility

Every source of randomness is seeded from `project.random_seed`. The LSTM's
per-refit seed derives from the training window boundary, so re-estimating the
same window twice is bit-identical regardless of what preceded it.

Removing a model from the suite and re-running reproduced the remaining five
models' QLIKE, RMSE, MAE and R² to four decimal places, which is the practical
check that the seeding works.

### Rebuilding from scratch

```bash
python scripts/clean.py
python scripts/run_pipeline.py && python scripts/summarise.py && python scripts/make_figures.py
```

Takes about three and a half minutes end to end and restores 35 of the 38
generated files.

This has been verified: after deleting every artefact and rebuilding from the
raw parquet alone, **every accuracy number was bit-identical** to the committed
run, including QLIKE, RMSE, MAE, R-squared, the regime breakdown, all
Diebold-Mariano statistics and p-values, the VaR exception counts and the
forecast ceilings. The only fields that changed were the wall-clock timings in
`results/tables/runtime.csv` and the two tables that carry them, which differ
between runs by a few seconds.

Three tables are **not** produced by `run_pipeline.py`, because they come from
the supplementary analyses rather than the main backtest. Regenerate them
individually if needed:

| Table | Script | Approximate runtime |
| --- | --- | --- |
| `lstm_tuning.csv` | `python scripts/tune_lstm.py --trials 40 --seeds 2` | 8 minutes |
| `sensitivity_refit_cadence.csv` | `python scripts/sensitivity.py` | 7 minutes |
| `lstm_forecast_ceiling.csv` | `python scripts/ceiling_check.py --top 5` | 1 minute |

---

## 9. Data

Daily OHLCV bars for SPY from 2 January 2015 to 31 December 2025, sourced via
the Interactive Brokers API, dividend-adjusted, stored as Parquet.

The ingestion layer reconciles the feed against a reconstructed NYSE session
calendar accounting for Good Friday, the 2022 Juneteenth rule change, ad-hoc
closures such as the 2018 national day of mourning, and the fact that the
exchange does **not** close on the Friday preceding a Saturday New Year's Day.

The delivered feed reconciles exactly: 2,766 expected sessions, 2,766 observed,
no gaps, no duplicates, no OHLC violations, no non-positive prices. The
forward-filling path in the loader is therefore never exercised on this data,
but is tested against synthetic gaps in `test_data_pipeline.py`.
