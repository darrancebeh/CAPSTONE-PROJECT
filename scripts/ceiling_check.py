"""Test whether the LSTM forecast ceiling is architecture-specific.

Section 4.7 of the report attributes the network's failure during the COVID-19
crash to a structural ceiling on the variance it can output, rather than to the
particular architecture that was selected. That is a claim about the model
family, so it needs checking against more than one configuration.

This script retrains the best-scoring architectures from the selection stage on
data through the end of 2019, forecasts the crash window, and reports the
highest variance each one issued against what the market actually delivered.

    python scripts/ceiling_check.py --top 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from volforecast.config import Config
from volforecast.models.lstm import LstmVolatilityModel
from volforecast.pipeline import prepare_dataset
from volforecast.utils import get_logger, set_global_seed

logger = get_logger("ceiling_check")

TRADING_DAYS = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--top", type=int, default=5, help="Architectures to retest")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--window-start", default="2020-02-20")
    parser.add_argument("--window-end", default="2020-04-30")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.load(args.config)
    set_global_seed(config.random_seed)

    tuning_path = config.backtest.table_dir / "lstm_tuning.csv"
    if not tuning_path.exists():
        logger.error("No tuning results at %s. Run scripts/tune_lstm.py first.", tuning_path)
        return 1

    dataset, _ = prepare_dataset(config)
    candidates = pd.read_csv(tuning_path).sort_values("qlike_mean").head(args.top)

    index = dataset.common_index()
    window = index[
        (index >= pd.Timestamp(args.window_start)) & (index <= pd.Timestamp(args.window_end))
    ]
    worst = float(dataset.proxy.loc[window].max())

    logger.info(
        "Crash window %s to %s: %d sessions, worst realised %.0f%% annualised",
        args.window_start,
        args.window_end,
        len(window),
        np.sqrt(worst * TRADING_DAYS),
    )

    rows = []
    for _, candidate in candidates.iterrows():
        model = LstmVolatilityModel(
            loss_name="qlike",
            sequence_length=int(candidate["sequence_length"]),
            hidden_size=int(candidate["hidden_size"]),
            num_layers=int(candidate["num_layers"]),
            dropout=float(candidate["dropout"]),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            batch_size=int(candidate["batch_size"]),
            max_epochs=300,
            patience=25,
            min_variance=float(config.models["lstm"]["min_variance"]),
            seed=config.random_seed,
        )
        model.fit(dataset, train_end=pd.Timestamp(args.train_end))
        forecasts = np.array([model.predict(dataset, date) for date in window])
        highest = float(forecasts.max())

        rows.append(
            {
                "hidden_size": int(candidate["hidden_size"]),
                "num_layers": int(candidate["num_layers"]),
                "sequence_length": int(candidate["sequence_length"]),
                "dropout": float(candidate["dropout"]),
                "highest_forecast": round(highest, 3),
                "highest_annualised_pct": round(float(np.sqrt(highest * TRADING_DAYS)), 1),
                "share_of_worst_day": round(highest / worst, 4),
            }
        )

    frame = pd.DataFrame(rows)
    output = config.backtest.table_dir / "lstm_forecast_ceiling.csv"
    frame.to_csv(output, index=False)

    pd.set_option("display.width", 180)
    print(f"\nWorst realised session in the window: {np.sqrt(worst * TRADING_DAYS):.0f}% annualised")
    print("\nHighest forecast issued by each architecture")
    print(frame.to_string(index=False))
    print(
        f"\nCeiling range: {frame['highest_annualised_pct'].min():.0f}% to "
        f"{frame['highest_annualised_pct'].max():.0f}% annualised across "
        f"{len(frame)} architectures spanning hidden sizes "
        f"{frame['hidden_size'].min()} to {frame['hidden_size'].max()}."
    )
    logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
