"""Architecture selection for the LSTM, isolated from the evaluation sample.

The comparison in this project is only meaningful if the network is given a
fair configuration. Hyperparameters are therefore chosen by random search on a
holdout year that sits entirely inside the initial training window and ends
before the first out-of-sample forecast date, so nothing about the evaluation
period informs the choice.

Random search rather than a grid: over a space this size a grid spends most of
its budget varying parameters that do not matter, whereas random search covers
each individual dimension at a finer resolution for the same number of fits.

The selected architecture is then used unchanged for both the QLIKE-trained
and the MSE-trained network, which is what makes the loss-function ablation an
ablation rather than two unrelated models.

    python scripts/tune_lstm.py --trials 40 --seeds 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from volforecast.config import Config
from volforecast.evaluation.metrics import mincer_zarnowitz, qlike_loss
from volforecast.models.base import ModelFitError
from volforecast.models.lstm import LstmVolatilityModel
from volforecast.pipeline import prepare_dataset
from volforecast.utils import get_logger, set_global_seed

logger = get_logger("tune_lstm")

SEARCH_SPACE: Dict[str, List] = {
    "hidden_size": [8, 16, 24, 32, 48, 64],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    "sequence_length": [10, 21, 42],
    "batch_size": [32, 64, 128],
    "weight_decay": [0.0, 1e-5, 1e-4],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=40, help="Random search budget")
    parser.add_argument(
        "--seeds",
        type=int,
        default=2,
        help="Independent initialisations per candidate; scores are averaged to "
        "reduce the influence of a lucky seed",
    )
    parser.add_argument(
        "--train-end",
        default="2017-12-29",
        help="Last session used to fit a candidate",
    )
    parser.add_argument(
        "--validation-end",
        default="2018-12-31",
        help="Last session of the selection holdout; must precede the backtest split",
    )
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


def sample_configuration(rng: np.random.Generator) -> Dict:
    return {name: rng.choice(values).item() for name, values in SEARCH_SPACE.items()}


def evaluate_candidate(
    dataset,
    candidate: Dict,
    train_end: pd.Timestamp,
    validation_dates: pd.DatetimeIndex,
    seeds: List[int],
    max_epochs: int,
    patience: int,
    min_variance: float,
) -> Dict:
    scores, slopes, epochs = [], [], []

    for seed in seeds:
        model = LstmVolatilityModel(
            loss_name="qlike",
            sequence_length=int(candidate["sequence_length"]),
            hidden_size=int(candidate["hidden_size"]),
            num_layers=int(candidate["num_layers"]),
            dropout=float(candidate["dropout"]),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            batch_size=int(candidate["batch_size"]),
            max_epochs=max_epochs,
            patience=patience,
            min_variance=min_variance,
            seed=seed,
        )
        model.fit(dataset, train_end=train_end)

        forecasts = np.array([model.predict(dataset, date) for date in validation_dates])
        realised = dataset.proxy.loc[validation_dates].to_numpy()

        scores.append(float(qlike_loss(realised, forecasts).mean()))
        slopes.append(mincer_zarnowitz(realised, forecasts)["mz_slope"])
        epochs.append(model.history.best_epoch)

    return {
        **candidate,
        "qlike_mean": float(np.mean(scores)),
        "qlike_worst": float(np.max(scores)),
        "qlike_spread": float(np.max(scores) - np.min(scores)),
        "mz_slope_mean": float(np.mean(slopes)),
        "best_epoch_mean": float(np.mean(epochs)),
    }


def main() -> int:
    args = parse_args()
    config = Config.load(args.config)
    set_global_seed(config.random_seed)

    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    split_date = pd.Timestamp(config.split.initial_train_end)

    if validation_end > split_date:
        logger.error(
            "Selection holdout ends %s, after the backtest split at %s. That would leak "
            "evaluation data into the architecture choice.",
            validation_end.date(),
            split_date.date(),
        )
        return 1

    dataset, _ = prepare_dataset(config)
    candidates_index = dataset.common_index()
    validation_dates = candidates_index[
        (candidates_index > train_end) & (candidates_index <= validation_end)
    ]

    logger.info(
        "Fitting on data through %s, selecting on %d sessions ending %s",
        train_end.date(),
        len(validation_dates),
        validation_end.date(),
    )

    rng = np.random.default_rng(config.random_seed)
    seeds = [config.random_seed + offset for offset in range(args.seeds)]
    min_variance = float(config.models["lstm"]["min_variance"])

    seen = set()
    records = []
    for trial in range(1, args.trials + 1):
        candidate = sample_configuration(rng)
        signature = tuple(sorted(candidate.items()))
        if signature in seen:
            continue
        seen.add(signature)

        try:
            record = evaluate_candidate(
                dataset=dataset,
                candidate=candidate,
                train_end=train_end,
                validation_dates=validation_dates,
                seeds=seeds,
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_variance=min_variance,
            )
        except ModelFitError as exc:
            logger.warning("Trial %d rejected: %s", trial, exc)
            continue

        records.append(record)
        logger.info(
            "Trial %2d/%d  QLIKE %.4f (spread %.4f)  h=%d L=%d do=%.1f lr=%.0e seq=%d bs=%d",
            trial,
            args.trials,
            record["qlike_mean"],
            record["qlike_spread"],
            record["hidden_size"],
            record["num_layers"],
            record["dropout"],
            record["learning_rate"],
            record["sequence_length"],
            record["batch_size"],
        )

    if not records:
        logger.error("No candidate completed successfully")
        return 1

    results = pd.DataFrame(records).sort_values("qlike_mean").reset_index(drop=True)
    output = config.backtest.table_dir
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "lstm_tuning.csv", index=False)

    best = results.iloc[0]
    pd.set_option("display.width", 200)
    print("\nTop candidates by holdout QLIKE")
    print(
        results.head(10)[
            [
                "hidden_size",
                "num_layers",
                "dropout",
                "learning_rate",
                "sequence_length",
                "batch_size",
                "weight_decay",
                "qlike_mean",
                "qlike_spread",
                "mz_slope_mean",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nSelected configuration for config/config.yaml:")
    print(f"  sequence_length: {int(best['sequence_length'])}   # under features:")
    for field in ("hidden_size", "num_layers", "batch_size"):
        print(f"  {field}: {int(best[field])}")
    for field in ("dropout", "learning_rate", "weight_decay"):
        print(f"  {field}: {best[field]}")
    print(f"\nHoldout QLIKE {best['qlike_mean']:.4f}, seed spread {best['qlike_spread']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
