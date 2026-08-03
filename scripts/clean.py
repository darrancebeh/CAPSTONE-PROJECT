"""Delete every generated artefact so the pipeline can be re-run from scratch.

Removes derived data, cached forecasts, result tables and figures. The raw
input under data/raw is never touched, and the script refuses to run if a
target directory would resolve inside it.

    python scripts/clean.py --dry-run     # list what would be removed
    python scripts/clean.py               # remove it
    python scripts/clean.py --caches      # also clear __pycache__ and .pytest_cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import _bootstrap  # noqa: F401

from volforecast.config import PROJECT_ROOT, Config
from volforecast.utils import get_logger

logger = get_logger("clean")


def collect_targets(config: Config, include_caches: bool) -> List[Path]:
    """Directories and files the pipeline writes, in deletion order."""
    summary_dir = Path(config.backtest.table_dir).parent / "summary"

    targets: List[Path] = [
        config.data.processed_dir,
        config.backtest.forecast_dir,
        config.backtest.table_dir,
        config.backtest.figure_dir,
        summary_dir,
    ]
    targets.extend(sorted(PROJECT_ROOT.glob("results/*.log")))

    if include_caches:
        targets.extend(sorted(PROJECT_ROOT.rglob("__pycache__")))
        targets.append(PROJECT_ROOT / ".pytest_cache")

    return [t for t in targets if t.exists()]


def guard_raw_data(config: Config, targets: List[Path]) -> None:
    """Abort if any target would delete the source data.

    The raw feed is the one thing in the project that cannot be regenerated,
    so this check runs before anything is removed rather than relying on the
    target list being correct by inspection.
    """
    raw_dir = Path(config.data.raw_path).resolve().parent
    for target in targets:
        resolved = target.resolve()
        if resolved == raw_dir or raw_dir in resolved.parents or resolved in raw_dir.parents:
            raise SystemExit(f"Refusing to run: target {target} overlaps the raw data at {raw_dir}")


def measure(path: Path) -> Tuple[int, int]:
    """Return the file count and total bytes under a path."""
    if path.is_file():
        return 1, path.stat().st_size
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="List the targets without deleting them"
    )
    parser.add_argument(
        "--caches",
        action="store_true",
        help="Also remove __pycache__ directories and .pytest_cache",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    targets = collect_targets(config, include_caches=args.caches)
    guard_raw_data(config, targets)

    if not targets:
        print("Nothing to remove; the project is already clean.")
        return 0

    total_files = 0
    total_bytes = 0
    for target in targets:
        count, size = measure(target)
        total_files += count
        total_bytes += size
        relative = target.relative_to(PROJECT_ROOT)
        display = f"{relative}/" if target.is_dir() else str(relative)
        action = "would remove" if args.dry_run else "removing"
        print(f"  {action:<13} {display:<28} {count:>4} files  {size / 1024:>8.0f} KB")

    print(f"\n  {total_files} files, {total_bytes / 1024 / 1024:.1f} MB")

    if args.dry_run:
        print("\nDry run; nothing was deleted.")
        return 0

    for target in targets:
        if target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    raw = Path(config.data.raw_path)
    print(f"\nRaw input retained: {raw.relative_to(PROJECT_ROOT)} ({raw.stat().st_size / 1024:.0f} KB)")
    print("Rebuild with: python scripts/run_pipeline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
