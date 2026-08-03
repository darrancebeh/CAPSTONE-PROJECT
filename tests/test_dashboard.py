"""Smoke test for the Streamlit application.

The dashboard is the deliverable a reader actually interacts with, so a broken
render is a real defect rather than a cosmetic one. This executes the whole
script headlessly against the cached run and fails on any uncaught exception.
It is skipped when no backtest has been run yet, so a clean checkout still has
a green suite.
"""

from pathlib import Path

import pytest

from volforecast.config import Config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _cached_run_available() -> bool:
    config = Config.load()
    return (config.backtest.forecast_dir / "evaluation_panel.parquet").exists()


requires_cache = pytest.mark.skipif(
    not _cached_run_available(),
    reason="No cached backtest; run scripts/run_pipeline.py first",
)


@pytest.fixture(scope="module")
def app():
    instance = AppTest.from_file(str(APP), default_timeout=300)
    instance.run()
    return instance


@requires_cache
def test_app_renders_without_error(app):
    assert not app.exception
    assert not app.error


@requires_cache
def test_every_panel_is_present(app):
    assert len(app.tabs) == 6
    assert app.title[0].value.startswith("GARCH versus LSTM")


@requires_cache
def test_headline_metrics_are_populated(app):
    labels = {metric.label for metric in app.metric}
    assert "Sessions evaluated" in labels
    assert "Realised volatility" in labels
    assert any(label.startswith("Best ") for label in labels)


@requires_cache
def test_accuracy_table_is_rendered(app):
    assert len(app.dataframe) >= 1
