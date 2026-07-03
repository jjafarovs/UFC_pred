"""refresh.py is pure sequencing over already-tested CLIs -- these tests
pin the call order/arguments via a mocked subprocess.run, not the
underlying fetch/clean/train logic (covered elsewhere).
"""
import subprocess
from unittest.mock import patch

import pytest

from src import refresh


def test_refresh_upcoming_fetches_then_cleans():
    with patch("src.refresh.subprocess.run") as mock_run:
        refresh.refresh_upcoming()
    calls = [call.args[0] for call in mock_run.call_args_list]
    assert any("--upcoming" in c and "--with-live-odds" in c for c in calls)
    assert any(c[-1] == "src.cleaner" for c in calls)
    # fetch must happen before clean -- clean reads what fetch just wrote
    fetch_idx = next(i for i, c in enumerate(calls) if "--upcoming" in c)
    clean_idx = next(i for i, c in enumerate(calls) if c[-1] == "src.cleaner")
    assert fetch_idx < clean_idx


def test_refresh_full_retrains_both_models_by_default():
    with patch("src.refresh.subprocess.run") as mock_run:
        refresh.refresh_full(retrain=True)
    calls = [call.args[0] for call in mock_run.call_args_list]
    assert any("--with-odds" in c for c in calls)
    assert any(c[-1] == "src.features" for c in calls)
    assert any("logistic" in c for c in calls)
    assert any("gbm" in c for c in calls)


def test_refresh_full_skips_retraining_when_disabled():
    with patch("src.refresh.subprocess.run") as mock_run:
        refresh.refresh_full(retrain=False)
    calls = [call.args[0] for call in mock_run.call_args_list]
    assert not any("logistic" in c or "gbm" in c for c in calls)
    assert any(c[-1] == "src.features" for c in calls)


def test_refresh_full_raises_if_a_step_fails():
    with patch("src.refresh.subprocess.run", side_effect=subprocess.CalledProcessError(1, "x")):
        with pytest.raises(subprocess.CalledProcessError):
            refresh.refresh_full()
