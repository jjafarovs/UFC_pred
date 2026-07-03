"""Single entry point tying fetch -> clean -> (feature/model rebuild)
together for scheduled, unattended use (cron or similar).

Two modes, because the pieces have very different natural refresh
cadences:
- `upcoming`: current card + live odds. Only useful if re-pulled often --
  a live line moves by the hour as a fight approaches. Cheap (a handful of
  events, a few dozen fighters); safe to run hourly.
- `full`: incremental completed-history fetch, feature rebuild, retrain.
  A new UFC event happens roughly weekly, and fetcher.bootstrap()'s
  incremental design means this only fetches what's new each time (see its
  docstring) -- but it's still slower than the upcoming refresh and has no
  reason to run more than daily.

Deliberately calls the existing, already-tested CLIs via subprocess rather
than re-implementing their logic here -- one source of truth per step, this
module is pure sequencing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def _run(args: list[str]) -> None:
    _log(f"$ {' '.join(args)}")
    subprocess.run(args, check=True)


def refresh_upcoming() -> None:
    """Fast refresh: current upcoming card + live odds, then re-normalize."""
    _log("refresh_upcoming: start")
    _run([sys.executable, "-m", "src.fetcher", "--upcoming", "--with-live-odds"])
    _run([sys.executable, "-m", "src.cleaner"])
    _log("refresh_upcoming: done")


def refresh_full(retrain: bool = True) -> None:
    """Slower refresh: incremental completed-events + odds fetch, feature
    rebuild, and (optionally) retrain both models.
    """
    _log("refresh_full: start")
    _run([sys.executable, "-m", "src.fetcher", "--with-odds"])
    _run([sys.executable, "-m", "src.cleaner"])
    _run([sys.executable, "-m", "src.features"])
    if retrain:
        _run([sys.executable, "-m", "src.model", "--model", "logistic", "--calibration", "sigmoid"])
        _run([sys.executable, "-m", "src.model", "--model", "gbm", "--calibration", "isotonic"])
    _log("refresh_full: done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled refresh entry point (cron-friendly)")
    parser.add_argument("--mode", choices=["upcoming", "full"], required=True)
    parser.add_argument("--no-retrain", action="store_true", help="With --mode full, skip retraining models")
    args = parser.parse_args()

    if args.mode == "upcoming":
        refresh_upcoming()
    else:
        refresh_full(retrain=not args.no_retrain)


if __name__ == "__main__":
    main()
