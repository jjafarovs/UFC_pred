"""Walk-forward backtest engine: rolling retrain, ROI/Kelly sizing, drawdown.

This is where the project's one hard rule gets enforced at the *training*
level, not just the per-row feature level. features.py already guarantees
each fight's own inputs are as-of-date safe; this module additionally
guarantees the *model* predicting a fold of fights was only ever fit on
fights strictly before that fold's earliest date. model.py's
chronological_split is a single diagnostic split (good for a quick check);
this is the real thing -- an expanding window that retrains every fold and
rolls forward, matching how the model would actually have been used live.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src import market, model

FEATURE_MATRIX_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "feature_matrix.parquet"
DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ufc.db"
BACKTESTS_DIR = Path(__file__).resolve().parent.parent / "backtests"


def walk_forward_predictions(
    df: pd.DataFrame,
    model_type: str = "logistic",
    calibration: str = "sigmoid",
    min_train_size: int = 100,
    fold_size: int = 20,
    calib_frac: float = 0.2,
    model_kwargs: dict | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Expanding-window walk-forward: for each fold of `fold_size` fights (in
    event_date order), trains + calibrates on ALL fights strictly before that
    fold's earliest date, predicts the fold, then grows the window and moves
    on. The training window only ever extends forward in time -- it never
    includes a fight from the fold being predicted or anything after it.

    A fold is skipped (not silently included with bad data) if its training
    window is too small to split into a meaningful train/calibration pair, or
    if the calibration slice ends up single-class (can happen on tiny
    windows) -- in that case the fold falls back to the base model's raw,
    uncalibrated probabilities rather than failing the whole run.

    `model_kwargs` (e.g. `{"C": 0.3}` for logistic, `{"max_depth": 3,
    "l2_regularization": 1.0}` for gbm) is applied identically to every
    fold's freshly-retrained model -- hyperparameters are tuned once, up
    front, on data strictly before this backtest's window (see the
    tuning script referenced in the README), not re-tuned per fold, which
    would be far more expensive and isn't what "walk-forward" is meant to
    validate here. `feature_columns` defaults to model.FEATURE_COLUMNS if
    not given, same as make_xy.
    """
    df = df.sort_values("event_date").reset_index(drop=True)
    model_kwargs = model_kwargs or {}
    predictions = []

    for fold_start in range(min_train_size, len(df), fold_size):
        fold_end = min(fold_start + fold_size, len(df))
        train_window = df.iloc[:fold_start]
        fold_df = df.iloc[fold_start:fold_end]

        n_calib = max(int(len(train_window) * calib_frac), 1)
        n_train = len(train_window) - n_calib
        if n_train < 10 or n_calib < 5:
            continue

        train_df = train_window.iloc[:n_train]
        calib_df = train_window.iloc[n_train:]
        assert train_df["event_date"].max() <= fold_df["event_date"].min(), (
            "walk-forward invariant violated: training window reached into the fold being predicted"
        )

        xy_kwargs = {"feature_columns": feature_columns} if feature_columns is not None else {}
        X_train, y_train = model.make_xy(train_df, **xy_kwargs)
        X_fold, _ = model.make_xy(fold_df, **xy_kwargs)

        base = (
            model.build_logistic_pipeline(**model_kwargs)
            if model_type == "logistic"
            else model.build_gradient_boosting_model(**model_kwargs)
        )
        base.fit(X_train, y_train)

        X_calib, y_calib = model.make_xy(calib_df, **xy_kwargs)
        if y_calib.nunique() < 2:
            proba = base.predict_proba(X_fold)[:, 1]
        else:
            calibrated = model.calibrate_model(base, X_calib, y_calib, method=calibration)
            proba = calibrated.predict_proba(X_fold)[:, 1]

        fold_result = fold_df[
            ["fight_id", "event_date", "fighter_1_id", "fighter_2_id", "label_fighter_1_win"]
        ].copy()
        fold_result["model_prob_fighter_1"] = proba
        fold_result["train_window_size"] = n_train
        predictions.append(fold_result)

    if not predictions:
        raise ValueError(
            f"No folds produced (need >= {min_train_size} rows before the first fold, "
            f"each with a large-enough calibration slice) -- dataset has {len(df)} rows"
        )
    return pd.concat(predictions, ignore_index=True)


def walk_forward_metrics(predictions: pd.DataFrame) -> dict:
    y = predictions["label_fighter_1_win"].to_numpy()
    proba = predictions["model_prob_fighter_1"].to_numpy()
    metrics = {
        "n": len(y),
        "accuracy": float(((proba >= 0.5).astype(int) == y).mean()),
        "brier_score": float(brier_score_loss(y, proba)),
    }
    metrics["log_loss"] = float(log_loss(y, proba, labels=[0, 1])) if len(set(y)) > 1 else None
    return metrics


def _average_decimal_odds(conn: sqlite3.Connection, fight_id: str, fighter_id: str, odds_type: str) -> float | None:
    row = conn.execute(
        "SELECT AVG(decimal_odds) FROM odds WHERE fight_id = ? AND fighter_id = ? AND odds_type = ?",
        (fight_id, fighter_id, odds_type),
    ).fetchone()
    return row[0]


def attach_market_data(conn: sqlite3.Connection, predictions: pd.DataFrame, odds_type: str = "close") -> pd.DataFrame:
    """Joins each walk-forward prediction with de-vigged market probability
    and both fighters' average decimal odds (needed to size/settle a bet).
    Fights with no two-sided odds recorded are dropped -- can't compute an
    edge or a payout without a real price, and silently defaulting one would
    fabricate a market that was never actually there.
    """
    rows = []
    for _, pred_row in predictions.iterrows():
        fighter_1_id, fighter_2_id = pred_row["fighter_1_id"], pred_row["fighter_2_id"]
        market_probs = market.market_probabilities_for_fight(conn, pred_row["fight_id"], odds_type=odds_type)
        if fighter_1_id not in market_probs or fighter_2_id not in market_probs:
            continue
        f1_decimal = _average_decimal_odds(conn, pred_row["fight_id"], fighter_1_id, odds_type)
        f2_decimal = _average_decimal_odds(conn, pred_row["fight_id"], fighter_2_id, odds_type)
        if f1_decimal is None or f2_decimal is None:
            continue

        r = pred_row.to_dict()
        r["market_prob_fighter_1"] = market_probs[fighter_1_id]
        r["fighter_1_decimal_odds"] = f1_decimal
        r["fighter_2_decimal_odds"] = f2_decimal
        r["edge_fighter_1"] = market.compute_edge(r["model_prob_fighter_1"], r["market_prob_fighter_1"])
        rows.append(r)
    return pd.DataFrame(rows)


def simulate_bets(
    df_with_market: pd.DataFrame,
    edge_threshold: float = 0.05,
    strategy: str = "flat",
    flat_stake: float = 1.0,
    kelly_fraction: float = 0.25,
    starting_bankroll: float = 100.0,
) -> pd.DataFrame:
    """Walks fights in chronological order and places at most one bet per
    fight, on whichever side clears `edge_threshold` (never "always bet the
    favorite" -- a fight where neither side clears the threshold gets no bet
    at all, which is the entire point of edge-based flagging over a plain
    predictor).

    Kelly stake uses the bookmaker's actual (vigged) payout odds, not the
    de-vigged market probability used for the edge gate -- those are
    deliberately different numbers (edge asks "do we disagree with a fair
    market", Kelly asks "how much to stake against the real, vigged price"),
    so a fight can clear the edge threshold while still working out to a
    slightly negative or near-zero Kelly fraction; that fraction is clamped
    at 0 (never a negative stake) rather than treated as impossible.
    """
    if strategy not in ("flat", "kelly"):
        raise ValueError(f"unknown strategy: {strategy}")

    bets = []
    bankroll = starting_bankroll
    for _, row in df_with_market.sort_values("event_date").iterrows():
        edge_f1 = row["edge_fighter_1"]
        edge_f2 = -edge_f1

        if edge_f1 >= edge_threshold and edge_f1 >= edge_f2:
            side = "fighter_1"
            model_p, decimal_odds, won, edge = (
                row["model_prob_fighter_1"],
                row["fighter_1_decimal_odds"],
                row["label_fighter_1_win"] == 1,
                edge_f1,
            )
        elif edge_f2 >= edge_threshold:
            side = "fighter_2"
            model_p, decimal_odds, won, edge = (
                1 - row["model_prob_fighter_1"],
                row["fighter_2_decimal_odds"],
                row["label_fighter_1_win"] == 0,
                edge_f2,
            )
        else:
            continue

        b = decimal_odds - 1
        if strategy == "flat":
            stake = flat_stake
        else:
            kelly_f = max((b * model_p - (1 - model_p)) / b, 0) if b > 0 else 0.0
            stake = kelly_fraction * kelly_f * bankroll
        stake = min(stake, bankroll)

        payout = stake * b if won else -stake
        bankroll += payout

        bets.append(
            {
                "fight_id": row["fight_id"],
                "event_date": row["event_date"],
                "side": side,
                "model_prob": model_p,
                "edge": edge,
                "decimal_odds": decimal_odds,
                "stake": stake,
                "won": won,
                "payout": payout,
                "bankroll_after": bankroll,
            }
        )
    return pd.DataFrame(bets)


def backtest_summary(bets: pd.DataFrame, starting_bankroll: float) -> dict:
    """ROI, hit rate, AND max drawdown/variance -- a headline average ROI
    without those is exactly the kind of number this project exists to be
    skeptical of.
    """
    if bets.empty:
        return {"n_bets": 0}

    total_staked = float(bets["stake"].sum())
    total_payout = float(bets["payout"].sum())
    equity = pd.concat([pd.Series([starting_bankroll]), bets["bankroll_after"]], ignore_index=True)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    per_bet_roi = bets["payout"] / bets["stake"]

    return {
        "n_bets": len(bets),
        "hit_rate": float(bets["won"].mean()),
        "total_staked": total_staked,
        "total_payout": total_payout,
        "roi": total_payout / total_staked if total_staked > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "stdev_per_bet_roi": float(per_bet_roi.std()) if len(bets) > 1 else 0.0,
        "final_bankroll": float(equity.iloc[-1]),
    }


def bootstrap_roi_ci(bets: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """95% bootstrap confidence interval on ROI, by resampling per-bet
    (stake, payout) pairs with replacement. A point-estimate ROI on a few
    hundred bets can look dramatic in either direction purely from variance
    -- this is what actually distinguishes "real edge" from "noise that
    looks like edge until you check." If the interval includes 0, the ROI
    is not distinguishable from no edge at all at this sample size.

    Concrete case this caught in practice (see README, Walk-forward
    backtest): an edge-threshold sweep showed GBM's ROI apparently climbing
    to +28.9% at a stricter threshold (75 bets) -- looked like a real
    signal getting purer under a harder filter. This function's 95% CI for
    that exact bucket is [-13.4%, +74.7%]: consistent with pure noise, not
    evidence of skill. Every threshold/model combination tested had the
    same problem -- none excluded zero.

    Seeded by default (unlike this project's usual "don't fake determinism"
    stance elsewhere) specifically so a reported CI is reproducible on
    rerun, matching the project's model-artifact-versioning philosophy: a
    backtest number should always be exactly reproducible, not "roughly
    the same if you run it again."
    """
    if bets.empty:
        return {"n": 0, "roi_low": None, "roi_median": None, "roi_high": None}
    stakes = bets["stake"].to_numpy()
    payouts = bets["payout"].to_numpy()
    n = len(bets)
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rois[i] = payouts[idx].sum() / stakes[idx].sum()
    lo, mid, hi = np.percentile(rois, [2.5, 50, 97.5])
    return {"n": n, "roi_low": float(lo), "roi_median": float(mid), "roi_high": float(hi)}


def plot_equity_curve(bets: pd.DataFrame, starting_bankroll: float, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity = pd.concat([pd.Series([starting_bankroll]), bets["bankroll_after"]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(equity)), equity, marker="o", markersize=3)
    ax.axhline(starting_bankroll, linestyle="--", color="gray", label="Starting bankroll")
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Bankroll")
    ax.set_title(title)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest: ROI/Kelly sizing, drawdown, calibration")
    parser.add_argument("--features", type=Path, default=FEATURE_MATRIX_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--model", choices=["logistic", "gbm"], default="logistic")
    parser.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="sigmoid")
    parser.add_argument("--min-train-size", type=int, default=100)
    parser.add_argument("--fold-size", type=int, default=20)
    parser.add_argument("--odds-type", default="close", choices=["open", "close", "live"])
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--strategy", choices=["flat", "kelly"], default="flat")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--starting-bankroll", type=float, default=100.0)
    args = parser.parse_args()

    df = model.load_feature_matrix(args.features)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    predictions = walk_forward_predictions(
        df, model_type=args.model, calibration=args.calibration,
        min_train_size=args.min_train_size, fold_size=args.fold_size,
    )
    print(f"walk-forward predictions: {len(predictions)} fights across "
          f"{predictions['train_window_size'].nunique()} distinct training windows")
    print("walk-forward metrics:", walk_forward_metrics(predictions))

    table = model.calibration_table(predictions["label_fighter_1_win"].to_numpy(), predictions["model_prob_fighter_1"].to_numpy())
    print(table.to_string(index=False))

    with_market = attach_market_data(conn, predictions, odds_type=args.odds_type)
    print(f"{len(with_market)} of {len(predictions)} walk-forward fights have matched two-sided {args.odds_type!r} odds")

    bets = simulate_bets(
        with_market, edge_threshold=args.edge_threshold, strategy=args.strategy,
        kelly_fraction=args.kelly_fraction, starting_bankroll=args.starting_bankroll,
    )
    summary = backtest_summary(bets, args.starting_bankroll)
    print("backtest summary:", summary)

    if not bets.empty:
        ci = bootstrap_roi_ci(bets)
        print(
            f"bootstrap 95% ROI CI: [{ci['roi_low']:+.3f}, {ci['roi_high']:+.3f}] "
            f"(median {ci['roi_median']:+.3f}, n={ci['n']}) -- "
            f"{'excludes zero: real signal' if ci['roi_low'] > 0 or ci['roi_high'] < 0 else 'includes zero: not distinguishable from noise'}"
        )

        plot_path = BACKTESTS_DIR / f"equity_curve_{args.model}_{args.strategy}.png"
        plot_equity_curve(bets, args.starting_bankroll, plot_path, title=f"{args.model} / {args.strategy}")
        print(f"backtest.py: wrote equity curve -> {plot_path}")


if __name__ == "__main__":
    main()
