"""Per-card report generator: model probability vs. market vs. edge.

Takes a list of upcoming matchups (fighter ID pairs) that do NOT need to
exist in the fights table -- an upcoming card, by definition, hasn't
happened yet, so this goes through features.matchup_feature_dict directly
rather than the fights-table-backed feature matrix used for training. Both
paths share the exact same feature computation, so a report can never
silently diverge from how the model was trained.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from src import features, market, model

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ufc.db"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


PRODUCTION_CONFIG_PATH = MODELS_DIR / "production.json"


def load_latest_model(name: str = "logistic", models_dir: Path = MODELS_DIR):
    """Loads the most recently saved artifact matching `name` (versioned
    filenames sort chronologically by their timestamp suffix). Useful for
    ad hoc CLI use right after retraining; anything meant to run
    unattended (the dashboard, refresh.py) should use load_production_model
    instead -- "whatever's newest" is fine for a one-off check, but not for
    something that keeps running across retrains you haven't reviewed yet.
    """
    candidates = sorted(models_dir.glob(f"{name}_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No saved model artifacts matching '{name}_*.joblib' in {models_dir}")
    return joblib.load(candidates[-1]), candidates[-1]


def load_production_model(name: str = "logistic", models_dir: Path = MODELS_DIR):
    """Loads the model artifact deliberately pinned as "production" in
    models/production.json (e.g. {"logistic": "logistic_20260702T202618Z.joblib"}),
    falling back to load_latest_model if no pin exists for `name` -- so this
    works out of the box before anyone's created a pin, but a real deployment
    (the dashboard) is never silently using a model nobody reviewed just
    because it happened to be retrained most recently.
    """
    if PRODUCTION_CONFIG_PATH.exists():
        pins = json.loads(PRODUCTION_CONFIG_PATH.read_text())
        pinned_filename = pins.get(name)
        if pinned_filename:
            path = models_dir / pinned_filename
            if not path.exists():
                raise FileNotFoundError(f"production.json pins '{name}' to {path}, which doesn't exist")
            return joblib.load(path), path
    return load_latest_model(name, models_dir)


def set_production_model(name: str, model_path: Path, config_path: Path = PRODUCTION_CONFIG_PATH) -> None:
    """Pins `name` (e.g. 'logistic') to a specific artifact filename in
    models/production.json. The dashboard/refresh.py should be pointed at a
    deliberately reviewed model, not whatever a retrain most recently produced.
    """
    pins = json.loads(config_path.read_text()) if config_path.exists() else {}
    pins[name] = model_path.name
    config_path.write_text(json.dumps(pins, indent=2))


def load_calibration_table(model_path: Path) -> list[dict] | None:
    """Reads the calibration table saved alongside a model artifact (see
    model.py's save_model_artifact) from its sibling .json metadata file.
    Returns None if the metadata file, or the calibration_table key in it,
    is missing -- e.g. an older artifact saved before this was tracked.
    """
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    metrics = json.loads(meta_path.read_text()).get("metrics", {})
    return metrics.get("calibration_table")


def load_feature_columns(model_path: Path) -> list[str]:
    """Reads the exact feature column list a model artifact was trained
    with, from its sibling .json metadata (model.save_model_artifact stores
    this). Falls back to the current model.FEATURE_COLUMNS only if metadata
    is missing entirely (an artifact saved before this was tracked) --
    that's a last resort, not the normal path.

    This exists specifically so a deliberately pinned/older production model
    (see load_production_model) keeps working correctly even after
    model.FEATURE_COLUMNS later gains new features, rather than crashing
    with scikit-learn's "feature names unseen at fit time" error. Caught in
    practice, not hypothetically: production.json was pinned to a model
    trained before diff_total_prior_fights/diff_finish_rate/
    diff_times_finished_rate existed (they didn't validate in the backtest,
    so that older model was deliberately kept as production), and the
    dashboard broke the moment those got added to FEATURE_COLUMNS -- because
    build_card_report was building its feature vector from the current
    code's column list, not from what the loaded model actually expects.
    """
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return model.FEATURE_COLUMNS
    meta = json.loads(meta_path.read_text())
    return meta.get("feature_columns", model.FEATURE_COLUMNS)


def fighter_display_name(conn: sqlite3.Connection, fighter_id: str) -> str:
    row = conn.execute("SELECT name FROM fighters WHERE fighter_id = ?", (fighter_id,)).fetchone()
    return row[0] if row else fighter_id


def _confidence_label(
    f1_n_prior: int, f2_n_prior: int, model_prob: float | None = None, calibration_table: list[dict] | None = None
) -> str:
    """Combines two independent signals, not just one heuristic: (1) does
    this specific matchup have real fight history to compute features from
    (if either side has zero tracked prior fights, most of the feature
    vector is null/imputed -- see the Phase 3 finding on sparse rolling
    features, no amount of calibration evidence rescues that), and (2) has
    this predicted probability actually been validated on enough held-out
    examples in the saved model's own calibration table (model.py's
    calibration_table, persisted via save_model_artifact) -- a probability
    near 0.5 backed by 1,800 held-out fights is a very different claim than
    the same number backed by 3.

    Falls back to a feature-completeness-only signal when no calibration
    table is available (e.g. an older model artifact) -- documented as
    weaker, not treated as equally rigorous.
    """
    if min(f1_n_prior, f2_n_prior) == 0:
        return "low"
    if model_prob is None or not calibration_table:
        return "medium" if min(f1_n_prior, f2_n_prior) < 3 else "high"

    bucket_n = next(
        (b["n"] for b in calibration_table if b["bin_low"] <= model_prob <= b["bin_high"]),
        0,
    )
    if bucket_n >= 100:
        return "high"
    if bucket_n >= 20:
        return "medium"
    return "low"


def build_card_report(
    conn: sqlite3.Connection,
    fitted_model,
    matchups: list[tuple[str, str]],
    as_of_date: str,
    odds_type: str = "live",
    n: int = 5,
    calibration_table: list[dict] | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """One row per matchup: model probability, de-vigged market probability
    (if odds are available for it), edge, and a confidence label.

    Market probability is looked up BEFORE prediction, not just after --
    it's one of model.FEATURE_COLUMNS now (see features.market_prob_feature
    for why), so the model needs it as an input, not only as a downstream
    number to compare against. For a genuinely upcoming matchup this uses
    whatever the CURRENT line is (`odds_type='live'` by default), which is
    the real-world equivalent of what a live prediction would have access
    to -- see the train/production mismatch noted in
    features.market_prob_feature's docstring (training uses the eventual
    CLOSING line, which is typically sharper than a live pre-fight price).

    `feature_columns` should almost always come from `load_feature_columns
    (model_path)`, not be left as the default -- it must match exactly what
    `fitted_model` was actually trained with, which can lag behind the
    current code's `model.FEATURE_COLUMNS` for a deliberately pinned older
    model (see load_feature_columns's docstring for a real case where
    skipping this broke the dashboard).
    """
    feature_columns = feature_columns if feature_columns is not None else model.FEATURE_COLUMNS
    # Built ONCE per report (a global computation, not a per-matchup one) --
    # see features.build_elo_ratings's docstring on why it isn't threaded
    # through fights_before() like everything else here.
    elo_timeline = features.build_elo_ratings(conn)
    rows = []
    for fighter_1_id, fighter_2_id in matchups:
        market_probs = market.market_probabilities_for_matchup(conn, fighter_1_id, fighter_2_id, odds_type=odds_type)
        market_prob = market_probs.get(fighter_1_id)

        feat = features.matchup_feature_dict(conn, fighter_1_id, fighter_2_id, as_of_date, n=n, elo_timeline=elo_timeline)
        feat["market_prob_fighter_1"] = market_prob
        X = pd.DataFrame([feat])[feature_columns].astype(float)
        model_prob = float(fitted_model.predict_proba(X)[0, 1])

        edge = market.compute_edge(model_prob, market_prob) if market_prob is not None else None

        rows.append(
            {
                "fighter_1": fighter_display_name(conn, fighter_1_id),
                "fighter_2": fighter_display_name(conn, fighter_2_id),
                "model_prob_fighter_1": model_prob,
                "market_prob_fighter_1": market_prob,
                "edge_fighter_1": edge,
                "confidence": _confidence_label(
                    feat["f1_n_prior_fights"], feat["f2_n_prior_fights"], model_prob, calibration_table
                ),
                # Exposed for strategy.bet_signal's REFINED_RULE (min_n_prior
                # filter) -- not just an internal detail of the confidence
                # label anymore, so it needs to be a real column here.
                "f1_n_prior_fights": feat["f1_n_prior_fights"],
                "f2_n_prior_fights": feat["f2_n_prior_fights"],
                # Raw (not de-vigged) decimal odds -- needed by strategy.bet_signal's
                # REFINED_PLUS_RULE odds-ceiling filter, which gates on an actual
                # payout price rather than a probability.
                "fighter_1_decimal_odds": market.average_decimal_odds_for_matchup(conn, fighter_1_id, odds_type=odds_type),
                "fighter_2_decimal_odds": market.average_decimal_odds_for_matchup(conn, fighter_2_id, odds_type=odds_type),
                # Raw stance strings (distinct from the model's own boolean
                # same_stance FEATURE) -- needed by strategy.bet_signal's
                # ELO_RULE cross-stance filter, which needs to know WHICH two
                # stances, not just whether they match.
                "fighter_1_stance": features.fighter_physical_features(conn, fighter_1_id, as_of_date)["stance"],
                "fighter_2_stance": features.fighter_physical_features(conn, fighter_2_id, as_of_date)["stance"],
            }
        )
    return pd.DataFrame(rows)


def build_historical_card_report(
    conn: sqlite3.Connection,
    fitted_model,
    fights: list,
    n: int = 5,
    calibration_table: list[dict] | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Like build_card_report, but for fights that have ALREADY happened and
    are in the `fights` table -- for the dashboard's "Past Cards" view.

    Two differences from build_card_report, both because the fight already
    has a real fight_id and a known outcome: (1) market probability comes
    from market.market_probabilities_for_fight (the verified CLOSING line,
    odds_type='close') rather than market_probabilities_for_matchup's
    fight_id-less live-line lookup -- there's no reason to use a live line
    for something that's already been settled; (2) the actual result is
    included, so a past card's picks can be checked for correctness.

    `fights` is a list of sqlite3.Row from the `fights` table (needs
    fight_id, fighter_1_id, fighter_2_id, event_date, result, winner_id).
    """
    feature_columns = feature_columns if feature_columns is not None else model.FEATURE_COLUMNS
    elo_timeline = features.build_elo_ratings(conn)
    rows = []
    for fight in fights:
        fighter_1_id, fighter_2_id = fight["fighter_1_id"], fight["fighter_2_id"]
        as_of_date = fight["event_date"]
        market_probs = market.market_probabilities_for_fight(conn, fight["fight_id"], odds_type="close")
        market_prob = market_probs.get(fighter_1_id)

        feat = features.matchup_feature_dict(conn, fighter_1_id, fighter_2_id, as_of_date, n=n, elo_timeline=elo_timeline)
        feat["market_prob_fighter_1"] = market_prob
        X = pd.DataFrame([feat])[feature_columns].astype(float)
        model_prob = float(fitted_model.predict_proba(X)[0, 1])

        edge = market.compute_edge(model_prob, market_prob) if market_prob is not None else None

        rows.append(
            {
                "fight_id": fight["fight_id"],
                "fighter_1_id": fighter_1_id,
                "fighter_2_id": fighter_2_id,
                "fighter_1": fighter_display_name(conn, fighter_1_id),
                "fighter_2": fighter_display_name(conn, fighter_2_id),
                "model_prob_fighter_1": model_prob,
                "market_prob_fighter_1": market_prob,
                "edge_fighter_1": edge,
                "confidence": _confidence_label(
                    feat["f1_n_prior_fights"], feat["f2_n_prior_fights"], model_prob, calibration_table
                ),
                "f1_n_prior_fights": feat["f1_n_prior_fights"],
                "f2_n_prior_fights": feat["f2_n_prior_fights"],
                "fighter_1_decimal_odds": market.average_decimal_odds_for_fight(conn, fight["fight_id"], fighter_1_id, odds_type="close"),
                "fighter_2_decimal_odds": market.average_decimal_odds_for_fight(conn, fight["fight_id"], fighter_2_id, odds_type="close"),
                "fighter_1_stance": features.fighter_physical_features(conn, fighter_1_id, as_of_date)["stance"],
                "fighter_2_stance": features.fighter_physical_features(conn, fighter_2_id, as_of_date)["stance"],
                "result": fight["result"],
                "winner_id": fight["winner_id"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-card model-vs-market report")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--model", default="logistic")
    parser.add_argument("--odds-type", default="live", choices=["open", "close", "live"])
    parser.add_argument("--as-of-date", default=None, help="ISO date; defaults to today")
    parser.add_argument(
        "--matchup",
        action="append",
        required=True,
        metavar="FIGHTER1_ID:FIGHTER2_ID",
        help="Repeatable, one per fight on the card, e.g. --matchup abc123:def456",
    )
    args = parser.parse_args()

    as_of_date = args.as_of_date or date.today().isoformat()
    matchups = [tuple(m.split(":", 1)) for m in args.matchup]

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    fitted_model, model_path = load_latest_model(args.model)
    calibration_table = load_calibration_table(model_path)
    feature_columns = load_feature_columns(model_path)
    print(f"report.py: using model {model_path.name}, as of {as_of_date}")

    report = build_card_report(
        conn, fitted_model, matchups, as_of_date, odds_type=args.odds_type,
        calibration_table=calibration_table, feature_columns=feature_columns,
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
