"""Persists every prediction the dashboard shows, so a losing (or winning)
stretch can be forensically analyzed with real numbers later instead of
reconstructed from memory of fighter names.

Before this module existed, the dashboard was completely stateless -- it
computed Logistic %/GBM %/Avg %/Market %/Edge live and never saved any of
it. That gap became concrete, not hypothetical: a real losing stretch
across three cards couldn't be analyzed at all, because there was no record
of what had actually been predicted for those specific fights at the time.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src import backtest

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ufc.db"


def log_card(conn: sqlite3.Connection, card_report: pd.DataFrame, fights_df: pd.DataFrame, event_row) -> int:
    """Upserts one row per fight in `card_report` (as produced by app.py --
    must already have logistic_prob/gbm_prob/avg_prob/market_prob_fighter_1/
    edge_fighter_1/confidence/refined_signal/refined_plus_signal/elo_signal/
    weight_class/title_fight columns) keyed on (fight_id, today's date) --
    see schema.sql's prediction_log table docstring for why that key, not
    one row per page view. `fights_df` supplies fighter_1_id/fighter_2_id/
    names/fight_id in the same row order as `card_report` (both are built
    from the same `matchups` list in app.py, so positions line up).

    Returns the number of rows upserted.
    """
    today = date.today().isoformat()
    now = datetime.now(timezone.utc).isoformat()
    rows = 0
    for i in range(len(card_report)):
        report_row = card_report.iloc[i]
        fight_row = fights_df.iloc[i]
        conn.execute(
            """
            INSERT INTO prediction_log (
                fight_id, log_date, fighter_1_id, fighter_2_id, fighter_1_name, fighter_2_name,
                event_id, event_name, event_date, weight_class, title_fight,
                logistic_prob_fighter_1, gbm_prob_fighter_1, avg_prob_fighter_1,
                market_prob_fighter_1, edge_fighter_1, confidence,
                refined_signal, refined_plus_signal, elo_signal,
                market_decimal_odds_fighter_1, market_decimal_odds_fighter_2, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fight_id, log_date) DO UPDATE SET
                logistic_prob_fighter_1=excluded.logistic_prob_fighter_1,
                gbm_prob_fighter_1=excluded.gbm_prob_fighter_1,
                avg_prob_fighter_1=excluded.avg_prob_fighter_1,
                market_prob_fighter_1=excluded.market_prob_fighter_1,
                edge_fighter_1=excluded.edge_fighter_1,
                confidence=excluded.confidence,
                refined_signal=excluded.refined_signal,
                refined_plus_signal=excluded.refined_plus_signal,
                elo_signal=excluded.elo_signal,
                market_decimal_odds_fighter_1=excluded.market_decimal_odds_fighter_1,
                market_decimal_odds_fighter_2=excluded.market_decimal_odds_fighter_2,
                logged_at=excluded.logged_at
            """,
            (
                fight_row["fight_id"], today, fight_row["fighter_1_id"], fight_row["fighter_2_id"],
                fight_row.get("fighter_1_name"), fight_row.get("fighter_2_name"),
                event_row["event_id"], event_row.get("event_name"), event_row["event_date"],
                report_row.get("weight_class"),
                int(report_row["title_fight"]) if pd.notna(report_row.get("title_fight")) else None,
                _none_if_nan(report_row.get("logistic_prob")),
                _none_if_nan(report_row.get("gbm_prob")),
                _none_if_nan(report_row.get("avg_prob")),
                _none_if_nan(report_row.get("market_prob_fighter_1")),
                _none_if_nan(report_row.get("edge_fighter_1")),
                report_row.get("confidence"),
                report_row.get("refined_signal"), report_row.get("refined_plus_signal"), report_row.get("elo_signal"),
                _none_if_nan(report_row.get("fighter_1_decimal_odds")),
                _none_if_nan(report_row.get("fighter_2_decimal_odds")),
                now,
            ),
        )
        rows += 1
    conn.commit()
    return rows


def _none_if_nan(x):
    return None if x is None or pd.isna(x) else float(x)


def resolve_report(conn: sqlite3.Connection) -> pd.DataFrame:
    """Joins every logged prediction against `fights` (only fights that have
    since actually happened and been backfilled show up here -- see
    README's Live pipeline section on running `--mode full`) to compute the
    real outcome of every rule's picks. One row per logged fight with a
    known result; a `*_correct` column per rule (None if that rule had no
    signal for this fight), plus the REAL closing decimal odds for whichever
    side each rule picked -- fetched fresh from `odds` (odds_type='close'),
    not from whatever live line happened to be showing when it was logged,
    since the closing line is what actually settles a bet and is what every
    other backtest number in this project is scored against.
    """
    logged = pd.read_sql_query("SELECT * FROM prediction_log", conn)
    if logged.empty:
        return logged

    results = pd.read_sql_query(
        "SELECT fight_id, winner_id, result FROM fights WHERE fight_id IN ({})".format(
            ",".join("?" for _ in logged["fight_id"].unique())
        ),
        conn, params=list(logged["fight_id"].unique()),
    )
    if results.empty:
        return pd.DataFrame()

    merged = logged.merge(results, on="fight_id", how="inner")
    merged = merged[merged["result"].isin(["fighter_1", "fighter_2"])].copy()
    merged["fighter_1_won"] = merged["result"] == "fighter_1"

    for rule_col in ["refined_signal", "refined_plus_signal", "elo_signal"]:
        correct_col = rule_col.replace("_signal", "_correct")
        merged[correct_col] = None
        has_signal = merged[rule_col].notna()
        picked_f1 = merged[rule_col] == "fighter_1"
        merged.loc[has_signal, correct_col] = (
            merged.loc[has_signal, "fighter_1_won"] == picked_f1[has_signal]
        )

    def _picked_decimal_odds(row, rule_col):
        side = row[rule_col]
        if side is None or pd.isna(side):
            return None
        fighter_id = row["fighter_1_id"] if side == "fighter_1" else row["fighter_2_id"]
        return backtest._average_decimal_odds(conn, row["fight_id"], fighter_id, odds_type="close")

    for rule_col in ["refined_signal", "refined_plus_signal", "elo_signal"]:
        odds_col = rule_col.replace("_signal", "_decimal_odds")
        merged[odds_col] = merged.apply(lambda r: _picked_decimal_odds(r, rule_col), axis=1)
    return merged


def summarize(resolved: pd.DataFrame) -> None:
    """Prints hit rate + ROI per rule from resolve_report's output, for
    whichever resolved picks also have a real closing line to settle
    against (a resolved fight can still lack one -- odds coverage was never
    100%, see README's Data sources section).
    """
    if resolved.empty:
        print("No resolved fights yet -- either nothing logged, or none of the logged fights have completed and been backfilled (run `python -m src.refresh --mode full`).")
        return
    for rule_name, prefix in [("Refined", "refined"), ("Refined+", "refined_plus"), ("Elo", "elo")]:
        correct_col, odds_col = f"{prefix}_correct", f"{prefix}_decimal_odds"
        picks = resolved[resolved[correct_col].notna()].copy()
        if picks.empty:
            print(f"{rule_name}: no resolved picks yet")
            continue
        n, wins = len(picks), int(picks[correct_col].sum())
        line = f"{rule_name}: {wins}W-{n-wins}L ({wins/n*100:.1f}% hit rate) over {n} resolved picks"
        priced = picks[picks[odds_col].notna()]
        if not priced.empty:
            payouts = priced.apply(lambda r: (r[odds_col] - 1) if r[correct_col] else -1.0, axis=1)
            line += f", ROI={payouts.sum()/len(priced)*100:+.1f}% ({len(priced)} with a real closing line)"
        print(line)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    resolved = resolve_report(conn)
    summarize(resolved)


if __name__ == "__main__":
    main()
