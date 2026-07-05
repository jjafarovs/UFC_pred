"""Builds the per-fight, per-fighter feature matrix -- strictly as-of-date safe.

Every feature builder in this module is required to route its historical
lookups through `fights_before()`. That function is the single place that
enforces `event_date < as_of_date` (strictly before, not <=): the target
fight itself, and anything on or after its date, is structurally excluded
from that fighter's own inputs. Do not add a feature that queries
`fights`/`fight_stats` directly -- go through `fights_before` so the
leakage guard can't be silently bypassed. See tests/test_no_leakage.py for
the tests that pin this behavior.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import market

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ufc.db"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def fights_before(conn: sqlite3.Connection, fighter_id: str, as_of_date: str, n: int | None = None) -> list[sqlite3.Row]:
    """All of `fighter_id`'s fights strictly before `as_of_date` (ISO 'YYYY-MM-DD'),
    most recent first, each row carrying both the fighter's own fight-total
    stats and their opponent's. This is the ONLY function in this module
    (or that should exist anywhere in the codebase) permitted to select from
    fights/fight_stats keyed on a fighter and a date -- every other function
    here calls this one rather than writing its own query, so the
    `event_date < as_of_date` filter can't be forgotten or subtly loosened
    to `<=` in a one-off query.
    """
    query = """
        SELECT
            f.fight_id, f.event_date, f.winner_id, f.result, f.method,
            fs_self.sig_str_landed AS self_sig_landed, fs_self.sig_str_attempted AS self_sig_attempted,
            fs_self.takedowns_landed AS self_td_landed, fs_self.takedowns_attempted AS self_td_attempted,
            fs_self.control_time_sec AS self_control_time_sec,
            fs_opp.sig_str_landed AS opp_sig_landed, fs_opp.sig_str_attempted AS opp_sig_attempted,
            fs_opp.takedowns_landed AS opp_td_landed, fs_opp.takedowns_attempted AS opp_td_attempted,
            fs_opp.control_time_sec AS opp_control_time_sec
        FROM fights f
        JOIN fight_stats fs_self
            ON fs_self.fight_id = f.fight_id AND fs_self.fighter_id = :fighter_id AND fs_self.round = 0
        JOIN fight_stats fs_opp
            ON fs_opp.fight_id = f.fight_id AND fs_opp.fighter_id != :fighter_id AND fs_opp.round = 0
        WHERE (f.fighter_1_id = :fighter_id OR f.fighter_2_id = :fighter_id)
          AND f.event_date < :as_of_date
        ORDER BY f.event_date DESC
    """
    if n is not None:
        query += " LIMIT :n"
    rows = conn.execute(query, {"fighter_id": fighter_id, "as_of_date": as_of_date, "n": n}).fetchall()
    return rows


def _safe_ratio(numer: int | None, denom: int | None) -> float | None:
    if not denom:
        return None
    return numer / denom


def fighter_rolling_features(conn: sqlite3.Connection, fighter_id: str, as_of_date: str, n: int = 5) -> dict:
    """Rolling form/output features over `fighter_id`'s last `n` fights before
    as_of_date. Landed/attempted are summed across the window before taking a
    ratio (rather than averaging per-fight ratios), which is more stable when
    a fight had few attempts.
    """
    rows = fights_before(conn, fighter_id, as_of_date, n=n)
    n_prior = len(rows)

    empty = {
        "n_prior_fights": 0,
        "win_pct": None,
        "current_streak": None,
        "sig_str_acc": None,
        "sig_str_def": None,
        "td_acc": None,
        "td_def": None,
        "days_since_last_fight": None,
        "control_time_pct": None,
    }
    if n_prior == 0:
        return empty

    wins = sum(1 for r in rows if r["winner_id"] == fighter_id)
    win_pct = wins / n_prior

    streak = 0
    for r in rows:  # most-recent-first
        won = r["winner_id"] == fighter_id
        lost = r["result"] in ("fighter_1", "fighter_2") and not won
        if streak == 0:
            streak = 1 if won else (-1 if lost else 0)
            if streak == 0:
                break
        elif (streak > 0 and won) or (streak < 0 and lost):
            streak += 1 if streak > 0 else -1
        else:
            break

    self_sig_landed = sum(r["self_sig_landed"] or 0 for r in rows)
    self_sig_attempted = sum(r["self_sig_attempted"] or 0 for r in rows)
    opp_sig_landed = sum(r["opp_sig_landed"] or 0 for r in rows)
    opp_sig_attempted = sum(r["opp_sig_attempted"] or 0 for r in rows)
    self_td_landed = sum(r["self_td_landed"] or 0 for r in rows)
    self_td_attempted = sum(r["self_td_attempted"] or 0 for r in rows)
    opp_td_landed = sum(r["opp_td_landed"] or 0 for r in rows)
    opp_td_attempted = sum(r["opp_td_attempted"] or 0 for r in rows)
    self_control_time = sum(r["self_control_time_sec"] or 0 for r in rows)
    opp_control_time = sum(r["opp_control_time_sec"] or 0 for r in rows)

    sig_str_def = _safe_ratio(opp_sig_landed, opp_sig_attempted)
    td_def = _safe_ratio(opp_td_landed, opp_td_attempted)
    control_time_pct = _safe_ratio(self_control_time, self_control_time + opp_control_time)

    most_recent_date = datetime.fromisoformat(rows[0]["event_date"]).date()
    as_of = datetime.fromisoformat(as_of_date).date()

    return {
        "n_prior_fights": n_prior,
        "win_pct": win_pct,
        "current_streak": streak,
        "sig_str_acc": _safe_ratio(self_sig_landed, self_sig_attempted),
        "sig_str_def": 1 - sig_str_def if sig_str_def is not None else None,
        "td_acc": _safe_ratio(self_td_landed, self_td_attempted),
        "td_def": 1 - td_def if td_def is not None else None,
        "days_since_last_fight": (as_of - most_recent_date).days,
        "control_time_pct": control_time_pct,
    }


def fighter_physical_features(conn: sqlite3.Connection, fighter_id: str, as_of_date: str) -> dict:
    """Height/reach/stance/age. Not time-varying except age, so no as-of-date
    lookup against fights/fight_stats is needed here -- these come straight
    from the fighters table.
    """
    row = conn.execute(
        "SELECT height_in, reach_in, stance, dob FROM fighters WHERE fighter_id = ?", (fighter_id,)
    ).fetchone()
    if row is None:
        return {"height_in": None, "reach_in": None, "stance": None, "age_years": None}

    age_years = None
    if row["dob"]:
        dob = datetime.fromisoformat(row["dob"]).date()
        as_of = datetime.fromisoformat(as_of_date).date()
        age_years = (as_of - dob).days / 365.25

    return {
        "height_in": row["height_in"],
        "reach_in": row["reach_in"],
        "stance": row["stance"],
        "age_years": age_years,
    }


_FINISH_METHODS = {"KO/TKO", "Submission"}


def fighter_career_features(conn: sqlite3.Connection, fighter_id: str, as_of_date: str) -> dict:
    """Career-long (NOT windowed to last N) aggregates: total prior fights,
    finish rate among wins, and how often the fighter has been finished
    among losses. Distinct from fighter_rolling_features' last-N window on
    purpose -- a 15-fight veteran and a 2-fight prospect can show identical
    last-5-fight stats, and this is exactly the kind of gap the project's
    walk-forward backtest flagged as worth closing with real features rather
    than leaning further on the market probability feature (see README's
    Walk-forward backtest section). Uses fights_before with n=None (the
    full as-of-date-safe history), so it's leak-free by the same
    construction as every other feature here -- not a separate query path.
    """
    rows = fights_before(conn, fighter_id, as_of_date, n=None)
    total_prior_fights = len(rows)
    if total_prior_fights == 0:
        return {"total_prior_fights": 0, "finish_rate": None, "times_finished_rate": None}

    wins = [r for r in rows if r["winner_id"] == fighter_id]
    losses = [r for r in rows if r["result"] in ("fighter_1", "fighter_2") and r["winner_id"] != fighter_id]
    finishes = sum(1 for r in wins if r["method"] in _FINISH_METHODS)
    times_finished = sum(1 for r in losses if r["method"] in _FINISH_METHODS)

    return {
        "total_prior_fights": total_prior_fights,
        "finish_rate": finishes / len(wins) if wins else None,
        "times_finished_rate": times_finished / len(losses) if losses else None,
    }


def matchup_feature_dict(conn: sqlite3.Connection, fighter_1_id: str, fighter_2_id: str, as_of_date: str, n: int = 5) -> dict:
    """The feature computation shared by build_fight_feature_row (training,
    where the matchup is a completed fight already in the DB) and report.py
    (a genuinely upcoming matchup that doesn't need to exist in the fights
    table at all, since it hasn't happened yet). One implementation, so a
    report can never silently diverge from how training features are built.
    """
    f1_roll = fighter_rolling_features(conn, fighter_1_id, as_of_date, n=n)
    f2_roll = fighter_rolling_features(conn, fighter_2_id, as_of_date, n=n)
    f1_phys = fighter_physical_features(conn, fighter_1_id, as_of_date)
    f2_phys = fighter_physical_features(conn, fighter_2_id, as_of_date)
    f1_career = fighter_career_features(conn, fighter_1_id, as_of_date)
    f2_career = fighter_career_features(conn, fighter_2_id, as_of_date)

    def diff(key, d1, d2):
        v1, v2 = d1.get(key), d2.get(key)
        return v1 - v2 if v1 is not None and v2 is not None else None

    return {
        "f1_n_prior_fights": f1_roll["n_prior_fights"],
        "f2_n_prior_fights": f2_roll["n_prior_fights"],
        "diff_win_pct": diff("win_pct", f1_roll, f2_roll),
        "diff_current_streak": diff("current_streak", f1_roll, f2_roll),
        "diff_sig_str_acc": diff("sig_str_acc", f1_roll, f2_roll),
        "diff_sig_str_def": diff("sig_str_def", f1_roll, f2_roll),
        "diff_td_acc": diff("td_acc", f1_roll, f2_roll),
        "diff_td_def": diff("td_def", f1_roll, f2_roll),
        "diff_days_since_last_fight": diff("days_since_last_fight", f1_roll, f2_roll),
        "diff_height_in": diff("height_in", f1_phys, f2_phys),
        "diff_reach_in": diff("reach_in", f1_phys, f2_phys),
        "diff_age_years": diff("age_years", f1_phys, f2_phys),
        "diff_total_prior_fights": diff("total_prior_fights", f1_career, f2_career),
        "diff_finish_rate": diff("finish_rate", f1_career, f2_career),
        "diff_times_finished_rate": diff("times_finished_rate", f1_career, f2_career),
        "diff_control_time_pct": diff("control_time_pct", f1_roll, f2_roll),
        "same_stance": (
            f1_phys["stance"] == f2_phys["stance"]
            if f1_phys["stance"] and f2_phys["stance"]
            else None
        ),
    }


def market_prob_feature(conn: sqlite3.Connection, fight_id: str, fighter_1_id: str, odds_type: str = "close") -> float | None:
    """De-vigged market-implied probability that fighter_1 wins, from this
    fight's own historical odds. None where no odds are matched for this
    fight (currently ~9% coverage across full history -- see README).

    Added as a feature (not just a downstream edge comparison) after
    confirming empirically that the model's raw, uncalibrated predictions
    systematically compress toward 0.5 relative to the market: fights the
    model calls a near-toss-up (0.4-0.6) split into true favorites that
    actually win ~72% of the time and true underdogs that win only ~29% --
    at the SAME predicted probability. No calibration scheme can fix that
    (a 1-D reshaping of the model's own score can't inject information the
    raw features don't have); giving the model direct access to what the
    market already knows can. HistGradientBoostingClassifier's native NaN
    handling makes this safe to add despite the coverage gap: the ~9% of
    rows where it's present still teach the model a strong relationship,
    and rows without it just fall back to the other ten features.

    NOT a leakage risk relative to the fight itself: a closing line is
    contemporaneous with the fight (the market's last price right before it
    starts), not information from after it. It IS a real train/production
    mismatch worth documenting: this trains on the eventual CLOSING line,
    but report.py's live use (a fight that hasn't happened yet) only has
    access to whatever the CURRENT line is (`odds_type='live'`), which may
    be less sharp than what the closing line eventually becomes.
    """
    return market.market_probabilities_for_fight(conn, fight_id, odds_type=odds_type).get(fighter_1_id)


def build_fight_feature_row(conn: sqlite3.Connection, fight: sqlite3.Row, n: int = 5) -> dict | None:
    """One training row for a completed fight: fighter_1-minus-fighter_2
    feature differences, as of the day of the fight, plus the label. Returns
    None for draws/no-contests -- there's no winner to learn from.
    """
    if fight["result"] not in ("fighter_1", "fighter_2"):
        return None

    as_of_date = fight["event_date"]
    row = matchup_feature_dict(conn, fight["fighter_1_id"], fight["fighter_2_id"], as_of_date, n=n)
    row.update(
        {
            "fight_id": fight["fight_id"],
            "event_date": as_of_date,
            "fighter_1_id": fight["fighter_1_id"],
            "fighter_2_id": fight["fighter_2_id"],
            "label_fighter_1_win": 1 if fight["result"] == "fighter_1" else 0,
            "market_prob_fighter_1": market_prob_feature(conn, fight["fight_id"], fight["fighter_1_id"]),
        }
    )
    return row


def build_feature_matrix(conn: sqlite3.Connection, n: int = 5) -> pd.DataFrame:
    """Builds one row per completed fight in the DB, ordered by event_date.
    Each fighter's features are computed as of that fight's own date, so a
    fighter's 3rd fight uses only their 1st and 2nd -- the DB may already
    contain their 4th, 5th, etc. (loaded in any order), and those must not
    leak in. See tests/test_no_leakage.py::test_feature_matrix_* for the
    tests that pin this.
    """
    fights = conn.execute(
        "SELECT fight_id, event_date, fighter_1_id, fighter_2_id, winner_id, result "
        "FROM fights ORDER BY event_date ASC"
    ).fetchall()
    rows = [build_fight_feature_row(conn, f, n=n) for f in fights]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the as-of-date-safe feature matrix")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--n", type=int, default=5, help="Rolling window size (last N fights)")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "feature_matrix.parquet")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    df = build_feature_matrix(conn, n=args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"features: wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
