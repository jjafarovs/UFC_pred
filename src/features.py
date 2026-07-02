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
            f.fight_id, f.event_date, f.winner_id, f.result,
            fs_self.sig_str_landed AS self_sig_landed, fs_self.sig_str_attempted AS self_sig_attempted,
            fs_self.takedowns_landed AS self_td_landed, fs_self.takedowns_attempted AS self_td_attempted,
            fs_opp.sig_str_landed AS opp_sig_landed, fs_opp.sig_str_attempted AS opp_sig_attempted,
            fs_opp.takedowns_landed AS opp_td_landed, fs_opp.takedowns_attempted AS opp_td_attempted
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

    sig_str_def = _safe_ratio(opp_sig_landed, opp_sig_attempted)
    td_def = _safe_ratio(opp_td_landed, opp_td_attempted)

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
        "same_stance": (
            f1_phys["stance"] == f2_phys["stance"]
            if f1_phys["stance"] and f2_phys["stance"]
            else None
        ),
    }


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
