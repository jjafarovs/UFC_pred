"""American/decimal odds -> de-vigged implied probability, and edge.

De-vigging here uses the basic multiplicative method: take each side's raw
implied probability and normalize the pair so it sums to 1 (a sportsbook's
two-sided line always sums to slightly more than 1 -- that excess is the
"vig"/juice). More sophisticated de-vig methods exist (Shin's method, the
power method, which model favorite/longshot bias differently) but this is
intentionally the simple starting point, matching the project's "start
simple" phasing -- the model side (src/model.py) is also a baseline before
gradient boosting, calibration, etc.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ufc.db"


def implied_prob_from_american(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return -american_odds / (-american_odds + 100)


def implied_prob_from_decimal(decimal_odds: float) -> float:
    return 1 / decimal_odds


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Normalizes two raw implied probabilities (typically summing to >1) so
    they sum to exactly 1, preserving their relative ratio.
    """
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError(f"probabilities must sum to a positive number, got {prob_a} + {prob_b}")
    return prob_a / total, prob_b / total


def _devig_and_average(rows) -> dict[str, float]:
    """Shared core for both lookup functions below: groups (fighter_id,
    sportsbook, decimal_odds) rows by sportsbook, de-vigs each book's
    two-sided line independently, then averages the resulting probability
    per fighter across books -- this smooths single-book noise/errors rather
    than trusting one line. A book that only has one side recorded (a data
    gap, not a real one-sided market) is skipped since there's nothing to
    de-vig against. Returns {} if no book has both sides recorded.
    """
    by_book: dict[str, dict[str, float]] = {}
    for fighter_id, sportsbook, decimal_odds in rows:
        by_book.setdefault(sportsbook, {})[fighter_id] = decimal_odds

    per_fighter_probs: dict[str, list[float]] = {}
    for fighter_odds in by_book.values():
        if len(fighter_odds) != 2:
            continue
        (fid_a, dec_a), (fid_b, dec_b) = fighter_odds.items()
        devigged_a, devigged_b = devig_two_way(
            implied_prob_from_decimal(dec_a), implied_prob_from_decimal(dec_b)
        )
        per_fighter_probs.setdefault(fid_a, []).append(devigged_a)
        per_fighter_probs.setdefault(fid_b, []).append(devigged_b)

    return {fid: sum(probs) / len(probs) for fid, probs in per_fighter_probs.items()}


def market_probabilities_for_fight(
    conn: sqlite3.Connection, fight_id: str, odds_type: str = "close"
) -> dict[str, float]:
    """De-vigged consensus market probability per fighter for one completed
    (or scheduled-and-already-linked) fight already in the fights table.
    """
    rows = conn.execute(
        "SELECT fighter_id, sportsbook, decimal_odds FROM odds WHERE fight_id = ? AND odds_type = ? AND fighter_id IS NOT NULL",
        (fight_id, odds_type),
    ).fetchall()
    return _devig_and_average(rows)


def market_probabilities_for_matchup(
    conn: sqlite3.Connection, fighter_1_id: str, fighter_2_id: str, odds_type: str = "live"
) -> dict[str, float]:
    """Same de-vigged consensus, but keyed on a fighter pair rather than a
    fight_id -- for a genuinely upcoming card, where there's no row in the
    fights table yet (report.py's whole reason for existing) and therefore
    no fight_id for match_and_store_odds to have linked odds against.
    Restricted to fight_id IS NULL rows: those are exactly the odds cleaner.py
    couldn't match to any completed fight, which is what an upcoming
    matchup's odds look like by definition.
    """
    rows = conn.execute(
        "SELECT fighter_id, sportsbook, decimal_odds FROM odds "
        "WHERE fight_id IS NULL AND fighter_id IN (?, ?) AND odds_type = ?",
        (fighter_1_id, fighter_2_id, odds_type),
    ).fetchall()
    return _devig_and_average(rows)


def average_decimal_odds_for_matchup(
    conn: sqlite3.Connection, fighter_id: str, odds_type: str = "live"
) -> float | None:
    """Raw (not de-vigged) average decimal odds for one fighter's side of an
    upcoming matchup -- needed by strategy filters that gate on an actual
    payout price (e.g. an odds ceiling), not a probability. `fight_id IS
    NULL` is what an upcoming matchup's odds look like, same convention as
    market_probabilities_for_matchup.
    """
    row = conn.execute(
        "SELECT AVG(decimal_odds) FROM odds WHERE fight_id IS NULL AND fighter_id = ? AND odds_type = ?",
        (fighter_id, odds_type),
    ).fetchone()
    return row[0]


def average_decimal_odds_for_fight(
    conn: sqlite3.Connection, fight_id: str, fighter_id: str, odds_type: str = "close"
) -> float | None:
    """Same as average_decimal_odds_for_matchup, but for a fight already
    linked to a fight_id (a completed fight, or an upcoming one already
    matched) -- same convention as market_probabilities_for_fight.
    """
    row = conn.execute(
        "SELECT AVG(decimal_odds) FROM odds WHERE fight_id = ? AND fighter_id = ? AND odds_type = ?",
        (fight_id, fighter_id, odds_type),
    ).fetchone()
    return row[0]


def compute_edge(model_prob: float, market_prob: float) -> float:
    """model probability minus de-vigged market probability. Positive means
    the model thinks this fighter is more likely to win than the market
    (de-vigged) implies -- a candidate for a positive-EV flag once backtest.py
    (Phase 5) validates whether that gap is real signal or model noise.
    """
    return model_prob - market_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Print de-vigged market probabilities for a fight")
    parser.add_argument("fight_id")
    parser.add_argument("--odds-type", default="close", choices=["open", "close", "live"])
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    probs = market_probabilities_for_fight(conn, args.fight_id, odds_type=args.odds_type)
    if not probs:
        print(f"No two-sided {args.odds_type!r} odds found for fight {args.fight_id}")
        return
    for fighter_id, prob in probs.items():
        name = conn.execute("SELECT name FROM fighters WHERE fighter_id = ?", (fighter_id,)).fetchone()
        label = name[0] if name else fighter_id
        print(f"{label}: {prob:.3f}")


if __name__ == "__main__":
    main()
