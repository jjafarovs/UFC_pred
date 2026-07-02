"""Leakage sanity checks for the data layer.

Phase 1 only produces raw fighter/fight/odds tables, so this file currently
guards the two leakage vectors that already exist at this stage:

1. The `fighters` table must never carry ufcstats' own pre-aggregated career
   stats (SLpM, win totals, etc.) -- those are computed over a fighter's
   *entire* career, including fights that happen after any given historical
   bout, so any feature built from them would leak the future into the past.
2. Every fight must carry a real, parseable event_date -- Phase 2's as-of-date
   filtering (src/features.py) depends on being able to strictly order fights
   in time; a fight with a missing/garbage date can't be placed on that
   timeline safely and must not reach the fights table at all.

As Phase 2 introduces features.py's as-of-date helper and Phase 3 introduces
walk-forward training, add the corresponding tests here (e.g. "a fighter's
rolling form feature for fight X excludes fight X and everything after it").
"""
import sqlite3

import pytest

from src import cleaner
from tests.test_cleaner import _fake_load_raw


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cleaner, "load_raw", _fake_load_raw)
    db_path = tmp_path / "test.db"
    cleaner.run(db_path=db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_fighters_table_has_no_career_aggregate_columns(populated_db):
    cols = {row[1].lower() for row in populated_db.execute("PRAGMA table_info(fighters)")}
    leaky_terms = {"slpm", "sapm", "td_avg", "sub_avg", "win_streak", "wins", "losses"}
    assert not (cols & leaky_terms), (
        "fighters table must not store ufcstats' whole-career aggregate stats; "
        "as-of-date rollups belong in features.py, computed from fight_stats only"
    )


def test_every_stored_fight_has_a_real_event_date(populated_db):
    rows = populated_db.execute("SELECT fight_id, event_date FROM fights").fetchall()
    assert len(rows) > 0
    for row in rows:
        assert row["event_date"] is not None
        # ISO 'YYYY-MM-DD' -- strict format check, not just "not null"
        year, month, day = row["event_date"].split("-")
        assert len(year) == 4 and len(month) == 2 and len(day) == 2


def test_no_scheduled_or_unresolved_fights_reach_the_fights_table(populated_db):
    rows = populated_db.execute("SELECT DISTINCT result FROM fights").fetchall()
    allowed = {"fighter_1", "fighter_2", "draw", "nc"}
    assert {r["result"] for r in rows} <= allowed
