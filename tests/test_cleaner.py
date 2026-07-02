import sqlite3
from pathlib import Path

import pytest

from src import cleaner

FIXTURE_RAW = {
    "fighters": [
        {
            "fighter_id": "f1",
            "name": "Fighter One",
            "nickname": "The One",
            "height_in": 70.0,
            "reach_in": 72.0,
            "stance": "Orthodox",
            "dob_raw": "Mar 05, 1993",
            "source_url": "http://ufcstats.com/fighter-details/f1",
        },
        {
            "fighter_id": "f2",
            "name": "Fighter Two",
            "nickname": None,
            "height_in": 68.0,
            "reach_in": 70.0,
            "stance": "Southpaw",
            "dob_raw": None,  # some fighters have no listed DOB
            "source_url": "http://ufcstats.com/fighter-details/f2",
        },
    ],
    "events": [
        {
            "event_id": "e1",
            "name": "UFC Fixture Night",
            "event_date_raw": "June 27, 2026",
            "location": "Las Vegas, Nevada, USA",
            "source_url": "http://ufcstats.com/event-details/e1",
        },
        {
            "event_id": "e2",
            "name": "UFC Bad Date Night",
            "event_date_raw": "Not A Real Date",
            "location": "Nowhere",
            "source_url": "http://ufcstats.com/event-details/e2",
        },
    ],
    "fights": [
        {
            "fight_id": "fight1",
            "event_id": "e1",
            "result": "fighter_1",
            "winner_id": "f1",
            "fighter_1_id": "f1",
            "fighter_2_id": "f2",
            "weight_class": "Lightweight",
            "title_fight": False,
            "scheduled_rounds": 3,
            "method": "KO/TKO",
            "method_detail": None,
            "end_round": 2,
            "end_time_sec": 15,
            "referee": "Marc Goddard",
            "source_url": "http://ufcstats.com/fight-details/fight1",
            "stats": {
                "fighter_1_total": {
                    "knockdowns": 1, "sig_str_landed": 25, "sig_str_attempted": 39,
                    "total_str_landed": 28, "total_str_attempted": 43,
                    "takedowns_landed": 2, "takedowns_attempted": 2,
                    "sub_attempts": 0, "reversals": 0, "control_time_sec": 42,
                },
                "fighter_2_total": {
                    "knockdowns": 0, "sig_str_landed": 22, "sig_str_attempted": 52,
                    "total_str_landed": 26, "total_str_attempted": 56,
                    "takedowns_landed": 0, "takedowns_attempted": 0,
                    "sub_attempts": 0, "reversals": 0, "control_time_sec": 6,
                },
                "per_round": [
                    {"fighter_idx": 1, "round": 1, "knockdowns": 0, "sig_str_landed": 10,
                     "sig_str_attempted": 15, "total_str_landed": 10, "total_str_attempted": 16,
                     "takedowns_landed": 1, "takedowns_attempted": 1, "sub_attempts": 0,
                     "reversals": 0, "control_time_sec": 30},
                    {"fighter_idx": 2, "round": 1, "knockdowns": 0, "sig_str_landed": 8,
                     "sig_str_attempted": 20, "total_str_landed": 8, "total_str_attempted": 20,
                     "takedowns_landed": 0, "takedowns_attempted": 0, "sub_attempts": 0,
                     "reversals": 0, "control_time_sec": 2},
                ],
            },
        },
        {
            # Not yet fought -- must be dropped, never written as a fake result.
            "fight_id": "fight_scheduled",
            "event_id": "e1",
            "result": "scheduled",
        },
        {
            # References an event whose date failed to parse -- must be dropped too,
            # since a fight with no event_date can't be placed in walk-forward order.
            "fight_id": "fight_bad_event",
            "event_id": "e2",
            "result": "fighter_1",
            "winner_id": "f1",
            "fighter_1_id": "f1",
            "fighter_2_id": "f2",
        },
    ],
}


_RAW_NAME_MAP = {"fighters": "fighters", "events_detail": "events", "fights": "fights"}


def _fake_load_raw(name):
    if name not in _RAW_NAME_MAP:
        raise FileNotFoundError(name)  # mirrors load_raw()'s real not-fetched-yet behavior
    return FIXTURE_RAW[_RAW_NAME_MAP[name]]


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cleaner, "load_raw", _fake_load_raw)
    db_path = tmp_path / "test.db"
    cleaner.run(db_path=db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_event_date_parses_to_iso(populated_db):
    row = populated_db.execute("SELECT event_date FROM events WHERE event_id='e1'").fetchone()
    assert row["event_date"] == "2026-06-27"


def test_event_with_unparseable_date_is_skipped(populated_db):
    row = populated_db.execute("SELECT * FROM events WHERE event_id='e2'").fetchone()
    assert row is None


def test_dob_parses_and_missing_dob_is_null(populated_db):
    f1 = populated_db.execute("SELECT dob FROM fighters WHERE fighter_id='f1'").fetchone()
    f2 = populated_db.execute("SELECT dob FROM fighters WHERE fighter_id='f2'").fetchone()
    assert f1["dob"] == "1993-03-05"
    assert f2["dob"] is None


def test_scheduled_fight_is_not_written(populated_db):
    row = populated_db.execute("SELECT * FROM fights WHERE fight_id='fight_scheduled'").fetchone()
    assert row is None


def test_fight_with_unresolvable_event_date_is_dropped(populated_db):
    row = populated_db.execute("SELECT * FROM fights WHERE fight_id='fight_bad_event'").fetchone()
    assert row is None


def test_completed_fight_written_with_denormalized_event_date(populated_db):
    row = populated_db.execute("SELECT * FROM fights WHERE fight_id='fight1'").fetchone()
    assert row is not None
    assert row["event_date"] == "2026-06-27"
    assert row["winner_id"] == "f1"
    assert row["result"] == "fighter_1"


def test_fight_stats_written_for_totals_and_rounds(populated_db):
    rows = populated_db.execute(
        "SELECT * FROM fight_stats WHERE fight_id='fight1' ORDER BY fighter_id, round"
    ).fetchall()
    # 2 fighters x (1 total row + 1 per-round row) = 4
    assert len(rows) == 4
    total_f1 = next(r for r in rows if r["fighter_id"] == "f1" and r["round"] == 0)
    assert total_f1["sig_str_landed"] == 25
    assert total_f1["control_time_sec"] == 42
