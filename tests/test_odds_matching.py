import sqlite3
from datetime import date

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


def test_normalize_fighter_name_strips_suffix_and_accents():
    assert cleaner.normalize_fighter_name("Khalil Rountree Jr.") == "khalil rountree"
    assert cleaner.normalize_fighter_name("KHALIL ROUNTREE") == "khalil rountree"
    assert cleaner.normalize_fighter_name("José Aldo") == "jose aldo"


def test_match_and_store_odds_links_fight_id_and_fighter_id(populated_db):
    # FIXTURE_RAW's fight1 is Fighter One (f1) vs Fighter Two (f2) on 2026-06-27.
    bfo_events = [
        {
            "event_name": "UFC Fixture Night",
            "event_date_raw": "June 27, 2026",
            "matchups": [
                {
                    "matchup_id": "999",
                    "fighters": [
                        {"fighter_name": "Fighter One", "odds": {"FanDuel": -150}},
                        {"fighter_name": "Fighter Two", "odds": {"FanDuel": 130}},
                    ],
                }
            ],
        }
    ]
    n_matched = cleaner.match_and_store_odds(populated_db, bfo_events)
    assert n_matched == 1

    rows = populated_db.execute("SELECT * FROM odds ORDER BY fighter_name_raw").fetchall()
    assert len(rows) == 2
    f1_row = next(r for r in rows if r["fighter_name_raw"] == "Fighter One")
    assert f1_row["fight_id"] == "fight1"
    assert f1_row["fighter_id"] == "f1"
    assert f1_row["american_odds"] == -150
    assert f1_row["decimal_odds"] == pytest.approx(1.6667, abs=1e-3)
    assert f1_row["odds_type"] == "close"
    assert f1_row["opponent_name_raw"] == "Fighter Two"


def test_match_and_store_odds_skips_unmatched_fighters(populated_db):
    bfo_events = [
        {
            "event_name": "Some Other Promotion",
            "event_date_raw": "June 27, 2026",
            "matchups": [
                {
                    "matchup_id": "111",
                    "fighters": [
                        {"fighter_name": "Nobody Here", "odds": {"FanDuel": -150}},
                        {"fighter_name": "Also Nobody", "odds": {"FanDuel": 130}},
                    ],
                }
            ],
        }
    ]
    n_matched = cleaner.match_and_store_odds(populated_db, bfo_events)
    assert n_matched == 0
    assert populated_db.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 0


def test_parse_last_change_valid_and_invalid():
    assert cleaner.parse_last_change("Jun 28th 2026 13:58 UTC") == "2026-06-28T13:58:00+00:00"
    assert cleaner.parse_last_change("Mar 6th 2016 05:54 UTC") == "2016-03-06T05:54:00+00:00"
    assert cleaner.parse_last_change(None) is None
    assert cleaner.parse_last_change("not a date") is None


def test_derive_odds_type_close_when_last_change_matches_fight_date():
    # UFC 196 real spot-check: fought Mar 5 2016, last change Mar 6 2016 05:54 UTC
    odds_type = cleaner._derive_odds_type("2016-03-05", "2016-03-06T05:54:00+00:00", date(2026, 1, 1))
    assert odds_type == "close"


def test_derive_odds_type_unverified_when_last_change_is_far_after_fight():
    # Line supposedly kept moving 30 days after the fight -- can't trust this as a real close.
    odds_type = cleaner._derive_odds_type("2026-01-01", "2026-01-31T00:00:00+00:00", date(2026, 6, 1))
    assert odds_type == "unverified"


def test_derive_odds_type_falls_back_to_today_heuristic_without_a_timestamp():
    assert cleaner._derive_odds_type("2026-01-01", None, date(2026, 6, 1)) == "close"
    assert cleaner._derive_odds_type("2026-12-01", None, date(2026, 6, 1)) == "live"


def test_match_and_store_odds_verifies_close_via_last_change_timestamp(populated_db):
    # fight1 is on 2026-06-27; a last_change the next UTC day is a real closing signal.
    bfo_events = [
        {
            "event_name": "UFC Fixture Night",
            "event_date_raw": "June 27, 2026",
            "last_change_raw": "Jun 28th 2026 05:00 UTC",
            "matchups": [
                {
                    "matchup_id": "999",
                    "fighters": [
                        {"fighter_name": "Fighter One", "odds": {"FanDuel": -150}},
                        {"fighter_name": "Fighter Two", "odds": {"FanDuel": 130}},
                    ],
                }
            ],
        }
    ]
    cleaner.match_and_store_odds(populated_db, bfo_events)
    row = populated_db.execute("SELECT * FROM odds WHERE fighter_name_raw='Fighter One'").fetchone()
    assert row["odds_type"] == "close"
    assert row["last_change_raw"] == "Jun 28th 2026 05:00 UTC"
    assert row["last_change_utc"] == "2026-06-28T05:00:00+00:00"


def test_match_and_store_odds_marks_unverified_when_last_change_is_implausible(populated_db):
    bfo_events = [
        {
            "event_name": "UFC Fixture Night",
            "event_date_raw": "June 27, 2026",
            "last_change_raw": "Sep 15th 2026 00:00 UTC",  # months after the fight -- suspicious
            "matchups": [
                {
                    "matchup_id": "999",
                    "fighters": [
                        {"fighter_name": "Fighter One", "odds": {"FanDuel": -150}},
                        {"fighter_name": "Fighter Two", "odds": {"FanDuel": 130}},
                    ],
                }
            ],
        }
    ]
    cleaner.match_and_store_odds(populated_db, bfo_events)
    row = populated_db.execute("SELECT * FROM odds WHERE fighter_name_raw='Fighter One'").fetchone()
    assert row["odds_type"] == "unverified"


def test_match_and_store_odds_is_idempotent_on_rerun(populated_db):
    bfo_events = [
        {
            "event_name": "UFC Fixture Night",
            "event_date_raw": "June 27, 2026",
            "matchups": [
                {
                    "matchup_id": "999",
                    "fighters": [
                        {"fighter_name": "Fighter One", "odds": {"FanDuel": -150}},
                        {"fighter_name": "Fighter Two", "odds": {"FanDuel": 130}},
                    ],
                }
            ],
        }
    ]
    cleaner.match_and_store_odds(populated_db, bfo_events)
    cleaner.match_and_store_odds(populated_db, bfo_events)  # re-running with the same data must not duplicate rows
    assert populated_db.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 2
