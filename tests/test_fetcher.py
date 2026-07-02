"""Parsing tests against saved HTML fixtures (no network I/O).

Fixtures were captured from real ufcstats.com pages; expected values below
were cross-checked by hand against that HTML. If ufcstats changes its markup,
these should fail loudly rather than silently parsing garbage.
"""
from pathlib import Path

from src import fetcher

FIXTURES = Path(__file__).parent / "fixtures"


class FixtureClient:
    """Stands in for UFCStatsClient: serves a fixed HTML string regardless of path/params."""

    def __init__(self, html: str):
        self.html = html

    def get(self, path, params=None):
        return self.html


def test_list_completed_events_parses_rows():
    html = (FIXTURES / "events_completed_page.html").read_text()
    events = fetcher.list_completed_events(FixtureClient(html))
    assert len(events) > 10
    first = events[0]
    assert first["event_id"] == "fccb0fee256b7b4d"
    assert "McGregor" in first["name"]
    assert first["event_date_raw"] == "July 11, 2026"
    completed_sample = next(e for e in events if e["event_id"] == "31e1ea6fe6b682f8")
    assert completed_sample["location"] == "Baku, Azerbaijan"


def test_parse_event_extracts_metadata_and_fight_ids():
    html = (FIXTURES / "event_completed.html").read_text()
    event = fetcher.parse_event(FixtureClient(html), "31e1ea6fe6b682f8")
    assert event["event_date_raw"] == "June 27, 2026"
    assert "Fiziev" in event["name"]
    assert "c13dc0cccef263f7" in event["fight_ids"]


def test_parse_fight_completed_extracts_result_and_totals():
    html = (FIXTURES / "fight_completed.html").read_text()
    fight = fetcher.parse_fight(FixtureClient(html), "c13dc0cccef263f7")

    assert fight["result"] == "fighter_1"
    assert fight["winner_id"] == "c814b4c899793af6"
    assert fight["fighter_1_id"] == "c814b4c899793af6"
    assert fight["fighter_2_id"] == "2e7878927067fdca"
    assert fight["method"] == "KO/TKO"
    assert fight["end_round"] == 2
    assert fight["end_time_sec"] == 15
    assert fight["weight_class"] == "Lightweight"
    assert fight["title_fight"] is False

    total_1 = fight["stats"]["fighter_1_total"]
    assert total_1["sig_str_landed"] == 25
    assert total_1["sig_str_attempted"] == 39
    assert total_1["control_time_sec"] == 42

    total_2 = fight["stats"]["fighter_2_total"]
    assert total_2["sig_str_landed"] == 22
    assert total_2["control_time_sec"] == 6

    assert len(fight["stats"]["per_round"]) > 0
    round_1_entries = [r for r in fight["stats"]["per_round"] if r["round"] == 1]
    assert len(round_1_entries) == 2


def test_parse_fight_scheduled_is_not_treated_as_completed():
    """A fight with no W/L flag yet must come back as 'scheduled', never a fake result."""
    html = (FIXTURES / "fight_scheduled.html").read_text()
    fight = fetcher.parse_fight(FixtureClient(html), "989760fa75321d69")
    assert fight["result"] == "scheduled"


def test_parse_fighter_extracts_bio_fields_only():
    html = (FIXTURES / "fighter_detail.html").read_text()
    fighter = fetcher.parse_fighter(FixtureClient(html), "c814b4c899793af6")
    assert fighter["name"] == "Rafael Fiziev"
    assert fighter["nickname"] == "Ataman"
    assert fighter["height_in"] == 68  # 5'8"
    assert fighter["reach_in"] == 71
    assert fighter["stance"] == "Switch"
    assert fighter["dob_raw"] == "Mar 05, 1993"
    # Leakage guard: career-aggregate fields (SLpM, win streaks, etc.) must
    # never appear on the parsed object -- see the docstring in parse_fighter.
    assert "slpm" not in {k.lower() for k in fighter}
