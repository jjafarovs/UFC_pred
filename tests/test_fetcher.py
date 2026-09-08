"""Parsing tests against saved HTML fixtures (no network I/O).

Fixtures were captured from real ufcstats.com pages; expected values below
were cross-checked by hand against that HTML. If ufcstats changes its markup,
these should fail loudly rather than silently parsing garbage.
"""
from pathlib import Path
from unittest.mock import MagicMock

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


def test_list_upcoming_events_parses_rows():
    html = (FIXTURES / "events_upcoming_page.html").read_text()
    events = fetcher.list_upcoming_events(FixtureClient(html))
    assert len(events) > 3
    first = events[0]
    assert first["event_id"] == "fccb0fee256b7b4d"
    assert "McGregor" in first["name"]


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
    """A fight with no W/L flag yet must come back as 'scheduled', never a fake result,
    but still carries fighter IDs/weight class -- that's exactly what upcoming-card
    discovery needs, and there's no result to report alongside it that would leak.
    """
    html = (FIXTURES / "fight_scheduled.html").read_text()
    fight = fetcher.parse_fight(FixtureClient(html), "989760fa75321d69")
    assert fight["result"] == "scheduled"
    assert fight["fighter_1_id"] == "f4c49976c75c5ab2"
    assert fight["fighter_2_id"] == "150ff4cc642270b9"
    assert "weight_class" in fight


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


class MultiFixtureClient:
    """Routes by path prefix to a fixed HTML string -- for orchestration
    functions (fetch_upcoming_card) that hit several distinct page types in
    one call.
    """

    def __init__(self, responses: dict[str, str]):
        self.responses = responses

    def get(self, path, params=None):
        for prefix, html in self.responses.items():
            if path.startswith(prefix):
                return html
        raise AssertionError(f"MultiFixtureClient has no response configured for {path}")


def test_fetch_upcoming_card_returns_scheduled_fights_and_fighter_ids():
    events_html = (FIXTURES / "events_upcoming_page.html").read_text()
    event_html = (FIXTURES / "event_upcoming.html").read_text()
    fight_html = (FIXTURES / "fight_scheduled.html").read_text()

    client = MultiFixtureClient(
        {
            "/statistics/events/upcoming": events_html,
            "/event-details/fccb0fee256b7b4d": event_html,
            "/fight-details/": fight_html,
        }
    )
    card = fetcher.fetch_upcoming_card(client, max_events=1)

    assert len(card["events"]) == 1
    assert card["events"][0]["event_id"] == "fccb0fee256b7b4d"
    assert len(card["fights"]) == 14  # every fight_id on that event's card
    assert all(f["result"] == "scheduled" for f in card["fights"])
    assert all(f["event_id"] == "fccb0fee256b7b4d" for f in card["fights"])
    assert "f4c49976c75c5ab2" in card["fighter_ids"]  # McGregor
    assert "150ff4cc642270b9" in card["fighter_ids"]  # Holloway


def test_merge_by_key_keeps_existing_entries_not_present_in_new_fetch():
    """Regression test: a bounded/incremental fetch (e.g. the 40 most recent
    bestfightodds events) must never be treated as the complete dataset --
    in practice, dumping its result directly wiped a 5-year, 2,164-fight
    odds backfill down to ~20 events, because the old, wider fetch's data
    lived only in this same file.
    """
    existing = [{"slug": "/events/old-1", "name": "Old Event 1"}, {"slug": "/events/old-2", "name": "Old Event 2"}]
    new = [{"slug": "/events/new-1", "name": "New Event 1"}]

    merged = fetcher._merge_by_key(existing, new, key="slug")

    slugs = {item["slug"] for item in merged}
    assert slugs == {"/events/old-1", "/events/old-2", "/events/new-1"}


def test_merge_by_key_refreshes_a_re_fetched_entry_rather_than_duplicating_it():
    existing = [{"slug": "/events/e1", "name": "Stale Name"}]
    new = [{"slug": "/events/e1", "name": "Fresh Name"}]

    merged = fetcher._merge_by_key(existing, new, key="slug")

    assert len(merged) == 1
    assert merged[0]["name"] == "Fresh Name"


def test_solve_pow_submission_has_a_timeout():
    """Regression test: this POST previously had no timeout at all, unlike
    every other request in this file -- a stall here hung the whole fetch
    (and the dashboard's Refresh button, which blocks on it) indefinitely,
    with no way to recover short of manually killing the process. Confirmed
    in practice, not just in theory: a real refresh sat stuck for 12+
    minutes with near-zero CPU use, which pointed at blocked network I/O
    rather than the SHA-256 grind (that finishes in well under a second).
    """
    client = fetcher.UFCStatsClient()
    client.session.post = MagicMock(return_value=MagicMock(status_code=200))
    # target difficulty 0 -> "0" * 0 == "", and every hash starts with "" ->
    # solved on the very first attempt (n=0), so this test is instant
    # regardless of the real site's actual difficulty.
    html = 'nonce="deadbeef" ... new Array(0+1) ...'

    solved = client._solve_pow(html)

    assert solved is True
    client.session.post.assert_called_once()
    assert client.session.post.call_args.kwargs.get("timeout") == 20
