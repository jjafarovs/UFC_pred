"""Tests for the fighter-profile-based historical odds discovery mechanism
(fetcher.search_bestfightodds_fighter_url / bestfightodds_fighter_event_rows /
fetch_bestfightodds_for_fighters) -- how odds beyond bestfightodds.com's
/archive (which has no pagination and only shows ~20-25 recent events across
every promotion) get located for a historical backfill.
"""
from pathlib import Path

from src import fetcher

FIXTURES = Path(__file__).parent / "fixtures"


class FixtureClient:
    """Serves fixed HTML per path, ignoring params -- same pattern as test_fetcher.py."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.requested = []

    def get(self, path: str, params=None):
        self.requested.append(path)
        for prefix, html in self.responses.items():
            if path.startswith(prefix):
                return html
        raise AssertionError(f"FixtureClient has no response configured for {path}")


def test_search_bestfightodds_fighter_url_returns_first_fighter_link():
    html = (FIXTURES / "bfo_search.html").read_text()
    client = FixtureClient({"/search": html})
    url = fetcher.search_bestfightodds_fighter_url(client, "Rafael Fiziev")
    assert url == "/fighters/Rafael-Fiziev-8258"


def test_bestfightodds_fighter_event_rows_parses_urls_and_dates():
    html = (FIXTURES / "bfo_fighter_profile.html").read_text()
    client = FixtureClient({"/fighters/Rafael-Fiziev-8258": html})
    rows = fetcher.bestfightodds_fighter_event_rows(client, "/fighters/Rafael-Fiziev-8258")

    assert ("/events/ufc-fight-night-4230", "2026-06-27") in rows
    assert ("/events/ufc-256-figueiredo-vs-moreno-1983", "2020-12-12") in rows
    # every row must be a unique event URL -- the page lists each event once per fight,
    # but the "event-header" rows themselves should already be deduplicated
    urls = [r[0] for r in rows]
    assert len(urls) == len(set(urls))


def test_fetch_bestfightodds_for_fighters_filters_by_since_date():
    profile_html = (FIXTURES / "bfo_fighter_profile.html").read_text()
    search_html = (FIXTURES / "bfo_search.html").read_text()
    event_html = (FIXTURES / "bfo_event.html").read_text()  # any parseable event page works as a stand-in

    client = FixtureClient(
        {
            "/search": search_html,
            "/fighters/Rafael-Fiziev-8258": profile_html,
            "/events/": event_html,
        }
    )
    events = fetcher.fetch_bestfightodds_for_fighters(client, ["Rafael Fiziev"], since_date="2024-07-02")

    # Fiziev's profile has 14 events total, only 4 are on/after 2024-07-02
    # (2026-06-27, 2026-02-01, 2025-06-21, 2025-03-09)
    assert len(events) == 4


def test_fetch_bestfightodds_for_fighters_fetches_everything_without_since_date():
    profile_html = (FIXTURES / "bfo_fighter_profile.html").read_text()
    search_html = (FIXTURES / "bfo_search.html").read_text()
    event_html = (FIXTURES / "bfo_event.html").read_text()

    client = FixtureClient(
        {
            "/search": search_html,
            "/fighters/Rafael-Fiziev-8258": profile_html,
            "/events/": event_html,
        }
    )
    events = fetcher.fetch_bestfightodds_for_fighters(client, ["Rafael Fiziev"], since_date=None)
    assert len(events) == 14


def test_fetch_bestfightodds_for_fighters_dedupes_across_fighters():
    profile_html = (FIXTURES / "bfo_fighter_profile.html").read_text()
    search_html = (FIXTURES / "bfo_search.html").read_text()
    event_html = (FIXTURES / "bfo_event.html").read_text()

    client = FixtureClient(
        {
            "/search": search_html,
            "/fighters/Rafael-Fiziev-8258": profile_html,
            "/events/": event_html,
        }
    )
    # "asking about the same fighter twice" is a simple stand-in for two different
    # fighters whose profiles both list an overlapping event -- the same fighter_url
    # is only ever fetched once, and the same event_url is only ever fetched once.
    events = fetcher.fetch_bestfightodds_for_fighters(client, ["Rafael Fiziev", "Rafael Fiziev"], since_date="2024-07-02")
    assert len(events) == 4
    assert client.requested.count("/fighters/Rafael-Fiziev-8258") == 1
