from pathlib import Path

from src import fetcher

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_bestfightodds_event_extracts_moneylines():
    html = (FIXTURES / "bfo_event.html").read_text()
    event = fetcher.parse_bestfightodds_event(html)

    assert event["event_name"] == "UFC Abu Dhabi"
    assert event["event_date_raw"] == "July 25, 2026"
    assert event["last_change_raw"] == "Jun 28th 2026 13:58 UTC"
    assert len(event["matchups"]) >= 1

    matchup = next(m for m in event["matchups"] if m["matchup_id"] == "43567")
    names = {f["fighter_name"] for f in matchup["fighters"]}
    assert names == {"Khalil Rountree Jr", "Magomed Ankalaev"}

    rountree = next(f for f in matchup["fighters"] if f["fighter_name"] == "Khalil Rountree Jr")
    ankalaev = next(f for f in matchup["fighters"] if f["fighter_name"] == "Magomed Ankalaev")

    assert rountree["odds"]["FanDuel"] == 240
    assert rountree["odds"]["Caesars"] == 230
    assert ankalaev["odds"]["FanDuel"] == -330
    assert ankalaev["odds"]["Caesars"] == -320
    # Unibet has no affiliate link (plain <span>, not <a>) -- must still parse.
    assert rountree["odds"]["Unibet"] == 235
