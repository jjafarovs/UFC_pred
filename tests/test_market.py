import sqlite3

import pytest

from src import cleaner, market


def test_implied_prob_from_american_favorite_and_underdog():
    assert market.implied_prob_from_american(150) == pytest.approx(100 / 250)
    assert market.implied_prob_from_american(-150) == pytest.approx(150 / 250)
    assert market.implied_prob_from_american(100) == pytest.approx(0.5)
    assert market.implied_prob_from_american(-100) == pytest.approx(0.5)


def test_implied_prob_from_decimal():
    assert market.implied_prob_from_decimal(2.5) == pytest.approx(0.4)
    assert market.implied_prob_from_decimal(1.5) == pytest.approx(1 / 1.5)


def test_devig_two_way_sums_to_one_and_preserves_ratio():
    # -150/+130 book line: raw implied probs sum to > 1 (the vig)
    prob_a = market.implied_prob_from_american(-150)  # 0.6
    prob_b = market.implied_prob_from_american(130)  # ~0.4348
    devigged_a, devigged_b = market.devig_two_way(prob_a, prob_b)

    assert devigged_a + devigged_b == pytest.approx(1.0)
    # relative ratio between the two sides is unchanged by the normalization
    assert devigged_a / devigged_b == pytest.approx(prob_a / prob_b)


def test_devig_two_way_equal_odds_split_evenly():
    devigged_a, devigged_b = market.devig_two_way(0.55, 0.55)
    assert devigged_a == pytest.approx(0.5)
    assert devigged_b == pytest.approx(0.5)


def test_compute_edge_is_model_minus_market():
    assert market.compute_edge(0.60, 0.52) == pytest.approx(0.08)
    assert market.compute_edge(0.40, 0.52) == pytest.approx(-0.12)


def _db_with_odds(tmp_path):
    db_path = tmp_path / "market_test.db"
    conn = cleaner.get_connection(db_path)
    cleaner.init_db(conn)
    conn.execute("INSERT INTO fighters (fighter_id, name, scraped_at) VALUES ('fA', 'Fighter A', 'now')")
    conn.execute("INSERT INTO fighters (fighter_id, name, scraped_at) VALUES ('fB', 'Fighter B', 'now')")
    conn.execute("INSERT INTO events (event_id, name, event_date, scraped_at) VALUES ('e1', 'Event 1', '2026-01-01', 'now')")
    conn.execute(
        "INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id, winner_id, result, scraped_at) "
        "VALUES ('fight1', 'e1', '2026-01-01', 'fA', 'fB', 'fA', 'fighter_1', 'now')"
    )
    return conn


def test_market_probabilities_for_fight_averages_across_books(tmp_path):
    conn = _db_with_odds(tmp_path)
    # Book 1: fA -150 (~0.6), fB +130 (~0.4348) -- devigs to (0.5793, 0.4207)
    # Book 2: fA -140 (~0.5833), fB +120 (~0.4545) -- devigs to (0.5619, 0.4381)
    for book, (fa_american, fa_dec), (fb_american, fb_dec) in [
        ("BookOne", (-150, cleaner._american_to_decimal(-150)), (130, cleaner._american_to_decimal(130))),
        ("BookTwo", (-140, cleaner._american_to_decimal(-140)), (120, cleaner._american_to_decimal(120))),
    ]:
        conn.execute(
            "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
            "VALUES ('fight1', 'fA', 'Fighter A', ?, 'close', ?, ?, 'now', 'test')",
            (book, fa_american, fa_dec),
        )
        conn.execute(
            "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
            "VALUES ('fight1', 'fB', 'Fighter B', ?, 'close', ?, ?, 'now', 'test')",
            (book, fb_american, fb_dec),
        )
    conn.commit()

    probs = market.market_probabilities_for_fight(conn, "fight1", odds_type="close")
    assert set(probs) == {"fA", "fB"}
    assert probs["fA"] + probs["fB"] == pytest.approx(1.0, abs=1e-6)
    # both books favor fA, so the averaged devigged probability should too
    assert probs["fA"] > probs["fB"]


def test_market_probabilities_for_fight_skips_one_sided_books(tmp_path):
    conn = _db_with_odds(tmp_path)
    # only fA's side was captured for this book -- can't de-vig a single side
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES ('fight1', 'fA', 'Fighter A', 'OneSidedBook', 'close', -150, ?, 'now', 'test')",
        (cleaner._american_to_decimal(-150),),
    )
    conn.commit()

    probs = market.market_probabilities_for_fight(conn, "fight1", odds_type="close")
    assert probs == {}


def test_market_probabilities_for_fight_returns_empty_when_no_odds(tmp_path):
    conn = _db_with_odds(tmp_path)
    probs = market.market_probabilities_for_fight(conn, "fight1", odds_type="close")
    assert probs == {}


def test_market_probabilities_for_matchup_uses_unmatched_odds_rows(tmp_path):
    conn = _db_with_odds(tmp_path)
    # An upcoming matchup: odds exist for fA/fB but with no fight_id yet,
    # since there's no completed fight row for cleaner.py to have linked them to.
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES (NULL, 'fA', 'Fighter A', 'BookOne', 'live', -150, ?, 'now', 'test')",
        (cleaner._american_to_decimal(-150),),
    )
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES (NULL, 'fB', 'Fighter B', 'BookOne', 'live', 130, ?, 'now', 'test')",
        (cleaner._american_to_decimal(130),),
    )
    conn.commit()

    probs = market.market_probabilities_for_matchup(conn, "fA", "fB", odds_type="live")
    assert set(probs) == {"fA", "fB"}
    assert probs["fA"] + probs["fB"] == pytest.approx(1.0, abs=1e-6)
    assert probs["fA"] > probs["fB"]


def test_market_probabilities_for_matchup_ignores_rows_already_linked_to_a_fight(tmp_path):
    conn = _db_with_odds(tmp_path)
    # These rows ARE linked to fight1 (fight_id is set) -- market_probabilities_for_matchup
    # should not pick them up, since that's market_probabilities_for_fight's job.
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES ('fight1', 'fA', 'Fighter A', 'BookOne', 'live', -150, ?, 'now', 'test')",
        (cleaner._american_to_decimal(-150),),
    )
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES ('fight1', 'fB', 'Fighter B', 'BookOne', 'live', 130, ?, 'now', 'test')",
        (cleaner._american_to_decimal(130),),
    )
    conn.commit()

    probs = market.market_probabilities_for_matchup(conn, "fA", "fB", odds_type="live")
    assert probs == {}
