from datetime import date

import pytest

from src import cleaner, features


def _make_db(tmp_path):
    db_path = tmp_path / "features_test.db"
    conn = cleaner.get_connection(db_path)
    cleaner.init_db(conn)
    return conn


def _insert_fighter(conn, fighter_id, name, dob=None, height_in=70, reach_in=72, stance="Orthodox"):
    conn.execute(
        "INSERT INTO fighters (fighter_id, name, height_in, reach_in, stance, dob, scraped_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'now')",
        (fighter_id, name, height_in, reach_in, stance, dob),
    )


def _insert_event(conn, event_id, name, event_date):
    conn.execute(
        "INSERT INTO events (event_id, name, event_date, scraped_at) VALUES (?, ?, ?, 'now')",
        (event_id, name, event_date),
    )


def _insert_fight(conn, fight_id, event_id, event_date, f1, f2, winner, stats, method=None):
    """stats: {f1: {sig_l, sig_a, td_l, td_a}, f2: {...}}"""
    conn.execute(
        """
        INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id,
                             winner_id, result, method, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'now')
        """,
        (fight_id, event_id, event_date, f1, f2, winner, "fighter_1" if winner == f1 else "fighter_2", method),
    )
    for fighter, s in stats.items():
        conn.execute(
            """
            INSERT INTO fight_stats (fight_id, fighter_id, round, sig_str_landed, sig_str_attempted,
                                      takedowns_landed, takedowns_attempted)
            VALUES (?, ?, 0, ?, ?, ?, ?)
            """,
            (fight_id, fighter, s["sig_l"], s["sig_a"], s["td_l"], s["td_a"]),
        )


@pytest.fixture
def fighter_a_history(tmp_path):
    """Fighter A: win vs B (2026-01-01), loss vs C (2026-03-01), win vs D (2026-06-01)."""
    conn = _make_db(tmp_path)
    for fid, name in [("fA", "Fighter A"), ("fB", "Fighter B"), ("fC", "Fighter C"), ("fD", "Fighter D")]:
        _insert_fighter(conn, fid, name, dob="1993-01-01")
    _insert_event(conn, "e1", "Event 1", "2026-01-01")
    _insert_event(conn, "e2", "Event 2", "2026-03-01")
    _insert_event(conn, "e3", "Event 3", "2026-06-01")

    _insert_fight(conn, "fight1", "e1", "2026-01-01", "fA", "fB", "fA",
                  {"fA": {"sig_l": 50, "sig_a": 100, "td_l": 2, "td_a": 4},
                   "fB": {"sig_l": 20, "sig_a": 80, "td_l": 0, "td_a": 2}})
    _insert_fight(conn, "fight2", "e2", "2026-03-01", "fA", "fC", "fC",
                  {"fA": {"sig_l": 30, "sig_a": 90, "td_l": 1, "td_a": 5},
                   "fC": {"sig_l": 40, "sig_a": 70, "td_l": 3, "td_a": 5}})
    _insert_fight(conn, "fight3", "e3", "2026-06-01", "fA", "fD", "fA",
                  {"fA": {"sig_l": 60, "sig_a": 110, "td_l": 4, "td_a": 6},
                   "fD": {"sig_l": 10, "sig_a": 60, "td_l": 0, "td_a": 1}})
    conn.commit()
    return conn


def test_fights_before_excludes_same_date_and_future_fights(fighter_a_history):
    conn = fighter_a_history
    # as_of exactly fight2's date -> fight2 itself and fight3 must NOT appear
    rows = features.fights_before(conn, "fA", "2026-03-01")
    assert [r["fight_id"] for r in rows] == ["fight1"]

    rows = features.fights_before(conn, "fA", "2026-06-01")
    assert {r["fight_id"] for r in rows} == {"fight1", "fight2"}

    rows = features.fights_before(conn, "fA", "2026-01-01")
    assert rows == []


def test_fighter_rolling_features_at_each_point_in_time(fighter_a_history):
    conn = fighter_a_history

    # Before fA's first fight: no history at all.
    feats = features.fighter_rolling_features(conn, "fA", "2026-01-01")
    assert feats["n_prior_fights"] == 0
    assert feats["win_pct"] is None

    # As of fight2's date: only fight1 (a win) counts.
    feats = features.fighter_rolling_features(conn, "fA", "2026-03-01")
    assert feats["n_prior_fights"] == 1
    assert feats["win_pct"] == 1.0
    assert feats["current_streak"] == 1
    assert feats["sig_str_acc"] == pytest.approx(50 / 100)
    assert feats["sig_str_def"] == pytest.approx(1 - 20 / 80)
    assert feats["days_since_last_fight"] == (date(2026, 3, 1) - date(2026, 1, 1)).days

    # As of fight3's date: fight1 (win) + fight2 (loss) count, NOT fight3 itself.
    feats = features.fighter_rolling_features(conn, "fA", "2026-06-01")
    assert feats["n_prior_fights"] == 2
    assert feats["win_pct"] == 0.5
    assert feats["current_streak"] == -1  # most recent prior result is the loss


def test_build_feature_matrix_uses_only_past_fights_per_row(fighter_a_history):
    conn = fighter_a_history
    df = features.build_feature_matrix(conn)
    assert len(df) == 3

    row1 = df[df["fight_id"] == "fight1"].iloc[0]
    row2 = df[df["fight_id"] == "fight2"].iloc[0]
    row3 = df[df["fight_id"] == "fight3"].iloc[0]

    # fight1 is fA's first fight ever -> zero prior history for fA.
    assert row1["f1_n_prior_fights"] == 0
    # fight2 is fA's second fight -> exactly 1 prior fight (fight1), never fight2/fight3.
    assert row2["f1_n_prior_fights"] == 1
    # fight3 is fA's third fight -> exactly 2 prior fights (fight1, fight2), never fight3 itself.
    assert row3["f1_n_prior_fights"] == 2

    assert row2["label_fighter_1_win"] == 0  # fC won fight2
    assert row3["label_fighter_1_win"] == 1  # fA won fight3


def test_market_prob_feature_returns_none_when_no_odds_matched(fighter_a_history):
    conn = fighter_a_history
    assert features.market_prob_feature(conn, "fight1", "fA") is None


def test_market_prob_feature_returns_devigged_probability_when_odds_exist(fighter_a_history):
    conn = fighter_a_history
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES ('fight1', 'fA', 'Fighter A', 'BookOne', 'close', -150, 1.6667, 'now', 'test')"
    )
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES ('fight1', 'fB', 'Fighter B', 'BookOne', 'close', 130, 2.3, 'now', 'test')"
    )
    conn.commit()
    prob = features.market_prob_feature(conn, "fight1", "fA")
    assert prob is not None
    assert 0.5 < prob < 0.7  # fA is the favorite here


def test_build_fight_feature_row_includes_market_prob_when_available(fighter_a_history):
    conn = fighter_a_history
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES ('fight1', 'fA', 'Fighter A', 'BookOne', 'close', -150, 1.6667, 'now', 'test')"
    )
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
        "VALUES ('fight1', 'fB', 'Fighter B', 'BookOne', 'close', 130, 2.3, 'now', 'test')"
    )
    conn.commit()
    fight = conn.execute("SELECT * FROM fights WHERE fight_id='fight1'").fetchone()
    row = features.build_fight_feature_row(conn, fight)
    assert row["market_prob_fighter_1"] is not None

    fight2 = conn.execute("SELECT * FROM fights WHERE fight_id='fight2'").fetchone()
    row2 = features.build_fight_feature_row(conn, fight2)
    assert row2["market_prob_fighter_1"] is None  # fight2 has no matched odds


@pytest.fixture
def fighter_a_career_history(tmp_path):
    """Fighter A: KO win vs B, decision win vs C, submission loss vs D, all
    before 2026-06-01 -- 3 prior fights, 2 wins (1 finish), 1 loss (a finish).
    """
    conn = _make_db(tmp_path)
    for fid, name in [("fA", "Fighter A"), ("fB", "Fighter B"), ("fC", "Fighter C"), ("fD", "Fighter D")]:
        _insert_fighter(conn, fid, name)
    _insert_event(conn, "e1", "Event 1", "2026-01-01")
    _insert_event(conn, "e2", "Event 2", "2026-02-01")
    _insert_event(conn, "e3", "Event 3", "2026-03-01")

    stats = {"fA": {"sig_l": 10, "sig_a": 20, "td_l": 0, "td_a": 0}, "fB": {"sig_l": 5, "sig_a": 15, "td_l": 0, "td_a": 0}}
    _insert_fight(conn, "fight1", "e1", "2026-01-01", "fA", "fB", "fA", stats, method="KO/TKO")
    stats2 = {"fA": {"sig_l": 10, "sig_a": 20, "td_l": 0, "td_a": 0}, "fC": {"sig_l": 5, "sig_a": 15, "td_l": 0, "td_a": 0}}
    _insert_fight(conn, "fight2", "e2", "2026-02-01", "fA", "fC", "fA", stats2, method="Decision - Unanimous")
    stats3 = {"fA": {"sig_l": 10, "sig_a": 20, "td_l": 0, "td_a": 0}, "fD": {"sig_l": 5, "sig_a": 15, "td_l": 0, "td_a": 0}}
    _insert_fight(conn, "fight3", "e3", "2026-03-01", "fD", "fA", "fD", stats3, method="Submission")
    conn.commit()
    return conn


def test_fighter_career_features_zero_prior_fights(fighter_a_career_history):
    conn = fighter_a_career_history
    result = features.fighter_career_features(conn, "fA", "2026-01-01")
    assert result == {"total_prior_fights": 0, "finish_rate": None, "times_finished_rate": None}


def test_fighter_career_features_counts_and_rates(fighter_a_career_history):
    conn = fighter_a_career_history
    result = features.fighter_career_features(conn, "fA", "2026-06-01")
    assert result["total_prior_fights"] == 3
    assert result["finish_rate"] == pytest.approx(0.5)  # 1 of 2 wins was a finish (KO/TKO, not the decision)
    assert result["times_finished_rate"] == pytest.approx(1.0)  # the 1 loss was a submission


def test_fighter_career_features_only_counts_fights_before_as_of_date(fighter_a_career_history):
    conn = fighter_a_career_history
    result = features.fighter_career_features(conn, "fA", "2026-02-01")  # only fight1 counts
    assert result["total_prior_fights"] == 1
    assert result["finish_rate"] == pytest.approx(1.0)  # the only win was a KO


def _insert_round_stats(conn, fight_id, fighter_id, round_num, sig_l):
    conn.execute(
        "INSERT INTO fight_stats (fight_id, fighter_id, round, sig_str_landed, sig_str_attempted) "
        "VALUES (?, ?, ?, ?, ?)",
        (fight_id, fighter_id, round_num, sig_l, sig_l + 10),
    )


@pytest.fixture
def fighter_a_round_history(tmp_path):
    """Fighter A has 3 fights, each providing round-level stats:
    fight1 (3 rounds, went the full 3): round1=40 landed, round2=20 landed -- clear fade (0.5 ratio).
    fight2 (1 round only, a first-round finish): no fade data -- must be excluded, not treated as ratio 1.0.
    fight3 (5 rounds): round1=30 landed, round4=30 landed -- no fade (ratio 1.0).
    """
    conn = _make_db(tmp_path)
    for fid, name in [("fA", "Fighter A"), ("fB", "Fighter B"), ("fC", "Fighter C"), ("fD", "Fighter D")]:
        _insert_fighter(conn, fid, name)
    _insert_event(conn, "e1", "Event 1", "2026-01-01")
    _insert_event(conn, "e2", "Event 2", "2026-02-01")
    _insert_event(conn, "e3", "Event 3", "2026-03-01")

    _insert_fight(conn, "fight1", "e1", "2026-01-01", "fA", "fB", "fA",
                  {"fA": {"sig_l": 60, "sig_a": 100, "td_l": 0, "td_a": 0},
                   "fB": {"sig_l": 30, "sig_a": 80, "td_l": 0, "td_a": 0}})
    conn.execute("UPDATE fights SET end_round=3 WHERE fight_id='fight1'")
    _insert_round_stats(conn, "fight1", "fA", 1, sig_l=40)
    _insert_round_stats(conn, "fight1", "fA", 2, sig_l=20)  # end_round-1 = 2 -> the "late" round used

    _insert_fight(conn, "fight2", "e2", "2026-02-01", "fA", "fC", "fA",
                  {"fA": {"sig_l": 15, "sig_a": 20, "td_l": 0, "td_a": 0},
                   "fC": {"sig_l": 5, "sig_a": 15, "td_l": 0, "td_a": 0}})
    conn.execute("UPDATE fights SET end_round=1 WHERE fight_id='fight2'")
    _insert_round_stats(conn, "fight2", "fA", 1, sig_l=15)  # only round 1 exists -- a first-round finish

    _insert_fight(conn, "fight3", "e3", "2026-03-01", "fA", "fD", "fA",
                  {"fA": {"sig_l": 90, "sig_a": 150, "td_l": 0, "td_a": 0},
                   "fD": {"sig_l": 40, "sig_a": 100, "td_l": 0, "td_a": 0}})
    conn.execute("UPDATE fights SET end_round=5 WHERE fight_id='fight3'")
    _insert_round_stats(conn, "fight3", "fA", 1, sig_l=30)
    _insert_round_stats(conn, "fight3", "fA", 4, sig_l=30)  # end_round-1 = 4 -> the "late" round used

    conn.commit()
    return conn


def test_fade_rate_averages_only_fights_with_a_valid_late_round_comparison(fighter_a_round_history):
    conn = fighter_a_round_history
    result = features.fighter_rolling_features(conn, "fA", "2026-04-01", n=5)
    # fight2 (1-round finish) must be excluded entirely, not counted as a
    # ratio of 1.0 -- only fight1 (0.5) and fight3 (1.0) contribute.
    assert result["fade_rate"] == pytest.approx((0.5 + 1.0) / 2)


def test_fade_rate_respects_the_as_of_date_leakage_guard(fighter_a_round_history):
    conn = fighter_a_round_history
    # As of right after fight1, only fight1's ratio (0.5) is available.
    result = features.fighter_rolling_features(conn, "fA", "2026-01-15", n=5)
    assert result["fade_rate"] == pytest.approx(0.5)


def test_fade_rate_is_none_when_no_fight_in_window_qualifies(tmp_path):
    conn = _make_db(tmp_path)
    _insert_fighter(conn, "fA", "Fighter A")
    _insert_fighter(conn, "fB", "Fighter B")
    _insert_event(conn, "e1", "Event 1", "2026-01-01")
    _insert_fight(conn, "fight1", "e1", "2026-01-01", "fA", "fB", "fA",
                  {"fA": {"sig_l": 15, "sig_a": 20, "td_l": 0, "td_a": 0},
                   "fB": {"sig_l": 5, "sig_a": 15, "td_l": 0, "td_a": 0}})
    conn.execute("UPDATE fights SET end_round=1 WHERE fight_id='fight1'")  # first-round finish only
    conn.commit()

    result = features.fighter_rolling_features(conn, "fA", "2026-02-01", n=5)
    assert result["fade_rate"] is None


@pytest.fixture
def elo_history(tmp_path):
    """A beats B (2026-01-01, decision), then B beats C (2026-02-01, KO)."""
    conn = _make_db(tmp_path)
    for fid, name in [("fA", "Fighter A"), ("fB", "Fighter B"), ("fC", "Fighter C")]:
        _insert_fighter(conn, fid, name)
    _insert_event(conn, "e1", "Event 1", "2026-01-01")
    _insert_event(conn, "e2", "Event 2", "2026-02-01")
    _insert_fight(conn, "fight1", "e1", "2026-01-01", "fA", "fB", "fA",
                  {"fA": {"sig_l": 10, "sig_a": 20, "td_l": 0, "td_a": 0},
                   "fB": {"sig_l": 5, "sig_a": 15, "td_l": 0, "td_a": 0}},
                  method="Decision - Unanimous")
    _insert_fight(conn, "fight2", "e2", "2026-02-01", "fB", "fC", "fB",
                  {"fB": {"sig_l": 10, "sig_a": 20, "td_l": 0, "td_a": 0},
                   "fC": {"sig_l": 5, "sig_a": 15, "td_l": 0, "td_a": 0}},
                  method="KO/TKO")
    conn.commit()
    return conn


def test_build_elo_ratings_starts_everyone_at_1500_and_updates_after_a_win(elo_history):
    conn = elo_history
    timeline = features.build_elo_ratings(conn)
    # fA's rating after beating fB (both started at 1500, even matchup) went up.
    assert timeline["fA"][0][1] > 1500.0
    # fB's rating after LOSING to fA went down.
    assert timeline["fB"][0][1] < 1500.0


def test_build_elo_ratings_weights_a_finish_more_than_a_decision(elo_history):
    conn = elo_history
    timeline = features.build_elo_ratings(conn)
    # fA's decision win margin vs fB (both starting at 1500).
    decision_gain = timeline["fA"][0][1] - 1500.0
    # fB's KO win margin vs fC -- fB entered this fight already BELOW 1500
    # (having just lost fight1), which on its own would shrink a same-size
    # gain; the 1.5x finish multiplier must still win out.
    ko_gain = timeline["fB"][1][1] - timeline["fB"][0][1]
    assert ko_gain > decision_gain


def test_fighter_elo_as_of_respects_the_as_of_date_leakage_guard(elo_history):
    conn = elo_history
    timeline = features.build_elo_ratings(conn)
    # Before fA's only fight: still the 1500.0 default.
    assert features.fighter_elo_as_of(timeline, "fA", "2026-01-01") == 1500.0
    # After fight1 but before/on fight2's date (fA isn't in fight2): fA's
    # rating reflects fight1, not anything from fB/fC's later fight.
    assert features.fighter_elo_as_of(timeline, "fA", "2026-02-01") == pytest.approx(timeline["fA"][0][1])
    # fB's rating as of a date between its two fights reflects only fight1.
    assert features.fighter_elo_as_of(timeline, "fB", "2026-02-01") == pytest.approx(timeline["fB"][0][1])
    # fB's rating strictly after fight2 reflects both fights.
    assert features.fighter_elo_as_of(timeline, "fB", "2026-03-01") == pytest.approx(timeline["fB"][1][1])


def test_fighter_elo_as_of_defaults_to_1500_for_an_unknown_fighter(elo_history):
    conn = elo_history
    timeline = features.build_elo_ratings(conn)
    assert features.fighter_elo_as_of(timeline, "nobody", "2026-06-01") == 1500.0


def test_elo_prob_feature_favors_the_higher_rated_fighter(elo_history):
    conn = elo_history
    timeline = features.build_elo_ratings(conn)
    # As of after both fights: fA (1 win, never lost) should be rated above fC (1 loss).
    prob = features.elo_prob_feature(timeline, "fA", "fC", "2026-06-01")
    assert prob > 0.5


def test_matchup_feature_dict_omits_elo_when_timeline_not_provided(elo_history):
    conn = elo_history
    feat = features.matchup_feature_dict(conn, "fA", "fB", "2026-06-01")
    assert feat["elo_prob_fighter_1"] is None


def test_matchup_feature_dict_includes_elo_when_timeline_provided(elo_history):
    conn = elo_history
    timeline = features.build_elo_ratings(conn)
    feat = features.matchup_feature_dict(conn, "fA", "fB", "2026-06-01", elo_timeline=timeline)
    assert feat["elo_prob_fighter_1"] is not None
    assert 0.0 < feat["elo_prob_fighter_1"] < 1.0


def test_draws_and_no_contests_are_excluded_from_the_matrix(tmp_path):
    conn = _make_db(tmp_path)
    _insert_fighter(conn, "fA", "Fighter A")
    _insert_fighter(conn, "fB", "Fighter B")
    _insert_event(conn, "e1", "Event 1", "2026-01-01")
    conn.execute(
        "INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id, "
        "winner_id, result, scraped_at) VALUES ('fightd', 'e1', '2026-01-01', 'fA', 'fB', NULL, 'draw', 'now')"
    )
    conn.commit()
    df = features.build_feature_matrix(conn)
    assert len(df) == 0
