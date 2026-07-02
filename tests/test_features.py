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


def _insert_fight(conn, fight_id, event_id, event_date, f1, f2, winner, stats):
    """stats: {f1: {sig_l, sig_a, td_l, td_a}, f2: {...}}"""
    conn.execute(
        """
        INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id,
                             winner_id, result, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'now')
        """,
        (fight_id, event_id, event_date, f1, f2, winner, "fighter_1" if winner == f1 else "fighter_2"),
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
