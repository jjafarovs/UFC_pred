import pandas as pd
import pytest

from src import cleaner, prediction_log


def _make_db(tmp_path):
    db_path = tmp_path / "prediction_log_test.db"
    conn = cleaner.get_connection(db_path)
    cleaner.init_db(conn)
    return conn


def _insert_fighter(conn, fighter_id, name):
    conn.execute(
        "INSERT INTO fighters (fighter_id, name, scraped_at) VALUES (?, ?, 'now')",
        (fighter_id, name),
    )


def _insert_event(conn, event_id, name, event_date):
    conn.execute(
        "INSERT INTO events (event_id, name, event_date, scraped_at) VALUES (?, ?, ?, 'now')",
        (event_id, name, event_date),
    )


def _insert_completed_fight(conn, fight_id, event_id, event_date, f1, f2, winner):
    conn.execute(
        """
        INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id,
                             winner_id, result, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'now')
        """,
        (fight_id, event_id, event_date, f1, f2, winner, "fighter_1" if winner == f1 else "fighter_2"),
    )


def _insert_close_odds(conn, fight_id, fighter_id, decimal_odds):
    conn.execute(
        "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, "
        "american_odds, decimal_odds, captured_at, source) "
        "VALUES (?, ?, 'x', 'BookOne', 'close', -150, ?, 'now', 'test')",
        (fight_id, fighter_id, decimal_odds),
    )


@pytest.fixture
def db(tmp_path):
    conn = _make_db(tmp_path)
    for fid, name in [("fA", "Fighter A"), ("fB", "Fighter B"), ("fC", "Fighter C"), ("fD", "Fighter D")]:
        _insert_fighter(conn, fid, name)
    _insert_event(conn, "e1", "Event 1", "2026-01-01")
    conn.commit()
    return conn


def _card_report_row(**overrides):
    row = {
        "fighter_1": "Fighter A", "fighter_2": "Fighter B",
        "logistic_prob": 0.7, "gbm_prob": 0.65, "avg_prob": 0.675,
        "market_prob_fighter_1": 0.6, "edge_fighter_1": 0.075, "confidence": "high",
        "weight_class": "Lightweight", "title_fight": False,
        "refined_signal": "fighter_1", "refined_plus_signal": None, "elo_signal": None,
    }
    row.update(overrides)
    return row


def test_log_card_inserts_one_row_per_fight(db):
    card_report = pd.DataFrame([_card_report_row()])
    fights_df = pd.DataFrame([{
        "fight_id": "fight1", "fighter_1_id": "fA", "fighter_2_id": "fB",
        "fighter_1_name": "Fighter A", "fighter_2_name": "Fighter B",
    }])
    event_row = {"event_id": "e1", "event_name": "Event 1", "event_date": "2026-01-01"}

    n = prediction_log.log_card(db, card_report, fights_df, event_row)

    assert n == 1
    row = db.execute("SELECT * FROM prediction_log WHERE fight_id='fight1'").fetchone()
    assert row["refined_signal"] == "fighter_1"
    assert row["avg_prob_fighter_1"] == pytest.approx(0.675)


def test_log_card_upserts_same_day_instead_of_duplicating(db):
    """Streamlit reruns the whole script on every widget interaction --
    re-logging the same card on the same day must update, not duplicate.
    """
    fights_df = pd.DataFrame([{
        "fight_id": "fight1", "fighter_1_id": "fA", "fighter_2_id": "fB",
        "fighter_1_name": "Fighter A", "fighter_2_name": "Fighter B",
    }])
    event_row = {"event_id": "e1", "event_name": "Event 1", "event_date": "2026-01-01"}

    prediction_log.log_card(db, pd.DataFrame([_card_report_row(avg_prob=0.60)]), fights_df, event_row)
    prediction_log.log_card(db, pd.DataFrame([_card_report_row(avg_prob=0.90)]), fights_df, event_row)

    rows = db.execute("SELECT * FROM prediction_log WHERE fight_id='fight1'").fetchall()
    assert len(rows) == 1
    assert rows[0]["avg_prob_fighter_1"] == pytest.approx(0.90)


def test_resolve_report_matches_signal_against_real_outcome(db):
    _insert_completed_fight(db, "fight1", "e1", "2026-01-01", "fA", "fB", winner="fA")
    db.commit()
    fights_df = pd.DataFrame([{
        "fight_id": "fight1", "fighter_1_id": "fA", "fighter_2_id": "fB",
        "fighter_1_name": "Fighter A", "fighter_2_name": "Fighter B",
    }])
    event_row = {"event_id": "e1", "event_name": "Event 1", "event_date": "2026-01-01"}
    # refined picked fighter_1 (correct, fA won); refined+ had no signal
    prediction_log.log_card(
        db, pd.DataFrame([_card_report_row(refined_signal="fighter_1", refined_plus_signal=None)]),
        fights_df, event_row,
    )

    resolved = prediction_log.resolve_report(db)

    assert len(resolved) == 1
    assert resolved.iloc[0]["refined_correct"] == True  # noqa: E712
    assert pd.isna(resolved.iloc[0]["refined_plus_correct"]) or resolved.iloc[0]["refined_plus_correct"] is None


def test_resolve_report_scores_elo_signal_too(db):
    _insert_completed_fight(db, "fight1", "e1", "2026-01-01", "fA", "fB", winner="fA")
    db.commit()
    fights_df = pd.DataFrame([{
        "fight_id": "fight1", "fighter_1_id": "fA", "fighter_2_id": "fB",
        "fighter_1_name": "Fighter A", "fighter_2_name": "Fighter B",
    }])
    event_row = {"event_id": "e1", "event_name": "Event 1", "event_date": "2026-01-01"}
    prediction_log.log_card(
        db, pd.DataFrame([_card_report_row(elo_signal="fighter_2")]), fights_df, event_row,
    )

    resolved = prediction_log.resolve_report(db)

    assert resolved.iloc[0]["elo_correct"] == False  # noqa: E712 -- fA won, elo picked fighter_2


def test_resolve_report_pulls_real_closing_odds_not_the_logged_live_line(db):
    _insert_completed_fight(db, "fight1", "e1", "2026-01-01", "fA", "fB", winner="fA")
    _insert_close_odds(db, "fight1", "fA", 1.5)
    _insert_close_odds(db, "fight1", "fB", 2.8)
    db.commit()
    fights_df = pd.DataFrame([{
        "fight_id": "fight1", "fighter_1_id": "fA", "fighter_2_id": "fB",
        "fighter_1_name": "Fighter A", "fighter_2_name": "Fighter B",
    }])
    event_row = {"event_id": "e1", "event_name": "Event 1", "event_date": "2026-01-01"}
    prediction_log.log_card(db, pd.DataFrame([_card_report_row(refined_signal="fighter_1")]), fights_df, event_row)

    resolved = prediction_log.resolve_report(db)

    assert resolved.iloc[0]["refined_decimal_odds"] == pytest.approx(1.5)


def test_resolve_report_returns_empty_when_nothing_has_completed_yet(db):
    fights_df = pd.DataFrame([{
        "fight_id": "fight1", "fighter_1_id": "fA", "fighter_2_id": "fB",
        "fighter_1_name": "Fighter A", "fighter_2_name": "Fighter B",
    }])
    event_row = {"event_id": "e1", "event_name": "Event 1", "event_date": "2026-01-01"}
    prediction_log.log_card(db, pd.DataFrame([_card_report_row()]), fights_df, event_row)

    resolved = prediction_log.resolve_report(db)

    assert resolved.empty
