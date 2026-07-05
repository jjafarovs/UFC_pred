import json

import numpy as np
import pandas as pd
import pytest

from src import cleaner, model, report


class _FakeModel:
    """Stands in for a fitted/calibrated sklearn model: report.py only needs
    predict_proba(X) -> shape (n, 2), so a fixed-probability stub is enough
    to test report.py's own logic (feature building, market join, confidence
    labeling) without coupling these tests to model.py's training stochastics.
    """

    def __init__(self, prob_fighter_1_win=0.7):
        self.prob = prob_fighter_1_win
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = X
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


def _db_with_history(tmp_path):
    """fA has 2 completed prior fights (vs fC, fD); fB has none yet."""
    db_path = tmp_path / "report_test.db"
    conn = cleaner.get_connection(db_path)
    cleaner.init_db(conn)
    for fid, name in [("fA", "Fighter A"), ("fB", "Fighter B"), ("fC", "Fighter C"), ("fD", "Fighter D")]:
        conn.execute(f"INSERT INTO fighters (fighter_id, name, scraped_at) VALUES ('{fid}', '{name}', 'now')")
    conn.execute("INSERT INTO events (event_id, name, event_date, scraped_at) VALUES ('e1', 'Event 1', '2026-01-01', 'now')")
    conn.execute("INSERT INTO events (event_id, name, event_date, scraped_at) VALUES ('e2', 'Event 2', '2026-03-01', 'now')")
    conn.execute(
        "INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id, winner_id, result, scraped_at) "
        "VALUES ('fight1', 'e1', '2026-01-01', 'fA', 'fC', 'fA', 'fighter_1', 'now')"
    )
    conn.execute(
        "INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id, winner_id, result, scraped_at) "
        "VALUES ('fight2', 'e2', '2026-03-01', 'fA', 'fD', 'fA', 'fighter_1', 'now')"
    )
    for fight_id, fid in [("fight1", "fA"), ("fight1", "fC"), ("fight2", "fA"), ("fight2", "fD")]:
        conn.execute(
            "INSERT INTO fight_stats (fight_id, fighter_id, round, sig_str_landed, sig_str_attempted, takedowns_landed, takedowns_attempted) "
            "VALUES (?, ?, 0, 10, 20, 1, 2)",
            (fight_id, fid),
        )
    conn.commit()
    return conn


def test_build_card_report_computes_model_probability_for_upcoming_matchup(tmp_path):
    conn = _db_with_history(tmp_path)
    fake_model = _FakeModel(prob_fighter_1_win=0.7)

    result = report.build_card_report(conn, fake_model, [("fA", "fB")], as_of_date="2026-06-01", odds_type="live")
    assert len(result) == 1
    row = result.iloc[0]
    assert row["fighter_1"] == "Fighter A"
    assert row["fighter_2"] == "Fighter B"
    assert row["model_prob_fighter_1"] == pytest.approx(0.7)
    assert row["market_prob_fighter_1"] is None
    assert row["edge_fighter_1"] is None


def test_build_card_report_includes_market_data_when_odds_exist(tmp_path):
    conn = _db_with_history(tmp_path)
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

    fake_model = _FakeModel(prob_fighter_1_win=0.7)
    result = report.build_card_report(conn, fake_model, [("fA", "fB")], as_of_date="2026-06-01", odds_type="live")
    row = result.iloc[0]
    assert row["market_prob_fighter_1"] is not None
    assert row["edge_fighter_1"] == pytest.approx(0.7 - row["market_prob_fighter_1"])

    # market_prob_fighter_1 must reach the model as an input feature, not just get
    # used afterward for the edge calc -- it's part of model.FEATURE_COLUMNS now.
    assert "market_prob_fighter_1" in fake_model.last_X.columns
    assert fake_model.last_X["market_prob_fighter_1"].iloc[0] == pytest.approx(row["market_prob_fighter_1"])


def test_build_card_report_passes_nan_market_prob_to_model_when_unavailable(tmp_path):
    conn = _db_with_history(tmp_path)
    fake_model = _FakeModel(prob_fighter_1_win=0.7)
    report.build_card_report(conn, fake_model, [("fA", "fB")], as_of_date="2026-06-01", odds_type="live")
    assert pd.isna(fake_model.last_X["market_prob_fighter_1"].iloc[0])


def test_build_card_report_confidence_reflects_prior_fight_counts(tmp_path):
    conn = _db_with_history(tmp_path)
    fake_model = _FakeModel(prob_fighter_1_win=0.55)

    # fA has 2 prior fights as of this date, fB has 0 -- min(2, 0) = 0 -> "low"
    result = report.build_card_report(conn, fake_model, [("fA", "fB")], as_of_date="2026-06-01")
    assert result.iloc[0]["confidence"] == "low"


def test_load_latest_model_picks_most_recently_timestamped_artifact(tmp_path):
    import joblib

    joblib.dump(_FakeModel(0.5), tmp_path / "logistic_20260101T000000Z.joblib")
    joblib.dump(_FakeModel(0.6), tmp_path / "logistic_20260601T120000Z.joblib")

    loaded, path = report.load_latest_model("logistic", models_dir=tmp_path)
    assert path.name == "logistic_20260601T120000Z.joblib"
    assert loaded.prob == 0.6


def test_load_latest_model_raises_when_none_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.load_latest_model("logistic", models_dir=tmp_path)


def test_load_production_model_falls_back_to_latest_without_a_pin(tmp_path, monkeypatch):
    import joblib

    # Must not depend on whatever the real project's models/production.json says --
    # point PRODUCTION_CONFIG_PATH at a tmp location that deliberately has no pin file.
    monkeypatch.setattr(report, "PRODUCTION_CONFIG_PATH", tmp_path / "production.json")
    joblib.dump(_FakeModel(0.5), tmp_path / "logistic_20260101T000000Z.joblib")
    joblib.dump(_FakeModel(0.6), tmp_path / "logistic_20260601T120000Z.joblib")

    loaded, path = report.load_production_model("logistic", models_dir=tmp_path)
    assert path.name == "logistic_20260601T120000Z.joblib"
    assert loaded.prob == 0.6


def test_set_and_load_production_model_pin(tmp_path):
    import joblib

    old_path = tmp_path / "logistic_20260101T000000Z.joblib"
    new_path = tmp_path / "logistic_20260601T120000Z.joblib"
    joblib.dump(_FakeModel(0.5), old_path)
    joblib.dump(_FakeModel(0.6), new_path)
    config_path = tmp_path / "production.json"

    # Deliberately pin the OLDER artifact -- proves the pin overrides "latest by filename".
    report.set_production_model("logistic", old_path, config_path=config_path)
    # load_production_model uses the module-level PRODUCTION_CONFIG_PATH by default,
    # so patch it to point at our tmp config for this test.
    import src.report as report_module

    original = report_module.PRODUCTION_CONFIG_PATH
    report_module.PRODUCTION_CONFIG_PATH = config_path
    try:
        loaded, path = report.load_production_model("logistic", models_dir=tmp_path)
    finally:
        report_module.PRODUCTION_CONFIG_PATH = original
    assert path.name == "logistic_20260101T000000Z.joblib"
    assert loaded.prob == 0.5


def test_load_production_model_raises_if_pinned_file_missing(tmp_path):
    import src.report as report_module

    config_path = tmp_path / "production.json"
    config_path.write_text('{"logistic": "does_not_exist.joblib"}')
    original = report_module.PRODUCTION_CONFIG_PATH
    report_module.PRODUCTION_CONFIG_PATH = config_path
    try:
        with pytest.raises(FileNotFoundError):
            report.load_production_model("logistic", models_dir=tmp_path)
    finally:
        report_module.PRODUCTION_CONFIG_PATH = original


def test_load_calibration_table_reads_from_sibling_metadata(tmp_path):
    import joblib

    model_path = tmp_path / "logistic_20260101T000000Z.joblib"
    joblib.dump(_FakeModel(0.5), model_path)
    meta = {"metrics": {"calibration_table": [{"bin_low": 0.5, "bin_high": 0.6, "n": 200}]}}
    model_path.with_suffix(".json").write_text(json.dumps(meta))

    table = report.load_calibration_table(model_path)
    assert table == [{"bin_low": 0.5, "bin_high": 0.6, "n": 200}]


def test_load_calibration_table_returns_none_when_metadata_missing(tmp_path):
    model_path = tmp_path / "logistic_20260101T000000Z.joblib"
    assert report.load_calibration_table(model_path) is None


def test_load_feature_columns_reads_from_sibling_metadata(tmp_path):
    model_path = tmp_path / "logistic_20260101T000000Z.joblib"
    model_path.with_suffix(".json").write_text(json.dumps({"feature_columns": ["diff_win_pct", "same_stance"]}))
    assert report.load_feature_columns(model_path) == ["diff_win_pct", "same_stance"]


def test_load_feature_columns_falls_back_to_current_constant_when_metadata_missing(tmp_path):
    model_path = tmp_path / "logistic_20260101T000000Z.joblib"
    assert report.load_feature_columns(model_path) == model.FEATURE_COLUMNS


def test_build_card_report_uses_an_older_models_own_feature_list_not_the_current_one(tmp_path):
    """The exact bug this was written to catch: a model trained on an older,
    shorter feature list must still work via build_card_report, even though
    model.FEATURE_COLUMNS has since grown -- because build_card_report is
    told the model's own list explicitly, not defaulting to the module
    constant.
    """
    conn = _db_with_history(tmp_path)
    # An arbitrary strict subset of the current columns stands in for "what an
    # older, already-trained model's saved feature list looked like" -- the
    # point is the mechanism (use the passed-in list, not model.FEATURE_COLUMNS),
    # not any specific column names, so this must not depend on what's
    # currently in/out of the module constant.
    old_feature_columns = model.FEATURE_COLUMNS[:-1]
    assert len(old_feature_columns) < len(model.FEATURE_COLUMNS)  # sanity: the fixture actually differs

    fake_model = _FakeModel(prob_fighter_1_win=0.6)
    result = report.build_card_report(
        conn, fake_model, [("fA", "fB")], as_of_date="2026-06-01", feature_columns=old_feature_columns
    )
    assert list(fake_model.last_X.columns) == old_feature_columns
    assert result.iloc[0]["model_prob_fighter_1"] == pytest.approx(0.6)


def test_confidence_label_is_low_when_either_fighter_has_no_history():
    calibration_table = [{"bin_low": 0.5, "bin_high": 0.6, "n": 1000}]
    assert report._confidence_label(0, 5, model_prob=0.55, calibration_table=calibration_table) == "low"


def test_confidence_label_reflects_calibration_bucket_sample_size():
    calibration_table = [
        {"bin_low": 0.4, "bin_high": 0.5, "n": 5},
        {"bin_low": 0.5, "bin_high": 0.6, "n": 50},
        {"bin_low": 0.6, "bin_high": 0.7, "n": 500},
    ]
    # both fighters have history in all three cases -- only the bucket sample size should vary
    assert report._confidence_label(3, 3, model_prob=0.45, calibration_table=calibration_table) == "low"
    assert report._confidence_label(3, 3, model_prob=0.55, calibration_table=calibration_table) == "medium"
    assert report._confidence_label(3, 3, model_prob=0.65, calibration_table=calibration_table) == "high"


def test_confidence_label_falls_back_to_completeness_without_calibration_table():
    assert report._confidence_label(1, 4, model_prob=0.6, calibration_table=None) == "medium"
    assert report._confidence_label(5, 5, model_prob=0.6, calibration_table=None) == "high"
