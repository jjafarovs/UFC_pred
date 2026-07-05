import numpy as np
import pandas as pd
import pytest

from src import model


def _make_matrix(n=100, seed=0):
    """Synthetic feature matrix shaped like features.py's real output, with a
    genuine (if noisy) relationship between diff_win_pct and the label so a
    fitted model has something real to learn and calibrate against.
    """
    rng = np.random.default_rng(seed)
    diff_win_pct = rng.uniform(-1, 1, n)
    prob = 1 / (1 + np.exp(-1.2 * diff_win_pct))  # mild signal, not near-perfectly separable
    label = rng.binomial(1, prob)
    dates = pd.date_range("2020-01-01", periods=n, freq="7D").astype(str)
    # ~10% odds coverage, matching the real data's sparse match rate
    market_prob = np.where(rng.uniform(0, 1, n) < 0.1, rng.uniform(0.1, 0.9, n), np.nan)
    return pd.DataFrame(
        {
            "fight_id": [f"f{i}" for i in range(n)],
            "event_date": dates,
            "label_fighter_1_win": label,
            "diff_win_pct": diff_win_pct,
            "diff_current_streak": rng.integers(-3, 3, n).astype(float),
            "diff_sig_str_acc": rng.uniform(-0.3, 0.3, n),
            "diff_sig_str_def": rng.uniform(-0.3, 0.3, n),
            "diff_td_acc": rng.uniform(-0.3, 0.3, n),
            "diff_td_def": rng.uniform(-0.3, 0.3, n),
            "diff_days_since_last_fight": rng.uniform(-200, 200, n),
            "diff_height_in": rng.uniform(-6, 6, n),
            "diff_reach_in": rng.uniform(-6, 6, n),
            "diff_age_years": rng.uniform(-10, 10, n),
            "same_stance": rng.choice([True, False], n),
            "diff_total_prior_fights": rng.integers(-10, 10, n).astype(float),
            "diff_finish_rate": rng.uniform(-0.5, 0.5, n),
            "diff_times_finished_rate": rng.uniform(-0.5, 0.5, n),
            "diff_control_time_pct": rng.uniform(-0.3, 0.3, n),
            "market_prob_fighter_1": market_prob,
        }
    )


def test_chronological_split_respects_time_order_and_is_non_overlapping():
    df = _make_matrix(100)
    # shuffle row order to prove the split re-sorts by date rather than trusting input order
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    train_df, calib_df, test_df = model.chronological_split(df, calib_frac=0.2, test_frac=0.2)

    assert len(train_df) == 60
    assert len(calib_df) == 20
    assert len(test_df) == 20
    assert train_df["event_date"].max() <= calib_df["event_date"].min()
    assert calib_df["event_date"].max() <= test_df["event_date"].min()


def test_chronological_split_raises_on_too_few_rows():
    df = _make_matrix(3)
    with pytest.raises(ValueError):
        model.chronological_split(df, calib_frac=0.2, test_frac=0.2)


def test_logistic_pipeline_and_calibration_produce_valid_probabilities():
    df = _make_matrix(200, seed=2)
    train_df, calib_df, test_df = model.chronological_split(df)
    X_train, y_train = model.make_xy(train_df)
    X_calib, y_calib = model.make_xy(calib_df)
    X_test, y_test = model.make_xy(test_df)

    base = model.build_logistic_pipeline()
    base.fit(X_train, y_train)
    calibrated = model.calibrate_model(base, X_calib, y_calib, method="sigmoid")

    proba = calibrated.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    metrics = model.evaluate(calibrated, X_test, y_test)
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["log_loss"] is not None


def test_gradient_boosting_handles_missing_features_natively():
    df = _make_matrix(200, seed=3)
    df.loc[df.index[:50], "diff_win_pct"] = np.nan  # simulate fighters with no tracked history
    train_df, calib_df, test_df = model.chronological_split(df)
    X_train, y_train = model.make_xy(train_df)
    X_test, y_test = model.make_xy(test_df)

    gbm = model.build_gradient_boosting_model()
    gbm.fit(X_train, y_train)  # must not raise despite NaNs, unlike the logistic pipeline's imputer
    proba = gbm.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calibration_table_bucketing_matches_hand_computed_rates():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.05, 0.15, 0.15, 0.85, 0.95, 0.95])
    table = model.calibration_table(y_true, y_prob, n_bins=10)

    bin_0 = table[(table["bin_low"] <= 0.05) & (table["bin_high"] > 0.05)].iloc[0]
    assert bin_0["n"] == 1
    assert bin_0["empirical_win_rate"] == 1.0

    bin_9 = table[table["bin_low"] == 0.9].iloc[0]
    assert bin_9["n"] == 2
    assert bin_9["empirical_win_rate"] == 0.5  # y_true = [1, 0] at prob ~0.95


def test_save_and_load_model_artifact_roundtrip(tmp_path):
    df = _make_matrix(150, seed=4)
    train_df, calib_df, test_df = model.chronological_split(df)
    X_train, y_train = model.make_xy(train_df)
    X_calib, y_calib = model.make_xy(calib_df)
    X_test, _ = model.make_xy(test_df)

    base = model.build_logistic_pipeline()
    base.fit(X_train, y_train)
    calibrated = model.calibrate_model(base, X_calib, y_calib)

    path = model.save_model_artifact(
        calibrated, model.FEATURE_COLUMNS, {"raw": {}, "calibrated": {}}, name="test", models_dir=tmp_path
    )
    assert path.exists()
    meta_path = path.with_suffix(".json")
    assert meta_path.exists()

    import joblib

    reloaded = joblib.load(path)
    np.testing.assert_array_equal(
        reloaded.predict_proba(X_test)[:, 1], calibrated.predict_proba(X_test)[:, 1]
    )
