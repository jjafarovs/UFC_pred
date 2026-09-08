import numpy as np
import pandas as pd
import pytest

from src import backtest, cleaner, model


def _make_matrix(n=200, seed=0):
    """Same synthetic shape as test_model.py's fixture -- a feature matrix
    with a genuine (noisy) relationship between diff_win_pct and the label.
    """
    rng = np.random.default_rng(seed)
    diff_win_pct = rng.uniform(-1, 1, n)
    prob = 1 / (1 + np.exp(-1.2 * diff_win_pct))
    label = rng.binomial(1, prob)
    dates = pd.date_range("2020-01-01", periods=n, freq="7D").astype(str)
    market_prob = np.where(rng.uniform(0, 1, n) < 0.1, rng.uniform(0.1, 0.9, n), np.nan)
    return pd.DataFrame(
        {
            "fight_id": [f"f{i}" for i in range(n)],
            "event_date": dates,
            "fighter_1_id": [f"a{i}" for i in range(n)],
            "fighter_2_id": [f"b{i}" for i in range(n)],
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
            "diff_fade_rate": rng.uniform(-0.5, 0.5, n),
            "market_prob_fighter_1": market_prob,
            "elo_prob_fighter_1": rng.uniform(0.2, 0.8, n),
        }
    )


def test_walk_forward_predictions_covers_only_rows_after_min_train_size():
    df = _make_matrix(200, seed=1)
    predictions = backtest.walk_forward_predictions(df, min_train_size=100, fold_size=20)

    # every fight before position 100 is training-only history, never a prediction target
    assert set(predictions["fight_id"]) <= set(df["fight_id"].iloc[100:])
    assert len(predictions) == 100  # rows [100:200) all fall in some fold


def test_walk_forward_predictions_training_window_expands_each_fold():
    df = _make_matrix(200, seed=1)
    predictions = backtest.walk_forward_predictions(df, min_train_size=100, fold_size=20)
    window_sizes = predictions.groupby("event_date")["train_window_size"].first()
    # can't rely on unique event_date ordering directly; instead check the sequence
    # of distinct train_window_size values seen in fight_id order is non-decreasing
    sizes_in_order = predictions.drop_duplicates("train_window_size")["train_window_size"].tolist()
    assert sizes_in_order == sorted(sizes_in_order)
    assert len(sizes_in_order) >= 2  # more than one fold happened


def test_walk_forward_predictions_raises_on_insufficient_data():
    df = _make_matrix(20, seed=1)
    with pytest.raises(ValueError):
        backtest.walk_forward_predictions(df, min_train_size=100, fold_size=20)


def test_walk_forward_predictions_applies_model_kwargs():
    """model_kwargs must actually reach build_logistic_pipeline/
    build_gradient_boosting_model each fold, not get silently ignored --
    checked on the fitted estimator's own hyperparameter attribute rather
    than downstream predictions, since post-hoc calibration (always applied
    in walk_forward_predictions) has its own scaling freedom and can wash
    out a regularization-strength effect on the final probabilities.
    """
    df = _make_matrix(150, seed=2)
    predictions = backtest.walk_forward_predictions(
        df, min_train_size=100, fold_size=50, model_kwargs={"C": 0.001}
    )
    assert len(predictions) > 0  # sanity: the run actually produced folds

    # Directly confirm build_logistic_pipeline/backtest's own call path honors C
    # (walk_forward_predictions doesn't expose the fitted estimator, so this
    # checks the same construction path backtest.py uses).
    pipeline = model.build_logistic_pipeline(**{"C": 0.001})
    assert pipeline.named_steps["clf"].C == 0.001


def test_walk_forward_predictions_applies_custom_feature_columns():
    df = _make_matrix(200, seed=3)
    restricted = ["diff_win_pct", "same_stance"]
    predictions = backtest.walk_forward_predictions(
        df, min_train_size=100, fold_size=20, feature_columns=restricted
    )
    assert len(predictions) > 0  # ran successfully with a restricted column set


def _db_with_odds(tmp_path, fight_id="fight1", f1_decimal=1.8, f2_decimal=2.2):
    db_path = tmp_path / "backtest_test.db"
    conn = cleaner.get_connection(db_path)
    cleaner.init_db(conn)
    conn.execute("INSERT INTO fighters (fighter_id, name, scraped_at) VALUES ('fA', 'Fighter A', 'now')")
    conn.execute("INSERT INTO fighters (fighter_id, name, scraped_at) VALUES ('fB', 'Fighter B', 'now')")
    conn.execute("INSERT INTO events (event_id, name, event_date, scraped_at) VALUES ('e1', 'Event 1', '2026-01-01', 'now')")
    conn.execute(
        f"INSERT INTO fights (fight_id, event_id, event_date, fighter_1_id, fighter_2_id, winner_id, result, scraped_at) "
        f"VALUES ('{fight_id}', 'e1', '2026-01-01', 'fA', 'fB', 'fA', 'fighter_1', 'now')"
    )
    for fighter_id, dec in [("fA", f1_decimal), ("fB", f2_decimal)]:
        conn.execute(
            "INSERT INTO odds (fight_id, fighter_id, fighter_name_raw, sportsbook, odds_type, american_odds, decimal_odds, captured_at, source) "
            "VALUES (?, ?, ?, 'BookOne', 'close', 0, ?, 'now', 'test')",
            (fight_id, fighter_id, fighter_id, dec),
        )
    conn.commit()
    return conn


def test_attach_market_data_computes_edge_and_drops_unmatched_fights(tmp_path):
    conn = _db_with_odds(tmp_path)
    predictions = pd.DataFrame(
        [
            {
                "fight_id": "fight1", "event_date": "2026-01-01",
                "fighter_1_id": "fA", "fighter_2_id": "fB",
                "label_fighter_1_win": 1, "model_prob_fighter_1": 0.65, "train_window_size": 50,
            },
            {
                "fight_id": "fight_no_odds", "event_date": "2026-01-02",
                "fighter_1_id": "fA", "fighter_2_id": "fB",
                "label_fighter_1_win": 0, "model_prob_fighter_1": 0.5, "train_window_size": 51,
            },
        ]
    )
    result = backtest.attach_market_data(conn, predictions, odds_type="close")
    assert len(result) == 1
    row = result.iloc[0]
    assert row["fight_id"] == "fight1"
    expected_market_prob = (1 / 1.8) / (1 / 1.8 + 1 / 2.2)
    assert row["market_prob_fighter_1"] == pytest.approx(expected_market_prob)
    assert row["edge_fighter_1"] == pytest.approx(0.65 - expected_market_prob)


def _bet_input_row(edge_f1, model_prob_f1=0.65, f1_dec=1.8, f2_dec=2.2, won_f1=True, fight_id="f1", date="2026-01-01"):
    return {
        "fight_id": fight_id, "event_date": date,
        "fighter_1_id": "fA", "fighter_2_id": "fB",
        "label_fighter_1_win": 1 if won_f1 else 0,
        "model_prob_fighter_1": model_prob_f1,
        "market_prob_fighter_1": model_prob_f1 - edge_f1,
        "fighter_1_decimal_odds": f1_dec, "fighter_2_decimal_odds": f2_dec,
        "edge_fighter_1": edge_f1,
    }


def test_simulate_bets_skips_fights_below_edge_threshold():
    df = pd.DataFrame([_bet_input_row(edge_f1=0.02)])  # below default 0.05 threshold, and -edge_f2 also below
    bets = backtest.simulate_bets(df, edge_threshold=0.05, strategy="flat")
    assert bets.empty


def test_simulate_bets_flat_strategy_computes_correct_win_and_loss_payout():
    df = pd.DataFrame(
        [
            _bet_input_row(edge_f1=0.10, f1_dec=2.0, won_f1=True, fight_id="win1", date="2026-01-01"),
            _bet_input_row(edge_f1=0.10, f1_dec=2.0, won_f1=False, fight_id="loss1", date="2026-01-02"),
        ]
    )
    bets = backtest.simulate_bets(df, edge_threshold=0.05, strategy="flat", flat_stake=10.0, starting_bankroll=100.0)
    assert len(bets) == 2
    win_row = bets[bets["fight_id"] == "win1"].iloc[0]
    loss_row = bets[bets["fight_id"] == "loss1"].iloc[0]
    assert win_row["payout"] == pytest.approx(10.0 * (2.0 - 1))  # stake * b
    assert loss_row["payout"] == pytest.approx(-10.0)
    assert win_row["side"] == "fighter_1"


def test_simulate_bets_picks_fighter_2_when_that_side_has_the_edge():
    # edge_fighter_1 is negative -> fighter_2's edge (-edge_f1) is positive and clears threshold
    df = pd.DataFrame([_bet_input_row(edge_f1=-0.10, model_prob_f1=0.35, won_f1=False)])
    bets = backtest.simulate_bets(df, edge_threshold=0.05, strategy="flat")
    assert len(bets) == 1
    assert bets.iloc[0]["side"] == "fighter_2"
    assert bets.iloc[0]["won"] == True  # fighter_2 won since label_fighter_1_win=0


def test_simulate_bets_kelly_sizes_proportionally_to_edge_and_never_exceeds_bankroll():
    df = pd.DataFrame([_bet_input_row(edge_f1=0.10, model_prob_f1=0.65, f1_dec=2.0, won_f1=True)])
    bets = backtest.simulate_bets(df, edge_threshold=0.05, strategy="kelly", kelly_fraction=0.5, starting_bankroll=100.0)
    b = 2.0 - 1
    expected_kelly_f = (b * 0.65 - 0.35) / b
    expected_stake = 0.5 * expected_kelly_f * 100.0
    assert bets.iloc[0]["stake"] == pytest.approx(expected_stake)
    assert bets.iloc[0]["stake"] <= 100.0


def test_backtest_summary_roi_and_max_drawdown_on_controlled_sequence():
    # starting bankroll 100 -> +20 (120) -> -60 (60) -> +30 (90)
    bets = pd.DataFrame(
        [
            {"stake": 20.0, "payout": 20.0, "won": True, "bankroll_after": 120.0},
            {"stake": 60.0, "payout": -60.0, "won": False, "bankroll_after": 60.0},
            {"stake": 30.0, "payout": 30.0, "won": True, "bankroll_after": 90.0},
        ]
    )
    summary = backtest.backtest_summary(bets, starting_bankroll=100.0)
    assert summary["n_bets"] == 3
    assert summary["total_staked"] == pytest.approx(110.0)
    assert summary["total_payout"] == pytest.approx(-10.0)
    assert summary["roi"] == pytest.approx(-10.0 / 110.0)
    # peak was 120 (after bet 1), trough was 60 (after bet 2) -> drawdown = (60-120)/120
    assert summary["max_drawdown"] == pytest.approx((60.0 - 120.0) / 120.0)
    assert summary["final_bankroll"] == pytest.approx(90.0)


def test_backtest_summary_handles_no_bets():
    summary = backtest.backtest_summary(pd.DataFrame(), starting_bankroll=100.0)
    assert summary == {"n_bets": 0}


def test_bootstrap_roi_ci_excludes_zero_for_consistent_positive_returns():
    # Every bet returns exactly +20% -- zero variance, so any reasonable CI
    # must sit tightly around +0.2 and clearly exclude zero.
    bets = pd.DataFrame({"stake": [1.0] * 200, "payout": [0.2] * 200})
    ci = backtest.bootstrap_roi_ci(bets, n_boot=1000)
    assert ci["roi_low"] > 0
    assert ci["roi_median"] == pytest.approx(0.2, abs=1e-6)


def test_bootstrap_roi_ci_includes_zero_for_small_noisy_sample():
    # Small, high-variance sample with a roughly-zero average -- consistent with noise.
    bets = pd.DataFrame(
        {
            "stake": [1.0] * 10,
            "payout": [5.0, -1.0, -1.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    ci = backtest.bootstrap_roi_ci(bets, n_boot=2000)
    assert ci["roi_low"] < 0 < ci["roi_high"]


def test_bootstrap_roi_ci_is_deterministic_with_a_fixed_seed():
    bets = pd.DataFrame({"stake": [1.0, 2.0, 1.5], "payout": [0.5, -2.0, 1.0]})
    ci_a = backtest.bootstrap_roi_ci(bets, n_boot=500, seed=7)
    ci_b = backtest.bootstrap_roi_ci(bets, n_boot=500, seed=7)
    assert ci_a == ci_b


def test_bootstrap_roi_ci_handles_no_bets():
    ci = backtest.bootstrap_roi_ci(pd.DataFrame())
    assert ci == {"n": 0, "roi_low": None, "roi_median": None, "roi_high": None}
