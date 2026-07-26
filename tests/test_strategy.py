import pytest

from src import strategy


def test_bet_signal_fighter_1_requires_both_avg_and_confirm():
    # avg = 0.70 > 0.62, both models >= 0.55 -> qualifies.
    assert strategy.bet_signal(0.80, 0.60, **strategy.ORIGINAL_RULE) == "fighter_1"
    # avg = 0.70 > 0.62, but one model (0.50) is below the 0.55 confirm floor -> no bet.
    assert strategy.bet_signal(0.90, 0.50, **strategy.ORIGINAL_RULE) is None


def test_bet_signal_fighter_2_requires_both_avg_and_confirm():
    # avg = 0.30 < 0.38, both models <= 0.45 -> qualifies.
    assert strategy.bet_signal(0.20, 0.40, **strategy.ORIGINAL_RULE) == "fighter_2"
    # avg = 0.30 < 0.38, but one model (0.50) is above the 0.45 confirm ceiling -> no bet.
    assert strategy.bet_signal(0.10, 0.50, **strategy.ORIGINAL_RULE) is None


def test_bet_signal_middle_is_not_bettable():
    assert strategy.bet_signal(0.55, 0.50, **strategy.ORIGINAL_RULE) is None
    assert strategy.bet_signal(0.50, 0.50, **strategy.ORIGINAL_RULE) is None


def test_tighter_rule_is_strictly_more_selective_than_original():
    """Every fight the tighter rule flags, the original rule must also flag
    (same confirm floor, stricter avg threshold) -- this is what makes
    "flagged by both" and "flagged by tighter" the same set in practice.
    """
    import random

    rng = random.Random(0)
    for _ in range(500):
        p1, p2 = rng.uniform(0, 1), rng.uniform(0, 1)
        tighter_signal = strategy.bet_signal(p1, p2, **strategy.TIGHTER_RULE)
        if tighter_signal is not None:
            assert strategy.bet_signal(p1, p2, **strategy.ORIGINAL_RULE) == tighter_signal


def test_bet_signal_handles_missing_probabilities():
    assert strategy.bet_signal(None, 0.9, **strategy.ORIGINAL_RULE) is None
    assert strategy.bet_signal(0.9, None, **strategy.ORIGINAL_RULE) is None


def test_refined_rule_blocks_on_too_few_tracked_fights():
    # Otherwise-qualifying fight (same numbers as the fighter_1 case above).
    kwargs = dict(logistic_prob=0.80, gbm_prob=0.60, **strategy.REFINED_RULE)
    assert strategy.bet_signal(**kwargs, f1_n_prior=5, f2_n_prior=5, title_fight=False) == "fighter_1"
    # Either fighter below the min_n_prior=3 floor blocks the bet.
    assert strategy.bet_signal(**kwargs, f1_n_prior=2, f2_n_prior=5, title_fight=False) is None
    assert strategy.bet_signal(**kwargs, f1_n_prior=5, f2_n_prior=1, title_fight=False) is None
    # Missing prior-fight counts are treated as unknown, not as "assume qualifies".
    assert strategy.bet_signal(**kwargs, f1_n_prior=None, f2_n_prior=5, title_fight=False) is None


def test_refined_rule_blocks_title_fights():
    kwargs = dict(logistic_prob=0.80, gbm_prob=0.60, f1_n_prior=10, f2_n_prior=10, **strategy.REFINED_RULE)
    assert strategy.bet_signal(**kwargs, title_fight=False) == "fighter_1"
    assert strategy.bet_signal(**kwargs, title_fight=True) is None


def test_original_and_tighter_rules_ignore_prior_fight_and_title_context():
    """ORIGINAL_RULE/TIGHTER_RULE don't set min_n_prior/exclude_title_fight,
    so passing title_fight=True or very low prior-fight counts must not
    silently start filtering them out.
    """
    assert strategy.bet_signal(0.80, 0.60, f1_n_prior=0, f2_n_prior=0, title_fight=True, **strategy.ORIGINAL_RULE) == "fighter_1"
