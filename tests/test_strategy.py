import pytest

from src import strategy


def test_bet_signal_fighter_1_requires_both_avg_and_confirm():
    # avg = 0.70 > 0.62, both models >= 0.55 -> qualifies.
    assert strategy.bet_signal(0.80, 0.60, avg_threshold=0.62, confirm_threshold=0.55) == "fighter_1"
    # avg = 0.70 > 0.62, but one model (0.50) is below the 0.55 confirm floor -> no bet.
    assert strategy.bet_signal(0.90, 0.50, avg_threshold=0.62, confirm_threshold=0.55) is None


def test_bet_signal_fighter_2_requires_both_avg_and_confirm():
    # avg = 0.30 < 0.38, both models <= 0.45 -> qualifies.
    assert strategy.bet_signal(0.20, 0.40, avg_threshold=0.62, confirm_threshold=0.55) == "fighter_2"
    # avg = 0.30 < 0.38, but one model (0.50) is above the 0.45 confirm ceiling -> no bet.
    assert strategy.bet_signal(0.10, 0.50, avg_threshold=0.62, confirm_threshold=0.55) is None


def test_bet_signal_middle_is_not_bettable():
    assert strategy.bet_signal(0.55, 0.50, avg_threshold=0.62, confirm_threshold=0.55) is None
    assert strategy.bet_signal(0.50, 0.50, avg_threshold=0.62, confirm_threshold=0.55) is None


def test_bet_signal_handles_missing_probabilities():
    assert strategy.bet_signal(None, 0.9, avg_threshold=0.62, confirm_threshold=0.55) is None
    assert strategy.bet_signal(0.9, None, avg_threshold=0.62, confirm_threshold=0.55) is None


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


def test_refined_plus_rule_is_strictly_more_selective_than_refined():
    """Every fight Refined+ flags, Refined must also flag (same thresholds,
    plus extra weight-class/odds filters) -- this is what makes "flagged by
    both" and "flagged by Refined+" the same set in practice.
    """
    import random

    rng = random.Random(0)
    weight_classes = ["Lightweight", "Heavyweight", "Light Heavyweight", "Welterweight"]
    for _ in range(500):
        p1, p2 = rng.uniform(0, 1), rng.uniform(0, 1)
        wc = rng.choice(weight_classes)
        odds = rng.uniform(1.1, 4.0)
        kwargs = dict(
            f1_n_prior=10, f2_n_prior=10, title_fight=False,
            weight_class=wc, fighter_1_decimal_odds=odds, fighter_2_decimal_odds=odds,
        )
        plus_signal = strategy.bet_signal(p1, p2, **kwargs, **strategy.REFINED_PLUS_RULE)
        if plus_signal is not None:
            assert strategy.bet_signal(p1, p2, **kwargs, **strategy.REFINED_RULE) == plus_signal


def test_refined_plus_rule_excludes_heavyweight_and_light_heavyweight():
    kwargs = dict(
        logistic_prob=0.80, gbm_prob=0.60, f1_n_prior=10, f2_n_prior=10, title_fight=False,
        fighter_1_decimal_odds=1.5, fighter_2_decimal_odds=3.0, **strategy.REFINED_PLUS_RULE,
    )
    assert strategy.bet_signal(**kwargs, weight_class="Lightweight") == "fighter_1"
    assert strategy.bet_signal(**kwargs, weight_class="Heavyweight") is None
    assert strategy.bet_signal(**kwargs, weight_class="Light Heavyweight") is None


def test_refined_plus_rule_enforces_odds_ceiling_on_picked_side():
    kwargs = dict(
        logistic_prob=0.80, gbm_prob=0.60, f1_n_prior=10, f2_n_prior=10, title_fight=False,
        weight_class="Lightweight", **strategy.REFINED_PLUS_RULE,
    )
    # Picked side (fighter_1) is a live underdog price (>=2.0) -> blocked.
    assert strategy.bet_signal(**kwargs, fighter_1_decimal_odds=2.5, fighter_2_decimal_odds=1.4) is None
    # Picked side is a short-enough favorite (<2.0) -> qualifies.
    assert strategy.bet_signal(**kwargs, fighter_1_decimal_odds=1.5, fighter_2_decimal_odds=3.0) == "fighter_1"
    # Missing odds data for the picked side is treated as "don't bet".
    assert strategy.bet_signal(**kwargs, fighter_1_decimal_odds=None, fighter_2_decimal_odds=3.0) is None


def test_elo_rule_is_strictly_more_selective_than_refined_plus():
    """Every fight Elo flags, Refined+ must also flag (same thresholds/
    filters, plus the cross-stance exclusion on top)."""
    import random

    rng = random.Random(0)
    weight_classes = ["Lightweight", "Heavyweight", "Light Heavyweight", "Welterweight"]
    stances = ["Orthodox", "Southpaw", "Switch", None]
    for _ in range(500):
        p1, p2 = rng.uniform(0, 1), rng.uniform(0, 1)
        wc = rng.choice(weight_classes)
        odds = rng.uniform(1.1, 4.0)
        kwargs = dict(
            f1_n_prior=10, f2_n_prior=10, title_fight=False,
            weight_class=wc, fighter_1_decimal_odds=odds, fighter_2_decimal_odds=odds,
            fighter_1_stance=rng.choice(stances), fighter_2_stance=rng.choice(stances),
        )
        elo_signal = strategy.bet_signal(p1, p2, **kwargs, **strategy.ELO_RULE)
        if elo_signal is not None:
            assert strategy.bet_signal(p1, p2, **kwargs, **strategy.REFINED_PLUS_RULE) == elo_signal


def test_elo_rule_excludes_cross_stance_matchups_regardless_of_picked_side():
    kwargs = dict(
        logistic_prob=0.80, gbm_prob=0.60, f1_n_prior=10, f2_n_prior=10, title_fight=False,
        weight_class="Lightweight", fighter_1_decimal_odds=1.5, fighter_2_decimal_odds=3.0,
        **strategy.ELO_RULE,
    )
    assert strategy.bet_signal(**kwargs, fighter_1_stance="Orthodox", fighter_2_stance="Orthodox") == "fighter_1"
    assert strategy.bet_signal(**kwargs, fighter_1_stance="Southpaw", fighter_2_stance="Orthodox") is None
    assert strategy.bet_signal(**kwargs, fighter_1_stance="Orthodox", fighter_2_stance="Southpaw") is None
    # Switch isn't Southpaw-vs-Orthodox -- not blocked.
    assert strategy.bet_signal(**kwargs, fighter_1_stance="Orthodox", fighter_2_stance="Switch") == "fighter_1"


def test_elo_rule_missing_stance_data_does_not_block_the_bet():
    kwargs = dict(
        logistic_prob=0.80, gbm_prob=0.60, f1_n_prior=10, f2_n_prior=10, title_fight=False,
        weight_class="Lightweight", fighter_1_decimal_odds=1.5, fighter_2_decimal_odds=3.0,
        **strategy.ELO_RULE,
    )
    assert strategy.bet_signal(**kwargs, fighter_1_stance=None, fighter_2_stance="Orthodox") == "fighter_1"
    assert strategy.bet_signal(**kwargs, fighter_1_stance="Southpaw", fighter_2_stance=None) == "fighter_1"
