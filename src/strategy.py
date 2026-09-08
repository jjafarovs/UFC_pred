"""Model-confidence betting-signal rules for the dashboard.

All rules require the AVERAGE of the logistic and GBM production models'
win probabilities to clear a threshold, AND require neither individual
model to fall below/above a confirmation floor -- so a fight where one
model is much more confident than the other doesn't qualify just because
the average alone clears the bar.

Validated via a real walk-forward backtest (min_train_size=7500,
fold_size=200, 95% bootstrap CI on ROI) before being wired into the
dashboard -- see README's Walk-forward backtest section for the full
methodology and numbers. Numbers below are from the current model/data,
POST the elo_prob_fighter_1 feature addition (see model.py's FEATURE_COLUMNS
and features.build_elo_ratings) -- retraining with that feature changed
these numbers for every rule below, not just the new one:

- REFINED_RULE (avg>62%/<38%, confirm>=55%/<=45%, plus: both fighters must
  have >=3 tracked prior fights, and title fights are excluded): 351 bets,
  78.3% hit rate, ROI +8.7%, CI [+2.6%, +14.9%] -- significant. Two
  independent, well-motivated observations drive the filters: (1) fights
  involving a fighter with very few tracked bouts have inherently noisier
  rolling-window features (see features.py), and restricting to >=3 prior
  fights removed a real source of bad predictions rather than just cutting
  the sample; (2) title fights specifically have historically performed far
  worse than non-title fights under this rule's thresholds.
- REFINED_PLUS_RULE (same as REFINED_RULE, plus: excludes Heavyweight and
  Light Heavyweight -- divisions where one-punch KO power flattens the
  favorite's edge more than the model/market account for -- and only bets
  when the picked fighter's decimal odds are below 2.0, i.e. never lays a
  bet on a live underdog even if the confidence math says to): 297 bets,
  80.8% hit rate, ROI +11.4%, CI [+4.7%, +18.0%]. Found via targeted
  domain-informed filtering (not a blind grid search) and
  robustness-checked across multiple fold-size variants.
- ELO_RULE (same as REFINED_PLUS_RULE, plus: excludes a Southpaw-vs-Orthodox
  matchup on either side, regardless of who's picked -- a cross-stance fight
  is inherently noisier to call, both models and the market included): 219
  bets, 83.1% hit rate, ROI +14.5%, CI [+7.2%, +21.5%] -- the strongest and
  most robust result found so far, and it holds up: 13.0%/14.5%/13.6% ROI
  across fold sizes 150/200/300, never close to crossing zero. Named for
  the feature that made the whole model stronger, not just this one rule --
  see model.py's FEATURE_COLUMNS docstring on elo_prob_fighter_1: a
  career-long, opponent-strength- and finish-weighted Elo rating (100%
  coverage, unlike market_prob_fighter_1's ~26%) that improved every rule
  above once added, not only this one.

ORIGINAL_RULE/TIGHTER_RULE (avg>62%/<38% and avg>75%/<25%, both with no
extra filters) were tested in earlier iterations of this project and
retired: as the model/data evolved, both dropped out of statistical
significance (their bootstrap CIs now cross zero), so they're no longer
distinguishable from noise and aren't worth using. Historical
prediction_log rows logged under those rules are left in place as
historical record but are no longer scored by prediction_log.py.
"""
from __future__ import annotations

REFINED_RULE = {"avg_threshold": 0.62, "confirm_threshold": 0.55, "min_n_prior": 3, "exclude_title_fight": True}
REFINED_PLUS_RULE = {
    "avg_threshold": 0.62,
    "confirm_threshold": 0.55,
    "min_n_prior": 3,
    "exclude_title_fight": True,
    "exclude_weight_classes": frozenset({"Heavyweight", "Light Heavyweight"}),
    "max_decimal_odds": 2.0,
}
ELO_RULE = {
    "avg_threshold": 0.62,
    "confirm_threshold": 0.55,
    "min_n_prior": 3,
    "exclude_title_fight": True,
    "exclude_weight_classes": frozenset({"Heavyweight", "Light Heavyweight"}),
    "max_decimal_odds": 2.0,
    "exclude_cross_stance": True,
}


def bet_signal(
    logistic_prob: float | None,
    gbm_prob: float | None,
    avg_threshold: float,
    confirm_threshold: float,
    min_n_prior: int = 0,
    exclude_title_fight: bool = False,
    exclude_weight_classes: frozenset[str] = frozenset(),
    max_decimal_odds: float | None = None,
    exclude_cross_stance: bool = False,
    f1_n_prior: int | None = None,
    f2_n_prior: int | None = None,
    title_fight: bool | None = None,
    weight_class: str | None = None,
    fighter_1_decimal_odds: float | None = None,
    fighter_2_decimal_odds: float | None = None,
    fighter_1_stance: str | None = None,
    fighter_2_stance: str | None = None,
) -> str | None:
    """Returns 'fighter_1', 'fighter_2', or None (not bettable) for one
    matchup, given both models' fighter_1-win probability and a rule's
    thresholds/filters. `avg_threshold` and `confirm_threshold` are always
    applied symmetrically for fighter_2 (as `1 - threshold`) -- a
    fighter_1-favoring rule and its fighter_2-favoring mirror are the same
    rule, not two.

    `min_n_prior`/`exclude_title_fight`/`exclude_weight_classes`/
    `max_decimal_odds`/`exclude_cross_stance` are no-ops (default off) for
    REFINED_RULE, which doesn't set them -- the corresponding context
    kwargs (`f1_n_prior`, `f2_n_prior`, `title_fight`, `weight_class`,
    `fighter_1_decimal_odds`, `fighter_2_decimal_odds`, `fighter_1_stance`,
    `fighter_2_stance`) are safe to always pass regardless of which rule's
    dict is spread into this call. The odds ceiling is checked against the
    PICKED side's own decimal odds (strictly less than `max_decimal_odds`),
    not either fighter's -- missing odds data is treated as "don't bet",
    not "assume it qualifies". The cross-stance filter excludes a
    Southpaw-vs-Orthodox matchup regardless of which side is picked
    (Switch/unknown stances never trigger it); missing stance data for
    either fighter is treated as "not a cross-stance matchup" (don't block
    the bet), matching features.py's own same_stance convention of only
    asserting a relationship when both stances are actually known.
    """
    if logistic_prob is None or gbm_prob is None:
        return None
    if min_n_prior > 0 and (
        f1_n_prior is None or f2_n_prior is None or min(f1_n_prior, f2_n_prior) < min_n_prior
    ):
        return None
    if exclude_title_fight and title_fight:
        return None
    if exclude_weight_classes and weight_class in exclude_weight_classes:
        return None
    if exclude_cross_stance and fighter_1_stance and fighter_2_stance:
        if {fighter_1_stance, fighter_2_stance} == {"Southpaw", "Orthodox"}:
            return None

    avg_prob = (logistic_prob + gbm_prob) / 2
    min_prob = min(logistic_prob, gbm_prob)
    max_prob = max(logistic_prob, gbm_prob)
    if avg_prob > avg_threshold and min_prob >= confirm_threshold:
        side = "fighter_1"
    elif avg_prob < (1 - avg_threshold) and max_prob <= (1 - confirm_threshold):
        side = "fighter_2"
    else:
        return None

    if max_decimal_odds is not None:
        picked_odds = fighter_1_decimal_odds if side == "fighter_1" else fighter_2_decimal_odds
        if picked_odds is None or picked_odds >= max_decimal_odds:
            return None
    return side
