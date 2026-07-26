"""Model-confidence betting-signal rules for the dashboard.

All rules require the AVERAGE of the logistic and GBM production models'
win probabilities to clear a threshold, AND require neither individual
model to fall below/above a confirmation floor -- so a fight where one
model is much more confident than the other doesn't qualify just because
the average alone clears the bar.

Validated via a real walk-forward backtest (min_train_size=7500,
fold_size=200, 2024-05-04 to 2026-06-27, 95% bootstrap CI on ROI) before
being wired into the dashboard -- see README's Walk-forward backtest
section for the full methodology and numbers:

- ORIGINAL_RULE (avg>62%/<38%, confirm>=55%/<=45%): 565 bets, 75.9% hit
  rate, ROI +4.4%, CI [-0.9%, +9.7%] -- consistently positive across a
  fold-size robustness sweep, but the CI still includes zero. Not proven,
  but a real, stable improvement over the project's earlier edge-threshold
  approach (which was a proven loss).
- TIGHTER_RULE (avg>75%/<25%, confirm>=55%/<=45%): 211 bets, 85.8% hit
  rate, ROI +7.3%, CI [+1.0%, +13.3%] -- the first strategy in this
  project's history to produce a bootstrap CI that excludes zero. Found via
  a threshold sweep, so treat with the appropriate multiple-comparisons
  caution: it held up across a fold-size robustness check (3/5
  significant, the other 2 barely miss) and sits inside a smooth,
  monotonic hit-rate/threshold trend rather than being an isolated spike,
  which is what makes it more trustworthy than a typical "found by
  grid search" result -- but it is still one backtest on ~2 years of data,
  not a guarantee.
- REFINED_RULE (same thresholds as ORIGINAL_RULE, plus: both fighters must
  have >=3 tracked prior fights, and title fights are excluded): 305 bets,
  79.3% hit rate, ROI +10.9%, CI [+3.8%, +17.6%]. The strongest result
  found in this project's history -- significant across EVERY fold-size
  robustness variant tested (5/5), versus 3/5 for TIGHTER_RULE and 0/5 for
  ORIGINAL_RULE alone. Two independent, well-motivated observations drive
  it: (1) fights involving a fighter with very few tracked bouts have
  inherently noisier rolling-window features (see features.py), and
  restricting to >=3 prior fights removed a real source of bad predictions
  rather than just cutting the sample; (2) title fights specifically
  performed terribly under ORIGINAL_RULE (26 bets, 57.7% hit rate, ROI
  -19.9%) while non-title fights alone were already significant on their
  own (+5.6%, CI [+0.2%,+10.8%]). Same caveat as the others: one ~2-year
  backtest window (odds coverage doesn't extend further back), not a
  guarantee of future performance.
"""
from __future__ import annotations

ORIGINAL_RULE = {"avg_threshold": 0.62, "confirm_threshold": 0.55}
TIGHTER_RULE = {"avg_threshold": 0.75, "confirm_threshold": 0.55}
REFINED_RULE = {"avg_threshold": 0.62, "confirm_threshold": 0.55, "min_n_prior": 3, "exclude_title_fight": True}


def bet_signal(
    logistic_prob: float | None,
    gbm_prob: float | None,
    avg_threshold: float,
    confirm_threshold: float,
    min_n_prior: int = 0,
    exclude_title_fight: bool = False,
    f1_n_prior: int | None = None,
    f2_n_prior: int | None = None,
    title_fight: bool | None = None,
) -> str | None:
    """Returns 'fighter_1', 'fighter_2', or None (not bettable) for one
    matchup, given both models' fighter_1-win probability and a rule's
    thresholds/filters. `avg_threshold` and `confirm_threshold` are always
    applied symmetrically for fighter_2 (as `1 - threshold`) -- a
    fighter_1-favoring rule and its fighter_2-favoring mirror are the same
    rule, not two.

    `min_n_prior`/`exclude_title_fight` are no-ops (default off) for
    ORIGINAL_RULE/TIGHTER_RULE, which don't set them -- `f1_n_prior`,
    `f2_n_prior`, and `title_fight` are safe to always pass regardless of
    which rule's dict is spread into this call.
    """
    if logistic_prob is None or gbm_prob is None:
        return None
    if min_n_prior > 0 and (
        f1_n_prior is None or f2_n_prior is None or min(f1_n_prior, f2_n_prior) < min_n_prior
    ):
        return None
    if exclude_title_fight and title_fight:
        return None

    avg_prob = (logistic_prob + gbm_prob) / 2
    min_prob = min(logistic_prob, gbm_prob)
    max_prob = max(logistic_prob, gbm_prob)
    if avg_prob > avg_threshold and min_prob >= confirm_threshold:
        return "fighter_1"
    if avg_prob < (1 - avg_threshold) and max_prob <= (1 - confirm_threshold):
        return "fighter_2"
    return None
