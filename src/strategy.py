"""Model-confidence betting-signal rules for the dashboard.

Both rules require the AVERAGE of the logistic and GBM production models'
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
"""
from __future__ import annotations

ORIGINAL_RULE = {"avg_threshold": 0.62, "confirm_threshold": 0.55}
TIGHTER_RULE = {"avg_threshold": 0.75, "confirm_threshold": 0.55}


def bet_signal(logistic_prob: float, gbm_prob: float, avg_threshold: float, confirm_threshold: float) -> str | None:
    """Returns 'fighter_1', 'fighter_2', or None (not bettable) for one
    matchup, given both models' fighter_1-win probability and a rule's two
    thresholds. `avg_threshold` and `confirm_threshold` are always applied
    symmetrically for fighter_2 (as `1 - threshold`) -- a fighter_1-favoring
    rule and its fighter_2-favoring mirror are the same rule, not two.
    """
    if logistic_prob is None or gbm_prob is None:
        return None
    avg_prob = (logistic_prob + gbm_prob) / 2
    min_prob = min(logistic_prob, gbm_prob)
    max_prob = max(logistic_prob, gbm_prob)
    if avg_prob > avg_threshold and min_prob >= confirm_threshold:
        return "fighter_1"
    if avg_prob < (1 - avg_threshold) and max_prob <= (1 - confirm_threshold):
        return "fighter_2"
    return None
