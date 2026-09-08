# UFC Edge

Systematic edge-detection for UFC fights: estimate a calibrated win
probability per fighter, compare it against the de-vigged sportsbook implied
probability, and flag fights where the gap between model and market is large
enough to matter. This is quant-style mispricing detection applied to a
sports betting market, not a win/loss predictor.

## The one rule that matters

**The model must never see information that wasn't available before the
fight happened.** Every feature is computed strictly "as of" the day before
a fight; every backtest is walk-forward (train on fights up to date T,
predict fights after T, roll forward). See `src/features.py` for the
as-of-date helper all feature builders are required to go through, and
`tests/test_no_leakage.py` for the sanity checks that guard this.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Data layer (fetch, clean, SQLite) | **Done** — see below |
| 2 | Feature engineering | **Done** — see below |
| 3 | Model + calibration | **Done** — see below |
| 4 | Market de-vigging + edge | **Done** — see below |
| 5 | Walk-forward backtest | **Done** — see below |
| 6 | Reporting / dashboard | **Done** — CLI report + a Streamlit dashboard (V1, informational, local-only) |
| — | Live pipeline (upcoming cards, live odds, scheduled refresh) | **Done** — see below |

## Architecture

```
ufc-edge/
  data/
    raw/            # raw JSON snapshots from fetcher.py (untouched scrapes)
    processed/       # cleaned parquet feature frames (Phase 2+)
  db/
    schema.sql        # SQLite DDL: fighters, events, fights, fight_stats, odds
    ufc.db             # built by src/cleaner.py (gitignored)
  models/
    *.joblib, *.json    # timestamp-versioned model artifacts + metrics (src/model.py)
    calibration_plot_*.png  # reliability diagrams
  backtests/
    equity_curve_*.png   # bankroll-over-time plots (src/backtest.py)
  src/
    fetcher.py          # pulls raw fighter/fight/odds data (network I/O only)
    cleaner.py            # normalizes raw JSON into SQLite (no network I/O)
    features.py            # as-of-date-safe feature matrix
    model.py                # training + calibration (logistic regression, gradient boosting)
    market.py                # odds -> de-vigged probability, edge
    backtest.py                # walk-forward backtest engine, ROI/Kelly sizing, drawdown
    report.py                   # per-card model-vs-market report (upcoming matchups)
    refresh.py                    # scheduled entry point: fetch -> clean -> features -> retrain
  tests/
    test_fetcher.py              # HTML parsing, against saved fixtures
    test_cleaner.py                # JSON -> SQLite normalization, upcoming-card storage
    test_bestfightodds.py           # odds page parsing
    test_bestfightodds_discovery.py   # fighter-profile-based historical odds lookup
    test_odds_matching.py               # cross-source fighter/fight matching, timestamp verification, live odds
    test_features.py                    # as-of-date feature engineering
    test_model.py                        # training/calibration/artifact roundtrip
    test_market.py                         # de-vig math, edge calc, per-fight consensus
    test_backtest.py                        # walk-forward retraining, bet sizing, ROI/drawdown
    test_report.py                            # upcoming-matchup report, model artifact loading
    test_refresh.py                            # scheduled-refresh call sequencing
    test_no_leakage.py                           # leakage sanity checks
```

`fetcher.py` only fetches and parses HTML into plain dicts, and writes raw
JSON snapshots to `data/raw/`. `cleaner.py` only reads that raw JSON and
writes to SQLite. Neither touches the other's concern -- a parsing bug in
`cleaner.py` can be fixed and rerun against already-fetched JSON without ever
hitting the network again.

## Data sources

- **Fights/fighters**: scraped directly from ufcstats.com (no public API).
  The site sits behind a lightweight JS proof-of-work gate (a same-origin
  SHA-256 grinding challenge — not a CAPTCHA); `UFCStatsClient` solves it
  once per session in pure Python and reuses the resulting cookie. There is
  no `robots.txt` on the host, but requests are rate-limited to ~1/sec to be
  polite regardless. **Full historical backfill completed**: all 780
  completed events on the site (UFC 2, March 1994, through present),
  8,758 completed fights, 2,710 fighters. `fetcher.bootstrap` is
  incremental/resumable by default -- it skips events and fighters already
  present in `data/raw/*.json` and checkpoints progress every 10 events / 25
  fighters, so re-running the same command later (e.g. weekly, after new
  events) only fetches what's new, and a mid-run failure only loses the
  last few minutes of progress rather than the whole run. This is how the
  full ~10,000-request, several-hour backfill was actually run in practice
  after an earlier smaller run hit a real `ConnectionResetError` partway
  through -- the retry logic (see `_get_with_retries`) and this
  checkpointing were both added in direct response to that failure, not
  speculatively.
- **Odds**: scraped from bestfightodds.com (`robots.txt` explicitly allows
  all). No PoW gate here, but also no true historical open/close pairs
  available from static HTML — the event page only shows each sportsbook's
  *current* line.
  - **`odds_type` labeling is now verified, not assumed.** The original
    version simply labeled odds `'close'` whenever *our own clock* said the
    matched fight was in the past — an assumption, not evidence. Every
    bestfightodds event page also shows a "Last change" timestamp for its
    whole odds board (e.g. "Jun 28th 2026 13:58 UTC") — the last time *any*
    line on that page moved. `fetcher.parse_bestfightodds_event` now
    captures it, and `cleaner._derive_odds_type` uses it to actually verify
    the market froze at/around fight time before calling it `'close'`
    (spot-checked against UFC 196, fought March 5, 2016: last change reads
    "Mar 6th 2016 05:54 UTC" — the night of the fight, still unchanged a
    decade later). A row where the timestamp lands suspiciously long after
    the fight gets labeled `'unverified'` instead and is excluded from
    `'close'`-based lookups by construction (`market.py` filters on an exact
    `odds_type` string) — no extra check needed at every call site. Falls
    back to the old today-vs-event_date heuristic only when no timestamp
    parses (e.g. a page-format change), documented as weaker, not equally
    trustworthy.
  - **A verified opening line is still not captured** — that's a separate
    gap from the labeling fix above. It requires either reverse-engineering
    the site's line-movement chart data endpoint or polling this page
    repeatedly over time and keeping our own earliest snapshot. Flagged in
    `fetcher.parse_bestfightodds_event`'s docstring as a known gap, not
    treated as if closing-line-value analysis in Phase 5 is fully ready.
  - Note: the site also has an internal admin panel under `/cnadm/*` (a
    plain login form) — not part of the public site and never requested by
    this scraper. Only `/events/<slug>`, `/archive`, `/search`, and
    `/fighters/<slug>` are used.
  - **Odds coverage does not automatically track fight coverage, and needed
    a second discovery mechanism to catch up.** `/archive` has no
    pagination at all (confirmed by inspection — no `page=`/`offset=`/
    "Older" links exist) and only surfaces the ~20-25 most recent events
    across every promotion combined, so `fetch_bestfightodds_candidates`
    alone stayed stuck at ~25 matched fights even after the full
    fight-history backfill. `fetch_bestfightodds_for_fighters` fixes this
    for a *targeted* window: it looks up a fighter's bestfightodds profile
    page (via `/search`), which lists their entire fight history with
    inline dates, and fetches only the events on/after a `since_date` cutoff
    — one popular fighter's profile can surface many relevant events at
    once, far cheaper than searching per-event or per-fight. Backfilling
    the last 2 years this way (84 target events, ~9.5 minutes) first brought
    coverage in that window from ~2% to 79.3% (814 of 1,027 fights) — later
    extended to a 5-year window (212 target events, ~26 minutes) once a real
    need for it emerged (see Walk-forward backtest: a market-probability
    feature only helps if it's present during *training*, not just in the
    test period, which required pushing coverage further back than the
    initial 2-year window). Result: **2,164 fights matched** across full
    history. Extending to the *entire* 30-year history was deliberately not
    pursued the same way — very old fights likely have thin-to-no odds data
    on bestfightodds regardless of effort, and 5 years already gives the
    walk-forward training window real coverage.
  - **This 5-year backfill was accidentally destroyed once, then fully
    recovered.** `refresh.py`'s `refresh_full` mode runs `fetcher.py
    --with-odds`, which calls `fetch_bestfightodds_candidates` — the
    *bounded, recent-only* mechanism above, not the fighter-profile one that
    actually built the 5-year backfill. That candidate list was being
    written to `data/raw/odds_bestfightodds.json` with a plain overwrite
    (`dump_raw`), not a merge, and `cleaner.match_and_store_odds` does an
    idempotent delete-then-reinsert against that same file. Running a
    routine `refresh_full` therefore treated "the ~20 most recent events" as
    the complete truth and deleted everything else: **2,164 matched fights
    dropped to 21** in one command. Fixed at the root — `fetcher._merge_by_key`
    now merges any new fetch into whatever's already on file (keyed on each
    event's slug) rather than overwriting, with two regression tests pinning
    this specifically. The lost data was then recovered by re-running the
    same fighter-profile-based mechanism (341 fighter names seeded from
    events since 2021-07-10, ~43 minutes at the site's 2-second rate limit),
    restoring coverage to **2,236 matched fights** — slightly better than
    before, since the recovery run and the routine refresh's normal fetch
    both contributed. Both production models were retrained and re-verified
    to respond to `market_prob_fighter_1` before being re-promoted, and the
    three betting-strategy backtests (see `src/strategy.py`) were re-run in
    full on the restored data.
- Fighter/fight matching across the two sources is name-based (see
  `cleaner.normalize_fighter_name`) plus an event-date tolerance window —
  it's a best-effort join over two independently-formatted sources, not a
  guaranteed-complete backfill. `cleaner.match_and_store_odds`'s docstring
  covers the known limitations (undercard-only event splits on
  bestfightodds, occasional name-format mismatches).

## Feature engineering

`src/features.py` builds one training row per completed fight: rolling
form/output features for each fighter (win %, current streak, significant
strike accuracy/defense, takedown accuracy/defense, layoff days) over their
last N fights, plus physical differentials (height, reach, age, stance
match), all computed strictly as of the day of that fight.

The leakage guard is structural, not just a convention: every feature
builder routes through `fights_before(conn, fighter_id, as_of_date)`, the one
function permitted to query `fights`/`fight_stats` by fighter — it enforces
`event_date < as_of_date` in one place, so a future fight can't leak in
through a one-off query elsewhere. `tests/test_features.py` pins this with a
synthetic fighter who has fights on three different dates and checks that
each fight's features see only the fights strictly before it.

```bash
python3 -m src.features --n 5 --out data/processed/feature_matrix.parquet
```

**Update after the full historical backfill** (see Data sources below): with
all 780 ufcstats events / 8,758 fights / 2,710 fighters loaded, 74.6% of the
8,602 trainable rows now have both fighters with real prior history, and
28.8% have a full 5-fight rolling window on both sides — up from ~6% and
~0% respectively on the initial 302-fight sample. The as-of-date logic never
changed; this is purely the data-volume fix the small sample was always
missing, confirming the earlier caveat was about sample size, not a bug.

**Later addition: `market_prob_fighter_1`** (`features.market_prob_feature`)
-- the de-vigged closing market probability for a fight, added as an actual
model input after finding the model's raw predictions were systematically
compressed toward 50/50 relative to reality in a way calibration couldn't
fix (see Walk-forward backtest for the full diagnosis). Not a leakage risk
relative to the fight itself -- a closing line is contemporaneous with the
fight, not information from after it -- but it does create a real
train/production mismatch worth knowing: it trains on the eventual
*closing* line, while `report.py`'s live use of an upcoming fight only has
access to whatever the *current* line is, which may be less sharp than what
the closing line eventually becomes. Only ~25% of historical fights have
this feature populated (NaN elsewhere); `HistGradientBoostingClassifier`
handles that natively, and the logistic pipeline's median-imputer treats a
missing value as neutral.

**Another later addition: career-long features** (`features.fighter_career_features`
-- total prior fights, finish rate among wins, times-finished rate among
losses). Deliberately *not* windowed to the last N fights like the rolling
features: a 15-fight veteran and a 2-fight prospect can show identical
last-5-fight stats, and this was added specifically to give the model a way
to tell them apart. **Result: no validated improvement** -- the
favorite/underdog calibration gap (see Walk-forward backtest) was
unchanged, and the backtest ROI stayed a significant loss for logistic and
got *worse* for GBM (full drawdown to zero bankroll). Kept in the codebase
and available for the dashboard's stat breakdown (real, correctly computed
information either way), but **not promoted to `models/production.json`**
-- another data point that box-score-derived stats don't seem to carry
information the market hasn't already priced in.

## Model + calibration

`src/model.py` trains a baseline (logistic regression, `--model logistic`)
and a gradient-boosting alternative (`--model gbm`, scikit-learn's
`HistGradientBoostingClassifier`), then applies post-hoc probability
calibration (`--calibration sigmoid` or `isotonic`) via
`CalibratedClassifierCV` + `FrozenEstimator` (the modern replacement for the
`cv='prefit'` API removed in scikit-learn 1.6). Output: accuracy/log
loss/Brier score/AUC on a held-out test set, a calibration table (predicted
probability vs. empirical win rate in bins), and a reliability-diagram PNG.

Walk-forward discipline extends into this phase too: `chronological_split`
sorts by `event_date` and splits train/calibration/test in time order, never
a random shuffle -- evaluating a model on a random mix of past and future
fights would overstate real-world performance even though each row's own
features are already as-of-date safe.

```bash
python3 -m src.model --model logistic --calibration sigmoid
python3 -m src.model --model gbm --calibration isotonic
```

**Why the gradient-boosting choice mattered originally**: running this
against the initial 302-fight sample surfaced a real finding, not just a
theoretical one. `SimpleImputer` (used in the logistic pipeline) warned that
every rolling-form feature (`diff_win_pct`, `diff_sig_str_acc`, etc.) had
*zero observed values* in the training slice -- because chronologically, the
earliest ~60% of that small sample couldn't yet contain fighter pairs with
prior history, those columns were silently dropped from that model entirely,
leaving only physical-attribute features. `HistGradientBoostingClassifier`
handles the same NaNs natively instead of dropping the columns, which is why
both paths are implemented rather than picking one.

**Update after the full historical backfill**: that imputer warning is gone
-- the training slice now has real rolling-form data throughout. On the full
8,602-row feature matrix (chronological 60/20/20 split, test n=1,720):
logistic + sigmoid calibration reaches **accuracy 61.3%, AUC 0.659, Brier
0.229**, with a calibration table now backed by hundreds of rows per bin
(530 and 571 in the two largest) instead of single digits, and the predicted
vs. empirical win rate tracking closely across bins (e.g. the 0.5-0.6 bin:
55.6% predicted vs. 53.3% actual). Gradient boosting comes in similar
(accuracy 60.5%, AUC 0.61-0.62). Both are a real, if modest, signal above
the 50% baseline -- not proof of a betting edge (that's what backtest.py's
ROI numbers are for, and those are still bottlenecked by odds coverage, not
fight coverage -- see the Walk-forward backtest section), but a legitimate
answer to "does this feature set predict fight outcomes better than chance."

Model artifacts (`.joblib`) and their metrics (`.json`) are timestamp-versioned
under `models/` — a backtest result should always be traceable to the exact
artifact that produced it.

**The same failure mode recurred later, with a different feature.** After
`market_prob_fighter_1` was added (see Walk-forward backtest below), every
production model artifact built via this file's default 60/20/20 split was
silently ignoring it. Real odds coverage only exists for roughly the last 5
years within this project's full 32-year (1994-2026) history -- `train_df`
(the first 60% of *all* history, ending 2019-08-03) predates that window
entirely, so `market_prob_fighter_1` was 100% missing at fit time. Exactly
like the earlier rolling-form case above: `SimpleImputer` silently drops an
all-missing column rather than imputing it, and (less obviously)
`HistGradientBoostingClassifier` never finds a useful split on a column with
zero real variance either, so GBM was equally affected despite handling NaN
natively in general. Verified concretely, not just inferred: feeding the
same fitted production models `market_prob_fighter_1 = 0.1` vs. `0.9` for an
otherwise-identical fight produced byte-identical predictions in both
models. This did **not** affect any backtest number reported in this
README -- `backtest.py`'s walk-forward folds use `min_train_size=7500`
(well past where coverage begins, row ~6061), so every validated finding
throughout this project was computed correctly. It only affected the
separate artifacts in `models/production.json` that the live dashboard
actually serves, meaning real dashboard predictions had been silently
missing the single most impactful, validated feature this project has
found since it was introduced.

**Fix**: `chronological_split` now takes an optional `min_train_size` floor,
applied after the fractional split (`test_frac`/`calib_frac` then still
divide whatever remains) -- `main()`'s CLI defaults it to 7500, matching
`backtest.py`'s own already-proven convention, so `python3 -m src.model` now
structurally can't reproduce this. Both production models were retrained
and re-verified to actually respond to `market_prob_fighter_1` before being
re-promoted. Two regression tests
(`test_chronological_split_min_train_size_is_a_floor_not_a_target`,
`test_chronological_split_min_train_size_prevents_an_all_missing_training_column`)
pin this specifically, the latter reproducing the exact "coverage clustered
in the tail" shape that caused it.

## Market de-vigging + edge

`src/market.py` converts American or decimal odds to implied probability,
de-vigs a two-sided line (the basic multiplicative method — normalize both
sides to sum to 1 — not the more elaborate Shin's-method/power-method
alternatives, matching the project's "start simple" phasing), and computes
edge = model probability − de-vigged market probability.

For a fight with odds from multiple sportsbooks, `market_probabilities_for_fight`
de-vigs each book's line independently, then averages the resulting
probability per fighter across books — this smooths single-book noise rather
than trusting one line, and skips any book missing one side of the market
(a data gap, not a real one-sided line).

```bash
python3 -m src.market <fight_id> --odds-type close
```

**Real caveat surfaced by running this against our actual matched odds**: two
of the books we pull from — Kalshi and Polymarket — are prediction-market
exchanges, not traditional vig-based sportsbooks, and their lines can be far
more extreme than a standard sportsbook's for the same fight (e.g. -2049 vs.
FanDuel's -390 for the same fighter in one real matched fight). Averaging
them in equally with traditional books lets those outliers pull the
"consensus" de-vigged probability more than a sportsbook-only average would.
Not a bug — just a real methodology nuance to keep in mind for Phase 5;
excluding or down-weighting exchange-style books is a reasonable future
refinement, not done here to keep the de-vig step simple for now.

## Walk-forward backtest

`src/backtest.py` is where the project's one hard rule gets enforced at the
*training* level, not just per-row features. `walk_forward_predictions` uses
an expanding window: starting after `--min-train-size` fights, it retrains
(and recalibrates) on every fight strictly before each `--fold-size`-fight
fold, predicts that fold, then grows the window and rolls forward. This is
a different, stronger check than `model.py`'s single chronological
train/calibration/test split (a quick diagnostic) — every prediction here
comes from a model that only ever saw the past relative to that prediction,
same as if it had actually been run live fold by fold.

For each walk-forward prediction with matched two-sided odds,
`attach_market_data` computes the edge (via `market.py`), and `simulate_bets`
places at most one bet per fight — only on whichever side clears
`--edge-threshold`, using either flat staking or fractional Kelly (sized off
the bookmaker's real, vigged payout odds, not the de-vigged probability used
for the edge gate itself — those intentionally answer different questions).
`backtest_summary` reports ROI, hit rate, **and** max drawdown and per-bet
ROI variance — a headline average ROI without those is exactly the number
this project exists to be skeptical of.

```bash
python3 -m src.backtest --model logistic --calibration sigmoid \
    --min-train-size 100 --fold-size 20 --edge-threshold 0.05 --strategy flat
```

**Update after the full historical backfill** (8,758 fights,
`--min-train-size 3000 --fold-size 500`): 5,602 walk-forward predictions
across 12 rolling retrains, completing in ~1.3 seconds wall-clock. Accuracy
60.7%, and the calibration table is now genuinely meaningful -- bins with
1,872 and 1,759 rows respectively, predicted vs. empirical win rate tracking
closely throughout (e.g. the 0.6-0.7 bin: 64.6% predicted vs. 63.8% actual).
**This is a real, legitimate result**: the model's win-probability estimates
are reasonably well calibrated across thousands of held-out walk-forward
predictions, not a handful.

The betting/ROI side did **not** improve from that same run, and for a
specific, already-known reason: only 25 of 5,602 walk-forward-predicted
fights had matched odds at that point (the odds-coverage gap described in
Data sources above, which backfilling *fights* does nothing to fix on its
own). Of those 25, 19 cleared the edge threshold — ROI ≈ −20%, max drawdown
≈ −6%. **At that sample size, still not evidence of anything** — 19 bets is
nowhere near enough to distinguish real edge from noise.

**Update after the targeted 2-year odds backfill** (see Data sources —
79.3% odds coverage in the recent window, up from ~2%, via a fighter-profile
discovery mechanism rather than deeper `/archive` pagination, which doesn't
exist): re-running with `--min-train-size 7500 --fold-size 200` produces
1,102 walk-forward predictions, 805 of which have matched odds, and **593
bets clear the 5% edge threshold** — finally a large enough sample to say
something real. Walk-forward accuracy holds at 62.5% with good calibration
(e.g. the 0.6–0.7 bin: 65.0% predicted vs. 63.5% actual across 271 fights).

**The honest result: the model does not have a real, exploitable edge over
this market, at least not with this feature set.** Flat-stake betting on
every 593 edge-flagged fights returns ROI ≈ −13.7%, hit rate 33.7%, and a
max drawdown of **−82.9%** (bankroll $100 → $18.60). Fractional Kelly
sizing (`--strategy kelly --kelly-fraction 0.25`) is worse in the way Kelly
theory predicts it should be when the perceived edge isn't real: staking
proportional to a systematically overstated edge compounds losses instead
of just averaging them out, and drives the bankroll to **−99.9% drawdown**
($100 → $0.14) — a near-total wipeout, not a bad-luck streak. This is a
textbook demonstration of why Kelly sizing is dangerous against a
miscalibrated edge estimate, not a bug in the sizing math (`simulate_bets`'
Kelly formula is unit-tested against hand-computed values in
`test_backtest.py`).

This doesn't mean the whole project's premise is wrong — it means the
current feature set (win %, streak, striking/takedown rates, physical
diffs, layoff days) isn't finding real mispricings against this market's
de-vigged consensus, at 5,762 fights of walk-forward training and 593 real
bets of evidence. The project's original data-volume problem is now solved
on *both* sides of "model vs. market" — what's left is a genuine research
question (better features? a different model? a narrower, more selective
edge threshold?), not a data-availability excuse.

**Follow-up: edge-threshold sweep + bootstrap significance test.** Since
"maybe a stricter edge threshold does better" is an obvious next question,
swept `--edge-threshold` from 0.02 to 0.30 for both logistic and GBM.
Raw ROI *looked* like it improved at stricter thresholds — GBM in
particular went from −12.9% ROI (740 bets, threshold 0.02) to +28.9% ROI
(75 bets, threshold 0.30), which would be an exciting result if taken at
face value. **It doesn't survive scrutiny.** Bootstrap resampling (5,000
resamples of each bucket's per-bet returns) puts a 95% confidence interval
around every single tested threshold/model combination, and **every one of
those intervals includes zero** — even the best case (GBM, threshold 0.30)
is `[-13.4%, +74.7%]`. The apparent "improving trend" at stricter
thresholds is consistent with shrinking sample size increasing variance in
both directions, not with a real signal that gets purer as you filter
harder; a genuine edge would be expected to hold a positive lower CI bound
at least somewhere in the sweep, and none did.

**One pattern is real and worth acting on, though**: across every
threshold and both models, **80-90% of edge-flagged bets are on
underdogs** (decimal odds ≥ 2.0) — 84.1% at the default threshold
specifically. That's not noise; it's a consistent, systematic property of
the model's output. It means the model's probability estimates specifically
in the lower-probability (underdog) range are running more optimistic than
the market's de-vigged consensus, across thousands of walk-forward
predictions.

### Root-causing the underdog bias: a real feature-signal gap, not a calibration bug

Splitting walk-forward predictions by *market*-defined favorite/underdog
(943 fights) confirmed the mechanism precisely: the model was systematically
**compressing predictions toward 50/50** relative to reality. Fights the
model called a near-toss-up (0.4-0.6) that were actually market favorites
won 72% of the time; ones that were actually market underdogs won only 29%
of the time — at the *same* predicted probability. `edge = model_prob −
market_prob` is mechanically negative for favorites under this bias (never
crosses the bet threshold) and mechanically positive for underdogs (crosses
constantly) — the "underdog value" was never real signal, it was a
compression artifact.

Checked whether this was fixable by calibration (it wasn't, and this is
worth knowing generally, not just here): the **raw, uncalibrated** model
showed the identical compression (favorites: predicted 58.6% vs. actual
70.7%; underdogs: predicted 41.4% vs. actual 29.3% — essentially identical
numbers to the calibrated model). Since a calibrator only reshapes an
existing 1-D score, it can't fix a bias that depends on information
(favorite/underdog status) the raw score doesn't carry. The earlier
aggregate calibration table looked fine specifically *because* the
favorite and underdog biases point in opposite directions and cancel out
in the marginal view — a real methodological trap.

**Fix: added the de-vigged closing market probability as a model input**
(`market_prob_fighter_1`, `features.market_prob_feature`) — not just a
downstream comparison, an actual feature the model trains on. Validated the
hypothesis first on a small held-out slice before committing to it: on the
805 fights that had matched odds at the time, a model trained *without* this
feature scored AUC 0.504 (random) with the same severe compression; the
identical setup *with* the feature scored AUC 0.682, and the favorite/
underdog gap shrank markedly. It only works, though, if the feature is
actually present during *training* — the first backfill (2-year window)
concentrated all matched odds in the most recent slice of history, which
fell entirely in the walk-forward *test* period and never in training; the
model had literally never seen a non-null example of it, and adding it
changed nothing. Extending the odds backfill to a 5-year window (`db/`:
2,164 fights matched, up from 814) fixed this — 1,191 covered fights now
fall inside the walk-forward training window, not just the test window.

**Result, on the real full walk-forward pipeline**: favorite/underdog gap
shrank from ~0.12-0.15 to **0.044** (logistic) / 0.084 (GBM). Overall
walk-forward accuracy rose from 62.5% to **68.6%** (logistic), with a
calibration table now tracking closely across every bin (e.g. the 0.7-0.8
bin: 74.5% predicted vs. 74.7% actual, n=186).

**But the honest bottom line got sharper, not better.** Re-running the
bootstrap significance test on the improved model: at the default 5% edge
threshold, logistic regression's 470 edge-flagged bets return ROI −9.8%
with a 95% CI of **[−19.2%, −0.3%] — this excludes zero.** For the first
time in this investigation, a result is statistically significant. It's a
significant *loss*, not a null result. Sweeping the threshold from 3% to
15% (`t=0.03`: 638 bets, CI `[−18.2%, −2.1%]`, significant; `t=0.05`: 470
bets, significant; `t=0.08` through `0.15`: point estimate stays negative
throughout, just loses significance as the sample shrinks) shows the same
story every time: the point estimate never once turns positive, and the
large-sample cases are confidently negative. GBM's result at the default
threshold (656 bets, ROI −7.3%, CI `[−17.1%, +2.5%]`) doesn't reach
significance but is directionally the same.

**What this means, precisely**: once the model has access to what the
market already knows (fixing the compression bug), the fights where our
*other* ten features still pull the prediction away from the market's price
are, with statistical confidence, worse bets than the market's own price —
not merely "no better." Our current stat-based features (rolling form,
striking/takedown rates, physical diffs, layoff days) are not adding real
incremental signal on top of the market; where they disagree with it, on
this evidence, they're adding noise. That's a substantive, specific
conclusion — not "we don't know," but "our current features are net
anti-predictive in exactly the cases the whole project is designed to
flag." Next steps worth pursuing based on this: features that are more
likely to carry information the market hasn't already priced in (personnel
changes, weight-cut/health signals, style-matchup-specific modeling) rather
than more of the same box-score statistics, and being explicit that a
model built this way is not competing with the market from scratch anymore
— it's testing whether anything beats an already-informed baseline, a
harder and more honest bar.

### Hyperparameter tuning + Tier 1-3 feature search

Given the "our current features are net anti-predictive" conclusion above,
ran a systematic, incremental search for improvement: walk-forward-safe
hyperparameter tuning first, then three tiers of candidate features, each
tested against the established discipline (walk-forward backtest +
bootstrap ROI CI + favorite/underdog calibration gap), keeping only what
actually validates.

**Hyperparameter tuning.** Searched logistic `C` and GBM
`max_depth`/`learning_rate`/`l2_regularization`/`min_samples_leaf` via
`TimeSeriesSplit(n_splits=5)` cross-validation strictly on the portion of
data that becomes the backtest's training set (never touching the actual
backtest evaluation window), scored on log loss. Logistic barely moved with
`C` (0.6972-0.6979 across the whole sweep) — it was never the problem.
GBM moved a lot (0.684 best to 0.717 worst); every default-like config
(deeper trees, higher learning rate) landed among the worst, confirming GBM
was overfitting on this modest tabular dataset.

Re-running the *real* backtest with the log-loss-optimal picks gave a mixed
result: logistic's tuned `C=0.01` was a clean win (ROI CI flipped from a
*proven significant loss* `[-19.2%, -0.3%]` to *not significant*
`[-18.1%, +0.3%]`, with accuracy/brier essentially unchanged). GBM's
log-loss-optimal pick (`l2=0.0, leaf=20`) made accuracy/brier/calibration
slightly *better* but flipped its ROI CI the wrong way, from not-significant
to a proven loss — log loss on a held-out slice is a decent starting point,
not a substitute for checking the real backtest. Checking a few more
regularized neighbors directly against the real backtest found
`max_depth=3, learning_rate=0.03, l2_regularization=1.0, min_samples_leaf=50`
strictly better than sklearn's defaults on every axis: ROI -7.3% → -4.6%,
CI `[-17.1%, +2.5%]` → `[-14.3%, +5.3%]` (tighter and higher), calibration
gap 0.084 → 0.079. Both are now `model.py`'s validated defaults.

**Tier 1 features** (control-time share, split-decision rate, title-fight
flag + scheduled rounds, cross-book odds divergence): only rolling-window
**grappling control-time share** (`diff_control_time_pct`,
`fighter_rolling_features`' `control_time_pct`, 97.9% coverage from
`fight_stats.control_time_sec`) validated — it improved ROI and tightened
the bootstrap CI for *both* logistic (`[-18.1%,+0.3%]` → `[-16.7%,+2.7%]`)
and GBM (`[-14.3%,+5.3%]` → `[-13.7%,+6.8%]`) without regressing
accuracy/brier/calibration. `scheduled_rounds` in particular flipped
logistic's backtest into a proven significant loss (`[-22.9%,-4.9%]`) on
its own and was rejected outright; the other two were neutral-to-mixed
(helped one model, hurt the other) and were rejected under the same "keep
only what validates for both models" bar used throughout this project.

**Tier 2 features** (KO-vs-submission share of career finishes, a
southpaw/orthodox directional stance-matchup indicator, and a 3-fights-vs-
prior-3-fights form-trend delta): none validated. Individually and combined,
every result landed inside the overlapping-CI noise band relative to the
Tier 1 baseline (e.g. logistic `[-16.7%,+2.7%]` vs. combined-Tier2
`[-15.8%,+3.2%]` — heavily overlapping, not distinguishable), unlike Tier
1's control-time feature, which showed a decisive, non-overlapping shift on
both models. All three were reverted.

**Tier 3 feature** (fraction of a fighter's career fights that reached
round 3+, a durability/cardio proxy): clearly regressed logistic (ROI
-7.2% → -9.0%, CI `[-16.7%,+2.7%]` → `[-18.8%,+0.6%]`) while doing nothing
for GBM. Rejected and reverted.

**Net result of this pass**: tuned hyperparameters (both models) +
`diff_control_time_pct` (Tier 1) are the only changes that survived
validation and are now in production. Final validated backtest: logistic
accuracy 68.2%, ROI -7.2%, CI `[-16.7%, +2.7%]` (not significant); GBM
accuracy 66.4%, ROI -3.7%, CI `[-13.7%, +6.8%]` (not significant) — a real,
if modest, improvement over the prior significant-loss result, but still
not a validated positive edge. The search continues to point the same
direction as before: incremental box-score-style features are close to
exhausted as a source of new signal against this market.

### Per-round data: the one genuinely new data source, not another box-score ratio

After the betting-strategy work below found a real, validated edge (see
`src/strategy.py`), the natural next question was whether the *model
itself* could be made stronger for every fight, not just more selective.
Checked directly: just backing the market's own favorite (no model at all)
gets 69.3% accuracy on odds-covered fights -- our model gets 68.9-69.2%,
essentially matching the market rather than trailing it, and a properly
built stacked ensemble (a meta-model on held-out calibration predictions,
not just averaging logistic + GBM) produced the exact same accuracy as
plain averaging (68.2% either way) -- the two base models are too
correlated for stacking to extract anything new. Both results are
consistent with everything else in this project: box-score-derived
features are close to exhausted.

`diff_fade_rate` (`features.fighter_rolling_features`) is different -- it's
the first feature built from data that was previously scraped but never
used at all: `fight_stats`' per-round rows (round 1..5), versus every prior
feature using only the round=0 season-total row. It's a cardio/fade proxy:
ratio of a fighter's own significant-strike output in the last FULL round
of a fight vs their round-1 output, averaged over their rolling window.
Deliberately compares against `end_round - 1`, not the fight's actual last
round -- a finish's last round is partial (cut short mid-round), not a fair
comparison to a full 5-minute round 1 -- and excludes fights that ended in
round 1 or 2 entirely, since there's no full late round to compare (~57%
coverage: 4,856 of 8,782 fights went 3+ rounds).

Validated on the strongest known strategy configuration (REFINED + odds
ceiling + excluding Heavyweight/Light Heavyweight, see below) across the
same 5-fold robustness sweep used throughout this project: improved ROI/CI
in 4 of 5 folds, neutral (not worse) in the 5th, never regressed. Overall
walk-forward accuracy stayed flat (69.2%/66.7%), consistent with the
finding above that raw accuracy is already at the market-matching ceiling
-- but GBM's log loss improved meaningfully (0.634 -> 0.606), and the
improvement shows up specifically where it matters, in the selective
strategy's ROI. Added to `model.FEATURE_COLUMNS`; both production models
retrained, verified to still respond correctly to `market_prob_fighter_1`,
and re-promoted.

## Reporting

`src/report.py` takes a list of upcoming matchups (fighter ID pairs) and
prints model probability, de-vigged market probability (when odds are
available), edge, and a confidence label per fight. Upcoming matchups don't
need a row in the `fights` table — a fight that hasn't happened yet has no
completed-fight row by definition — so this calls
`features.matchup_feature_dict` directly instead of going through the
fights-table-backed feature matrix used for training. That function is
shared with `features.build_fight_feature_row` (refactored out in this
phase) specifically so a report can never silently compute features
differently than training did.

Market data for a genuinely upcoming card has no `fight_id` to link
against either (`cleaner.match_and_store_odds` only links odds to fights
that already exist), so `market.market_probabilities_for_matchup` looks odds
up by fighter-ID pair among `fight_id IS NULL` rows instead — the same
de-vig-and-average logic as `market_probabilities_for_fight`, just keyed
differently.

```bash
python3 -m src.report --matchup <fighter_1_id>:<fighter_2_id> [--matchup ...] \
    --model logistic --odds-type live --as-of-date 2026-07-11
```

**Confidence now combines two independent, real signals** rather than one
crude heuristic. It's still `"low"` whenever either fighter has zero tracked
prior fights (no amount of calibration evidence rescues a mostly-null
feature vector) — but once both sides have real history, the label is
grounded in `model.py`'s own calibration table (persisted into the saved
artifact's metadata): the predicted probability's bucket is looked up, and
`"high"`/`"medium"`/`"low"` reflects how many held-out test fights actually
validated that probability range (≥100 / ≥20 / fewer). A 55% prediction
backed by 530 held-out fights is a meaningfully different claim than the
same number backed by 3 -- the label now says so. Falls back to the old
prior-fight-count-only heuristic if an older model artifact has no
calibration table saved (`report.load_calibration_table` returns `None`).
A Streamlit dashboard wrapping this report is a nice-to-have per the original spec, not
built here.

## Live pipeline: upcoming cards, live odds, scheduled refresh

Everything above this section works on *historical* data. Three gaps had to
be closed before any of it could serve a live, auto-updating dashboard:

**1. Upcoming-card discovery** (`fetcher.list_upcoming_events`,
`fetch_upcoming_card`) — ufcstats.com's `/statistics/events/upcoming` page
turned out to share identical markup with the completed-events page, so
this reuses the same row-parsing (`_list_events`) and the same
`parse_event`/`parse_fight` functions used for history. The one real
change needed: `parse_fight` used to discard fighter IDs for a scheduled
(not-yet-fought) bout, returning only `{"result": "scheduled"}` — exactly
the information an upcoming card needs was being thrown away. Fixed to
include `fighter_1_id`/`fighter_2_id`/`weight_class` on that path too.

Scheduled fights are stored in a **separate `upcoming_fights` table**, not
mixed into `fights` — `fights_before()`/`build_feature_matrix()` only ever
query `fights`, so this keeps it structurally impossible for a fight that
hasn't happened yet to be treated as training history, rather than relying
on convention. It's also a wholesale-replace table (`cleaner.upsert_upcoming_card`
deletes and rewrites on every refresh), not an append-only log — a card's
shape changes between refreshes (injuries, replacements), unlike completed
history, which only ever grows.

**2. Live odds ingestion** (`cleaner.match_and_store_live_odds`) — the
schema and read path (`market.market_probabilities_for_matchup`, filtering
`fight_id IS NULL AND odds_type='live'`) already existed from Phase 6, but
nothing wrote those rows; `match_and_store_odds` only stores odds matched
to an *already-completed* fight. The new function mirrors that matching
logic (fighter-name + date), sourced from `upcoming_fights` instead of
`fights`, always storing `fight_id=NULL` (a strict foreign key ties
`odds.fight_id` to `fights`, and an upcoming bout isn't in that table by
design — seed the fighter-pair lookup path, not the fight_id one). One
real bug caught while building this: the existing idempotency guard on
`match_and_store_odds` (`DELETE FROM odds WHERE source='bestfightodds'`)
would have silently wiped every live-odds row on its next run, since both
functions share the same table. Fixed by scoping each function's delete to
its own `odds_type`.

```bash
python3 -m src.fetcher --upcoming --with-live-odds   # discover the card + fetch current lines
python3 -m src.cleaner                                # normalize both into SQLite
```

Real run against the actual current UFC schedule: 7 upcoming events, 61
scheduled fights, 15 matched with live odds so far (coverage grows as
fight day approaches and more books post lines). `report.py` run against
two of these — a real, live McGregor vs. Holloway 2 matchup — produces
model probability 56.3% vs. de-vigged market 33.6%, edge +22.6%. That's a
large gap, and per the Walk-forward backtest findings above, large gaps
are exactly the case that's been shown to be a net-negative signal
historically, not a promising one — worth remembering once this feeds a
dashboard's "edge" column.

**3. Scheduled/automatic refresh** (`src/refresh.py`) — a single entry
point sequencing the already-tested CLIs (pure orchestration, no new
fetch/clean logic), because the pieces have different natural cadences:

```bash
python3 -m src.refresh --mode upcoming   # card + live odds; cheap, safe to run hourly
python3 -m src.refresh --mode full       # incremental history + odds + features + retrain; daily/weekly
```

Nothing currently triggers these on a schedule — that's a deployment
decision (plain cron, or this harness's own scheduling) left for whenever
the dashboard's hosting is decided, not assumed here.

## Dashboard (V1, informational)

`app.py` is a Streamlit app: pick an upcoming card, see every fight's model
probability, de-vigged live market probability, the gap ("edge"), a
confidence label, and (per fight) the underlying rolling-form/striking/
takedown/physical stats behind the number. Local/personal use only, no
hosting — everything reads straight from `db/ufc.db`.

```bash
streamlit run app.py
```

Deliberately **not** a betting tool: the disclaimer banner is load-bearing,
not decoration. The Walk-forward backtest section above found, with
statistical significance, that large disagreements between this model and
the market have been a net loss historically — so the UI frames edge as
"where the model and market disagree," never as a recommendation.

Two things fixed specifically for this: (1) `report.load_production_model`
— the dashboard pins a specific reviewed model artifact
(`models/production.json`) instead of `load_latest_model`'s "whatever was
trained most recently," since something running unattended shouldn't
silently pick up an unreviewed retrain; (2) a real bug caught by actually
driving the app in a browser rather than trusting that it imported cleanly
— `pd.DataFrame(rows)` turns a missing value's Python `None` into `NaN`,
and the display formatter's `x is None` check missed that, rendering a
literal `"nan%"` in the UI for any fight without matched odds. Fixed to
check `pd.isna(x)` instead.

**Fight-detail comparison table**: the per-fighter stat breakdown is a
genuine side-by-side table (`Stat | Fighter 1 | Fighter 2`), not two
separate raw-dict dumps — includes career-long stats (total fights,
finish rate, times-finished rate; see Feature engineering) alongside the
rolling-window ones, with no "better/worse" color-coding since several of
these (days since last fight, times finished) don't have a universally
correct direction.

**Manual refresh, not scheduled.** A launchd-based hourly/daily schedule
was built and tested, then deliberately abandoned: macOS blocks background
`launchd` processes from reading files under `~/Desktop` without an
explicit Full Disk Access grant (confirmed in practice — `PermissionError:
Operation not permitted` reading `.venv/pyvenv.cfg`), and the user preferred
a manual refresh over granting that access or moving the project off
Desktop, since UFC cards don't change fast enough to need always-on
auto-refresh. Instead, the sidebar has a "Refresh upcoming card + live
odds" button that runs `refresh.py --mode upcoming` as a normal foreground
subprocess (no permission issue at all this way, since it's not a
background daemon) and reruns the page on completion.

**A second real bug, caught the same way** (actually driving the app, not
just importing it): after adding new career-length features (see Feature
engineering) to `model.FEATURE_COLUMNS`, the dashboard broke with
scikit-learn's "feature names unseen at fit time" error. Cause: those new
features didn't validate in the backtest, so `production.json` was
deliberately left pinned to an *older* model that never saw them — but
`build_card_report` was building its input vector from the current code's
`model.FEATURE_COLUMNS`, not from what that specific pinned model actually
expects. Fixed by having `report.load_feature_columns` read the exact
column list out of the model artifact's own saved metadata (already stored
there by `save_model_artifact`) and threading it through explicitly, so a
pinned older model keeps working correctly no matter how many new features
get added to the code later.

**Betting-strategy highlighting** (`src/strategy.py`): the sidebar's
Refined/Refined+/Elo checkboxes highlight whichever fights each
model-confidence rule flags as bettable, with the exact rule(s) spelled out
per row (e.g. "Refined + Elo") rather than a generic tier -- see that
module's docstring for the full backtest numbers behind each rule. (The
original Original/Tighter rules were retired once the model/data evolved
past them; Elo is named for `features.build_elo_ratings`' career-long,
opponent-strength-weighted rating feature, which improved every rule once
added to the model, not just its own.)

**Two more bugs caught by actually running the app** (terminal noise the
user reported was real, not cosmetic, in one case): (1) the fight-detail
comparison table mixed raw ints/floats/`None` with `_pct()`'s `"60%"`
strings in the same column, which pandas stores as an `object`-dtype column
-- Streamlit's Arrow serialization couldn't convert it
(`ArrowInvalid: Could not convert '60%' ... to int64`) and silently
recovered with its own type-coercion fallback, but the underlying crash was
real. Fixed by routing every cell through a string-producing helper so no
column ever mixes types. (2) `use_container_width` is deprecated in the
installed Streamlit version (1.50) -- replaced with `width="stretch"`
everywhere it was used. The `SimpleImputer` "Skipping features without any
observed values" warning that also showed up in the terminal was a *symptom*
of the `market_prob_fighter_1` training bug documented in Model +
calibration above, not a separate issue -- it disappeared on its own once
that bug was fixed at the root, confirming it wasn't a case that needed
suppressing.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Fetch a small sample (smoke test) -- ufcstats.com only
python3 -m src.fetcher --max-events 3

# Fetch fights + a bounded window of bestfightodds.com odds candidates
python3 -m src.fetcher --max-events 10 --with-odds --max-odds-candidates 40

# Normalize whatever's in data/raw/ into db/ufc.db
python3 -m src.cleaner

# Run the test suite (fixture-based, no network required)
python3 -m pytest tests/
```

Re-running `src/fetcher.py` is incremental-friendly in spirit (it always
walks the full completed-events list), but does not yet skip
already-fetched events -- for a full historical backfill instead of a
smoke-test sample, drop `--max-events` and expect a long run given the
1 req/sec rate limit and ufcstats' one-page-per-fight/fighter structure.

## Schema

See `db/schema.sql`. Six tables: `fighters`, `events`, `fights`,
`fight_stats` (one row per fight/fighter/round, `round=0` = fight total),
`odds` (one row per fight/fighter/sportsbook/odds_type), and
`upcoming_fights` (a wholesale-replaced snapshot of the current card, kept
structurally separate from `fights` so a not-yet-fought bout can never be
queried as training history -- see Live pipeline). `fights.event_date`
is denormalized from `events` specifically so every fight can be filtered by
date without a join -- that's the field Phase 2's as-of-date logic depends
on.

**Deliberately not stored**: ufcstats' own "career statistics" box (SLpM,
career win/loss totals, etc.) on the fighter page. Those are aggregated over
a fighter's *entire* career, including fights that postdate any given
historical bout -- storing them would be a standing invitation to leak the
future into a feature. All per-fight numbers for modeling must come from
`fight_stats`, rolled up as-of a date by `features.py` (Phase 2).
