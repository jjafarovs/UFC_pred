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
| 6 | Reporting / dashboard | **Done** (CLI report) — see below; Streamlit dashboard not built (optional per spec) |

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
  tests/
    test_fetcher.py              # HTML parsing, against saved fixtures
    test_cleaner.py                # JSON -> SQLite normalization
    test_bestfightodds.py           # odds page parsing
    test_bestfightodds_discovery.py   # fighter-profile-based historical odds lookup
    test_odds_matching.py               # cross-source fighter/fight matching, timestamp verification
    test_features.py                    # as-of-date feature engineering
    test_model.py                        # training/calibration/artifact roundtrip
    test_market.py                         # de-vig math, edge calc, per-fight consensus
    test_backtest.py                        # walk-forward retraining, bet sizing, ROI/drawdown
    test_report.py                            # upcoming-matchup report, model artifact loading
    test_no_leakage.py                          # leakage sanity checks
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
    just the last 2 years this way (84 target events, one fighter lookup
    each, ~9.5 minutes) brought coverage in that window from ~2% to
    **79.3%** (814 of 1,027 fights). Full-history odds coverage was
    deliberately *not* pursued the same way — the last 2 years is where a
    backtest has the most practical relevance, and extending further back
    is explicitly left as a "decide based on what these results show" step
    (see Walk-forward backtest).
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
predictions. This is a concrete, actionable lead for future feature/model
work — e.g. checking calibration quality split by favorite vs. underdog
separately (the aggregate calibration table hides this asymmetry), rather
than a reason to expect the next feature added will just fix things
generally.

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

See `db/schema.sql`. Five tables: `fighters`, `events`, `fights`,
`fight_stats` (one row per fight/fighter/round, `round=0` = fight total), and
`odds` (one row per fight/fighter/sportsbook/odds_type). `fights.event_date`
is denormalized from `events` specifically so every fight can be filtered by
date without a join -- that's the field Phase 2's as-of-date logic depends
on.

**Deliberately not stored**: ufcstats' own "career statistics" box (SLpM,
career win/loss totals, etc.) on the fighter page. Those are aggregated over
a fighter's *entire* career, including fights that postdate any given
historical bout -- storing them would be a standing invitation to leak the
future into a feature. All per-fight numbers for modeling must come from
`fight_stats`, rolled up as-of a date by `features.py` (Phase 2).
