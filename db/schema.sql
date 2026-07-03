-- ufc.db schema. All fight-timing fields are required because walk-forward
-- validation (see src/features.py) filters strictly on event_date.

CREATE TABLE IF NOT EXISTS fighters (
    fighter_id      TEXT PRIMARY KEY,      -- ufcstats.com fighter hex id
    name            TEXT NOT NULL,
    nickname        TEXT,
    height_in       REAL,                  -- inches
    reach_in        REAL,                  -- inches
    stance          TEXT,
    dob             TEXT,                  -- ISO date, nullable (not all fighters list DOB)
    source_url      TEXT,
    scraped_at      TEXT NOT NULL          -- ISO timestamp this row was last (re)fetched
);

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,      -- ufcstats.com event hex id
    name            TEXT NOT NULL,
    event_date      TEXT NOT NULL,         -- ISO date (YYYY-MM-DD), UFC event date
    location         TEXT,
    source_url      TEXT,
    scraped_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fights (
    fight_id            TEXT PRIMARY KEY,   -- ufcstats.com fight hex id
    event_id            TEXT NOT NULL REFERENCES events(event_id),
    event_date          TEXT NOT NULL,      -- denormalized copy of events.event_date for fast as-of filtering
    weight_class        TEXT,
    title_fight         INTEGER NOT NULL DEFAULT 0,
    scheduled_rounds    INTEGER,
    fighter_1_id        TEXT NOT NULL REFERENCES fighters(fighter_id),
    fighter_2_id        TEXT NOT NULL REFERENCES fighters(fighter_id),
    winner_id           TEXT REFERENCES fighters(fighter_id),  -- NULL if draw / no-contest
    result              TEXT NOT NULL,      -- 'fighter_1' | 'fighter_2' | 'draw' | 'nc'
    method              TEXT,               -- e.g. 'KO/TKO', 'SUB', 'U-DEC', 'S-DEC', 'M-DEC'
    method_detail       TEXT,
    end_round           INTEGER,
    end_time_sec        INTEGER,
    referee             TEXT,
    source_url          TEXT,
    scraped_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fights_event_date ON fights(event_date);
CREATE INDEX IF NOT EXISTS idx_fights_fighter1 ON fights(fighter_1_id);
CREATE INDEX IF NOT EXISTS idx_fights_fighter2 ON fights(fighter_2_id);

-- Deliberately a SEPARATE table from `fights`, not a nullable-result row in
-- it: fights_before()/build_feature_matrix() only ever query `fights`, and
-- keeping scheduled (not-yet-happened) bouts out of that table entirely
-- means there is no query path by which an upcoming fight could be treated
-- as historical training data, structurally, not just by convention. This
-- is a snapshot of "what's currently scheduled" -- cleaner.py replaces its
-- contents wholesale on each refresh rather than appending, since cards
-- change (fighters pulled/swapped) rather than growing monotonically like
-- completed history does.
CREATE TABLE IF NOT EXISTS upcoming_fights (
    fight_id        TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL,
    event_name      TEXT,
    event_date      TEXT NOT NULL,
    location        TEXT,
    weight_class    TEXT,
    title_fight     INTEGER NOT NULL DEFAULT 0,
    fighter_1_id    TEXT NOT NULL REFERENCES fighters(fighter_id),
    fighter_2_id    TEXT NOT NULL REFERENCES fighters(fighter_id),
    source_url      TEXT,
    scraped_at      TEXT NOT NULL
);

-- One row per (fight, fighter, round). round = 0 means fight-total row.
CREATE TABLE IF NOT EXISTS fight_stats (
    fight_id                TEXT NOT NULL REFERENCES fights(fight_id),
    fighter_id              TEXT NOT NULL REFERENCES fighters(fighter_id),
    round                   INTEGER NOT NULL,   -- 0 = total
    knockdowns              INTEGER,
    sig_str_landed          INTEGER,
    sig_str_attempted       INTEGER,
    total_str_landed        INTEGER,
    total_str_attempted     INTEGER,
    takedowns_landed        INTEGER,
    takedowns_attempted     INTEGER,
    sub_attempts            INTEGER,
    reversals               INTEGER,
    control_time_sec        INTEGER,
    PRIMARY KEY (fight_id, fighter_id, round)
);

-- Odds are stored per fighter per fight. A fight may have zero rows (no
-- market found/matched yet), or several (open/close from different books).
CREATE TABLE IF NOT EXISTS odds (
    odds_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id        TEXT REFERENCES fights(fight_id),   -- nullable: unmatched odds rows are kept for later matching
    fighter_id      TEXT REFERENCES fighters(fighter_id),
    fighter_name_raw TEXT NOT NULL,   -- name as it appeared at the odds source, for matching/debugging
    opponent_name_raw TEXT,
    sportsbook      TEXT NOT NULL,
    odds_type       TEXT NOT NULL,    -- 'open' | 'close' | 'live' | 'unverified' (see cleaner._derive_odds_type)
    american_odds   INTEGER NOT NULL,
    decimal_odds    REAL NOT NULL,
    event_date_raw  TEXT,             -- date as parsed at the odds source (may differ slightly from ufcstats)
    last_change_raw TEXT,             -- bestfightodds' own "Last change" text for this event's odds board, e.g. "Jun 28th 2026 13:58 UTC"
    last_change_utc TEXT,             -- last_change_raw parsed to ISO 8601 UTC; NULL if unparseable
    captured_at     TEXT NOT NULL,    -- ISO timestamp this row was fetched
    source          TEXT NOT NULL     -- 'bestfightodds' | 'theoddsapi'
);

CREATE INDEX IF NOT EXISTS idx_odds_fight ON odds(fight_id);
