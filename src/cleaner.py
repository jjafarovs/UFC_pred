"""Normalizes raw JSON fetched by fetcher.py into the SQLite schema.

Kept as a separate step from fetcher.py so that (a) a parsing bug can be
fixed and re-run against already-fetched raw JSON without hitting the
network again, and (b) the DB write path has one entry point regardless of
whether data came from a fresh scrape or an old raw/ snapshot.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ufc.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize_fighter_name(name: str) -> str:
    """'Khalil Rountree Jr.' -> 'khalil rountree'; used only for fuzzy matching
    fighter names between ufcstats.com and bestfightodds.com, which format
    suffixes/accents/punctuation differently. Never used as a stored key.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z]+", ascii_name.lower())
    return " ".join(t for t in tokens if t not in _NAME_SUFFIXES)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_event_date(raw: str | None) -> str | None:
    """'July 11, 2026' -> '2026-07-11'. Returns None if unparseable."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


def parse_dob(raw: str | None) -> str | None:
    """'Mar 05, 1993' -> '1993-03-05'. Returns None if unparseable/missing."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%b %d, %Y").date().isoformat()
    except ValueError:
        return None


_LAST_CHANGE_RE = re.compile(r"([A-Za-z]+) (\d{1,2})(?:st|nd|rd|th) (\d{4}) (\d{1,2}):(\d{2}) UTC")


def parse_last_change(raw: str | None) -> str | None:
    """'Jun 28th 2026 13:58 UTC' -> ISO 8601 UTC string. Returns None if
    missing/unparseable -- callers must treat that as "no verification
    evidence available", not as a silent stand-in for any particular date.
    """
    if not raw:
        return None
    m = _LAST_CHANGE_RE.match(raw.strip())
    if not m:
        return None
    month, day, year, hour, minute = m.groups()
    try:
        dt = datetime.strptime(f"{month} {day} {year} {hour}:{minute}", "%b %d %Y %H:%M")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat()


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def load_raw(name: str) -> list | dict:
    path = RAW_DIR / f"{name}.json"
    return json.loads(path.read_text())


def upsert_fighters(conn: sqlite3.Connection, fighters: list[dict]) -> int:
    rows = [
        (
            f["fighter_id"],
            f["name"],
            f.get("nickname"),
            f.get("height_in"),
            f.get("reach_in"),
            f.get("stance"),
            parse_dob(f.get("dob_raw")),
            f.get("source_url"),
            _now(),
        )
        for f in fighters
    ]
    conn.executemany(
        """
        INSERT INTO fighters (fighter_id, name, nickname, height_in, reach_in, stance, dob, source_url, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fighter_id) DO UPDATE SET
            name=excluded.name, nickname=excluded.nickname, height_in=excluded.height_in,
            reach_in=excluded.reach_in, stance=excluded.stance, dob=excluded.dob,
            source_url=excluded.source_url, scraped_at=excluded.scraped_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_events(conn: sqlite3.Connection, events: list[dict]) -> int:
    rows = []
    skipped = []
    for e in events:
        event_date = parse_event_date(e.get("event_date_raw") or e.get("event_date"))
        if event_date is None:
            skipped.append(e.get("event_id"))
            continue
        rows.append(
            (
                e["event_id"],
                e["name"],
                event_date,
                e.get("location"),
                e.get("source_url"),
                _now(),
            )
        )
    if skipped:
        print(f"cleaner: skipped {len(skipped)} events with unparseable dates: {skipped}")
    conn.executemany(
        """
        INSERT INTO events (event_id, name, event_date, location, source_url, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            name=excluded.name, event_date=excluded.event_date, location=excluded.location,
            source_url=excluded.source_url, scraped_at=excluded.scraped_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_fights(conn: sqlite3.Connection, fights: list[dict]) -> tuple[int, int]:
    """Writes fights + fight_stats. Silently drops fights whose result is not
    a decided outcome (scheduled/unknown) -- those have no fight_stats and no
    place in a walk-forward backtest.
    """
    event_dates = dict(conn.execute("SELECT event_id, event_date FROM events").fetchall())

    fight_rows = []
    stat_rows = []
    dropped = []
    for f in fights:
        if f.get("result") not in ("fighter_1", "fighter_2", "draw", "nc"):
            dropped.append(f.get("fight_id"))
            continue
        event_date = event_dates.get(f["event_id"])
        if event_date is None:
            dropped.append(f.get("fight_id"))
            continue

        winner_id = f.get("winner_id")
        fight_rows.append(
            (
                f["fight_id"],
                f["event_id"],
                event_date,
                f.get("weight_class"),
                int(bool(f.get("title_fight"))),
                f.get("scheduled_rounds"),
                f["fighter_1_id"],
                f["fighter_2_id"],
                winner_id,
                f["result"],
                f.get("method"),
                f.get("method_detail"),
                f.get("end_round"),
                f.get("end_time_sec"),
                f.get("referee"),
                f.get("source_url"),
                _now(),
            )
        )

        stats = f.get("stats") or {}
        for i, fighter_key in enumerate(("fighter_1_id", "fighter_2_id"), start=1):
            total = stats.get(f"fighter_{i}_total")
            if total:
                stat_rows.append(_stat_row(f["fight_id"], f[fighter_key], 0, total))
        for entry in stats.get("per_round", []):
            fighter_id = f["fighter_1_id"] if entry["fighter_idx"] == 1 else f["fighter_2_id"]
            stat_rows.append(_stat_row(f["fight_id"], fighter_id, entry["round"], entry))

    if dropped:
        print(f"cleaner: dropped {len(dropped)} non-completed/unmatched fights: {dropped}")

    conn.executemany(
        """
        INSERT INTO fights (
            fight_id, event_id, event_date, weight_class, title_fight, scheduled_rounds,
            fighter_1_id, fighter_2_id, winner_id, result, method, method_detail,
            end_round, end_time_sec, referee, source_url, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fight_id) DO UPDATE SET
            event_id=excluded.event_id, event_date=excluded.event_date,
            weight_class=excluded.weight_class, title_fight=excluded.title_fight,
            scheduled_rounds=excluded.scheduled_rounds, fighter_1_id=excluded.fighter_1_id,
            fighter_2_id=excluded.fighter_2_id, winner_id=excluded.winner_id,
            result=excluded.result, method=excluded.method, method_detail=excluded.method_detail,
            end_round=excluded.end_round, end_time_sec=excluded.end_time_sec,
            referee=excluded.referee, source_url=excluded.source_url, scraped_at=excluded.scraped_at
        """,
        fight_rows,
    )
    conn.executemany(
        """
        INSERT INTO fight_stats (
            fight_id, fighter_id, round, knockdowns, sig_str_landed, sig_str_attempted,
            total_str_landed, total_str_attempted, takedowns_landed, takedowns_attempted,
            sub_attempts, reversals, control_time_sec
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fight_id, fighter_id, round) DO UPDATE SET
            knockdowns=excluded.knockdowns, sig_str_landed=excluded.sig_str_landed,
            sig_str_attempted=excluded.sig_str_attempted, total_str_landed=excluded.total_str_landed,
            total_str_attempted=excluded.total_str_attempted, takedowns_landed=excluded.takedowns_landed,
            takedowns_attempted=excluded.takedowns_attempted, sub_attempts=excluded.sub_attempts,
            reversals=excluded.reversals, control_time_sec=excluded.control_time_sec
        """,
        stat_rows,
    )
    conn.commit()
    return len(fight_rows), len(stat_rows)


def remove_completed_fights_from_upcoming(conn: sqlite3.Connection) -> int:
    """Deletes any `upcoming_fights` row whose fight_id now has a completed
    record in `fights`. Needed because `--mode full` (refresh.py) fetches
    completed results but never re-fetches/re-writes the upcoming-card
    snapshot -- without this, a card that just finished lingers in
    `upcoming_fights` until the next `--mode upcoming` run happens to
    wholesale-replace it, and the dashboard's Upcoming tab keeps showing it
    with a stale LIVE odds line instead of the fight's real, now-available
    CLOSING line (which the Past Cards tab correctly uses) -- the two tabs
    showed different numbers for the same fight because of exactly this.
    Called unconditionally after every upsert_fights, regardless of mode,
    so this invariant (a fight_id is never "upcoming" once it's completed)
    can't be violated by which refresh mode ran or in what order.
    """
    cur = conn.execute("DELETE FROM upcoming_fights WHERE fight_id IN (SELECT fight_id FROM fights)")
    conn.commit()
    return cur.rowcount


def _stat_row(fight_id: str, fighter_id: str, round_num: int, s: dict) -> tuple:
    return (
        fight_id,
        fighter_id,
        round_num,
        s.get("knockdowns"),
        s.get("sig_str_landed"),
        s.get("sig_str_attempted"),
        s.get("total_str_landed"),
        s.get("total_str_attempted"),
        s.get("takedowns_landed"),
        s.get("takedowns_attempted"),
        s.get("sub_attempts"),
        s.get("reversals"),
        s.get("control_time_sec"),
    )


def _american_to_decimal(american_odds: int) -> float:
    if american_odds > 0:
        return round(american_odds / 100 + 1, 4)
    return round(100 / abs(american_odds) + 1, 4)


CLOSE_VERIFICATION_BUFFER_DAYS = 3  # fights run late-night US time -> early UTC the next day(s)


def _derive_odds_type(fight_event_date: str, last_change_utc: str | None, today: date) -> str:
    """'close' requires evidence the market actually stopped moving at/around
    the fight, not just that our own clock says the fight is in the past --
    verified via bestfightodds' own "Last change" timestamp (see
    fetcher.parse_bestfightodds_event), which empirically lands within a day
    of the real fight date for genuinely settled events (spot-checked
    against UFC 196, 2016: fought Mar 5, last change Mar 6 05:54 UTC, still
    unchanged a decade later).

    Falls back to the coarser today-vs-event_date heuristic only when no
    timestamp could be parsed (e.g. a page-format change) -- that fallback
    is documented as weaker, not silently treated as equally trustworthy.

    Returns 'unverified' when a timestamp IS available but lands well after
    the fight's date -- that's a real data-quality signal (wrong event
    matched, or the book's line was edited well after the fact) that
    downstream code should not silently trust as a genuine closing price.
    market.py's lookups only ever filter on an exact odds_type, so an
    'unverified' row is automatically excluded from 'close'-based analysis
    rather than needing an extra check at every call site.
    """
    fight_date = datetime.fromisoformat(fight_event_date).date()
    if last_change_utc is None:
        return "close" if fight_date <= today else "live"
    last_change_date = datetime.fromisoformat(last_change_utc).date()
    if last_change_date <= fight_date + timedelta(days=CLOSE_VERIFICATION_BUFFER_DAYS):
        return "close"
    return "unverified"


def match_and_store_odds(conn: sqlite3.Connection, bfo_events: list[dict], date_tolerance_days: int = 2) -> int:
    """Match bestfightodds.com events to our fights by (normalized fighter-name
    pair, event date within `date_tolerance_days`), then insert one odds row
    per fighter per sportsbook. Unmatched bestfightodds fighters/events are
    silently skipped -- this is a best-effort join over two independently
    formatted sources (different event naming, occasional undercard-only
    "event" splits on bestfightodds), not a guaranteed-complete backfill. See
    fetcher.fetch_bestfightodds_candidates for the known coverage limits.

    Idempotent: clears previously-stored *non-live* bestfightodds rows
    before re-inserting, so re-running this against a growing
    odds_bestfightodds.json (e.g. after a later, wider backfill) doesn't
    duplicate rows the way a plain INSERT without this would -- unlike
    fighters/fights/events, the odds table has no natural per-row upsert
    key to conflict on (a fighter can have several legitimate rows: one per
    sportsbook). Scoped to exclude odds_type='live' specifically so this
    doesn't wipe out match_and_store_live_odds' rows when both run in the
    same cleaner pass -- they're two independent, differently-scoped
    idempotent writers sharing one table.
    """
    conn.execute("DELETE FROM odds WHERE source = 'bestfightodds' AND odds_type != 'live'")

    our_fights = conn.execute(
        """
        SELECT f.fight_id, f.event_date, f.fighter_1_id, f.fighter_2_id,
               f1.name AS f1_name, f2.name AS f2_name
        FROM fights f
        JOIN fighters f1 ON f1.fighter_id = f.fighter_1_id
        JOIN fighters f2 ON f2.fighter_id = f.fighter_2_id
        """
    ).fetchall()

    by_name_pair: dict[frozenset, list[sqlite3.Row]] = {}
    for row in our_fights:
        key = frozenset({normalize_fighter_name(row["f1_name"]), normalize_fighter_name(row["f2_name"])})
        by_name_pair.setdefault(key, []).append(row)

    today = date.today()
    now = _now()
    odds_rows = []
    matched_fights = set()

    for event in bfo_events:
        event_date = parse_event_date(event.get("event_date_raw"))
        last_change_utc = parse_last_change(event.get("last_change_raw"))
        for matchup in event.get("matchups", []):
            fighters = matchup.get("fighters", [])
            if len(fighters) != 2:
                continue
            key = frozenset(normalize_fighter_name(f["fighter_name"]) for f in fighters)
            candidates = by_name_pair.get(key, [])

            best = None
            if event_date:
                event_dt = datetime.fromisoformat(event_date).date()
                for row in candidates:
                    row_dt = datetime.fromisoformat(row["event_date"]).date()
                    if abs((row_dt - event_dt).days) <= date_tolerance_days:
                        best = row
                        break
            elif len(candidates) == 1:
                best = candidates[0]

            if best is None:
                continue

            name_to_fighter_id = {
                normalize_fighter_name(best["f1_name"]): best["fighter_1_id"],
                normalize_fighter_name(best["f2_name"]): best["fighter_2_id"],
            }
            odds_type = _derive_odds_type(best["event_date"], last_change_utc, today)
            matched_fights.add(best["fight_id"])

            for f in fighters:
                fighter_id = name_to_fighter_id.get(normalize_fighter_name(f["fighter_name"]))
                opponent_name = next(
                    (o["fighter_name"] for o in fighters if o is not f), None
                )
                for book, american in f.get("odds", {}).items():
                    odds_rows.append(
                        (
                            best["fight_id"],
                            fighter_id,
                            f["fighter_name"],
                            opponent_name,
                            book,
                            odds_type,
                            american,
                            _american_to_decimal(american),
                            event.get("event_date_raw"),
                            event.get("last_change_raw"),
                            last_change_utc,
                            now,
                            "bestfightodds",
                        )
                    )

    conn.executemany(
        """
        INSERT INTO odds (
            fight_id, fighter_id, fighter_name_raw, opponent_name_raw, sportsbook,
            odds_type, american_odds, decimal_odds, event_date_raw, last_change_raw,
            last_change_utc, captured_at, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        odds_rows,
    )
    conn.commit()
    return len(matched_fights)


def match_and_store_live_odds(conn: sqlite3.Connection, bfo_events: list[dict], date_tolerance_days: int = 3) -> int:
    """Same name+date matching as match_and_store_odds, but sourced from
    upcoming_fights instead of fights, and always stored with fight_id=NULL,
    odds_type='live'.

    fight_id is NULL deliberately, not a bug: odds.fight_id has a strict
    foreign key to fights(fight_id), and an upcoming bout isn't in that
    table (see upcoming_fights' docstring on why they're kept separate).
    market.market_probabilities_for_matchup already reads exactly this shape
    of row (`fight_id IS NULL AND fighter_id IN (?, ?) AND odds_type='live'`)
    -- this is the write side of a read path that already existed.

    Idempotent within its own scope: clears prior odds_type='live' rows
    before inserting (a live line changes constantly as a fight approaches,
    so "replace," not "append," is correct) -- deliberately does NOT touch
    odds_type='close'/'unverified' rows, which match_and_store_odds owns.
    """
    conn.execute("DELETE FROM odds WHERE source = 'bestfightodds' AND odds_type = 'live'")

    upcoming = conn.execute(
        """
        SELECT u.fight_id, u.event_date, u.fighter_1_id, u.fighter_2_id,
               f1.name AS f1_name, f2.name AS f2_name
        FROM upcoming_fights u
        JOIN fighters f1 ON f1.fighter_id = u.fighter_1_id
        JOIN fighters f2 ON f2.fighter_id = u.fighter_2_id
        """
    ).fetchall()

    by_name_pair: dict[frozenset, list[sqlite3.Row]] = {}
    for row in upcoming:
        key = frozenset({normalize_fighter_name(row["f1_name"]), normalize_fighter_name(row["f2_name"])})
        by_name_pair.setdefault(key, []).append(row)

    now = _now()
    odds_rows = []
    matched_fights = set()

    for event in bfo_events:
        event_date = parse_event_date(event.get("event_date_raw"))
        for matchup in event.get("matchups", []):
            fighters = matchup.get("fighters", [])
            if len(fighters) != 2:
                continue
            key = frozenset(normalize_fighter_name(f["fighter_name"]) for f in fighters)
            candidates = by_name_pair.get(key, [])

            best = None
            if event_date:
                event_dt = datetime.fromisoformat(event_date).date()
                for row in candidates:
                    row_dt = datetime.fromisoformat(row["event_date"]).date()
                    if abs((row_dt - event_dt).days) <= date_tolerance_days:
                        best = row
                        break
            elif len(candidates) == 1:
                best = candidates[0]

            if best is None:
                continue

            name_to_fighter_id = {
                normalize_fighter_name(best["f1_name"]): best["fighter_1_id"],
                normalize_fighter_name(best["f2_name"]): best["fighter_2_id"],
            }
            matched_fights.add(best["fight_id"])

            for f in fighters:
                fighter_id = name_to_fighter_id.get(normalize_fighter_name(f["fighter_name"]))
                opponent_name = next((o["fighter_name"] for o in fighters if o is not f), None)
                for book, american in f.get("odds", {}).items():
                    odds_rows.append(
                        (
                            None,  # fight_id: deliberately NULL, see docstring
                            fighter_id,
                            f["fighter_name"],
                            opponent_name,
                            book,
                            "live",
                            american,
                            _american_to_decimal(american),
                            event.get("event_date_raw"),
                            event.get("last_change_raw"),
                            parse_last_change(event.get("last_change_raw")),
                            now,
                            "bestfightodds",
                        )
                    )

    conn.executemany(
        """
        INSERT INTO odds (
            fight_id, fighter_id, fighter_name_raw, opponent_name_raw, sportsbook,
            odds_type, american_odds, decimal_odds, event_date_raw, last_change_raw,
            last_change_utc, captured_at, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        odds_rows,
    )
    conn.commit()
    return len(matched_fights)


def upsert_upcoming_card(conn: sqlite3.Connection, events: list[dict], fights: list[dict]) -> int:
    """Replaces upcoming_fights wholesale with the current snapshot. Unlike
    completed events/fights (which only ever grow), an upcoming card changes
    shape between refreshes -- fighters get pulled or swapped, fights get
    added -- so the right model is "this is the current truth," not
    "append what's new." Rows referencing a fighter_id not yet in the
    fighters table (a brand-new promotional debut) will fail their foreign
    key -- callers must upsert_fighters with that fighter's bio first (see
    fetcher.fetch_upcoming_card's fighter_ids / _fetch_missing_fighters).
    """
    conn.execute("DELETE FROM upcoming_fights")

    event_lookup = {e["event_id"]: e for e in events}
    rows = []
    skipped = []
    for f in fights:
        event = event_lookup.get(f["event_id"], {})
        event_date = parse_event_date(f.get("event_date_raw") or event.get("event_date_raw"))
        if event_date is None:
            skipped.append(f.get("fight_id"))
            continue
        rows.append(
            (
                f["fight_id"],
                f["event_id"],
                f.get("event_name") or event.get("name"),
                event_date,
                f.get("location") or event.get("location"),
                f.get("weight_class"),
                int(bool(f.get("title_fight"))),
                f["fighter_1_id"],
                f["fighter_2_id"],
                f.get("source_url"),
                _now(),
            )
        )
    if skipped:
        print(f"cleaner: skipped {len(skipped)} upcoming fights with unparseable dates: {skipped}")

    conn.executemany(
        """
        INSERT INTO upcoming_fights (
            fight_id, event_id, event_name, event_date, location,
            weight_class, title_fight, fighter_1_id, fighter_2_id, source_url, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def run(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    init_db(conn)

    fighters = load_raw("fighters")
    n_fighters = upsert_fighters(conn, fighters)

    events = load_raw("events_detail")
    n_events = upsert_events(conn, events)

    fights = load_raw("fights")
    n_fights, n_stats = upsert_fights(conn, fights)
    n_removed_from_upcoming = remove_completed_fights_from_upcoming(conn)

    print(
        f"cleaner: wrote {n_fighters} fighters, {n_events} events, "
        f"{n_fights} fights, {n_stats} fight_stats rows -> {db_path}"
    )
    if n_removed_from_upcoming:
        print(f"cleaner: removed {n_removed_from_upcoming} now-completed fight(s) from upcoming_fights")

    try:
        bfo_events = load_raw("odds_bestfightodds")
    except FileNotFoundError:
        bfo_events = None
    if bfo_events is not None:
        n_matched = match_and_store_odds(conn, bfo_events)
        print(f"cleaner: matched odds for {n_matched} fights from {len(bfo_events)} bestfightodds events")

    # Upcoming card must land before live-odds matching below -- that step
    # matches against the upcoming_fights table this just populated.
    try:
        upcoming_events = load_raw("upcoming_events")
        upcoming_fights = load_raw("upcoming_fights")
    except FileNotFoundError:
        upcoming_events = None
    if upcoming_events is not None:
        n_upcoming = upsert_upcoming_card(conn, upcoming_events, upcoming_fights)
        print(f"cleaner: wrote {n_upcoming} upcoming fights from {len(upcoming_events)} events")

    try:
        live_bfo_events = load_raw("odds_bestfightodds_live")
    except FileNotFoundError:
        live_bfo_events = None
    if live_bfo_events is not None:
        n_live_matched = match_and_store_live_odds(conn, live_bfo_events)
        print(f"cleaner: matched live odds for {n_live_matched} upcoming fights from {len(live_bfo_events)} bestfightodds events")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw ufcstats JSON into SQLite")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    run(db_path=args.db)


if __name__ == "__main__":
    main()
