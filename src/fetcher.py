"""Pulls raw fighter/fight/odds data from ufcstats.com and free odds sources.

ufcstats.com sits behind a lightweight JS proof-of-work gate (a same-origin
SHA-256 grinding challenge, not a CAPTCHA) before serving pages. UFCStatsClient
solves it once per session and reuses the resulting cookie. There is no public
API, so everything here is HTML scraping — keep request volume low and cached
(see RAW_DIR below) so re-runs don't re-hit the site unnecessarily.

This module only fetches and lightly parses HTML into plain dicts/lists. It
does not write to the database — that normalization step lives in cleaner.py,
so a fetch can always be re-parsed without a network round-trip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://ufcstats.com"  # site does not serve https (connection refused)
BFO_BASE_URL = "https://www.bestfightodds.com"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MIN_REQUEST_INTERVAL_SEC = 1.0  # be polite; no robots.txt exists on this host
BFO_MIN_REQUEST_INTERVAL_SEC = 2.0

_POW_NONCE_RE = re.compile(r'nonce\s*=\s*"([0-9a-f]+)"')
_POW_TARGET_RE = re.compile(r"new Array\((\d+)\+1\)")


class UFCStatsClient:
    """Thin HTTP client that transparently solves ufcstats.com's PoW gate."""

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL_SEC):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "ufc-edge-research-bot/0.1 (contact: research use only)"}
        )
        self.min_interval = min_interval
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _solve_pow(self, html: str) -> bool:
        nonce_match = _POW_NONCE_RE.search(html)
        target_match = _POW_TARGET_RE.search(html)
        if not (nonce_match and target_match):
            return False
        nonce = nonce_match.group(1)
        target = "0" * int(target_match.group(1))
        n = 0
        while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
            n += 1
        self.session.post(
            f"{BASE_URL}/__c",
            data=urlencode({"nonce": nonce, "n": n}),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return True

    def get(self, path: str, params: dict | None = None) -> str:
        """GET a path, solving the PoW challenge (at most once) if presented."""
        url = f"{BASE_URL}{path}"
        for attempt in range(2):
            self._throttle()
            resp = _get_with_retries(lambda: self.session.get(url, params=params, timeout=20))
            resp.raise_for_status()
            if "Checking your browser" in resp.text and attempt == 0:
                if self._solve_pow(resp.text):
                    continue
            return resp.text
        return resp.text


def _get_with_retries(request_fn, max_retries: int = 4, backoff_base: float = 2.0):
    """Retries a GET on transient connection errors/timeouts/5xx with
    exponential backoff. A multi-hundred-request crawl against ufcstats.com
    hits an occasional dropped connection or reset -- without a retry here,
    one blip kills the entire run (this happened in practice: a ~300-fighter
    fetch died on a single ConnectionResetError partway through). 4xx errors
    are not retried -- those are permanent (e.g. a bad fighter id), not
    transient.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = request_fn()
            if resp.status_code >= 500:
                resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base * (2**attempt))
    raise last_exc


def _clean_text(node) -> str:
    return node.get_text(strip=True) if node else ""


def _list_events(client: UFCStatsClient, path: str) -> list[dict]:
    """Shared row-parsing for both the completed and upcoming events listing
    pages -- identical markup (same table/row/link classes) on both, just a
    different path and time direction. Uses ?page=all so this is a single
    request rather than ~30 paginated ones for the completed listing (the
    upcoming listing is short enough that this is moot but harmless).
    """
    html = client.get(path, params={"page": "all"})
    soup = BeautifulSoup(html, "lxml")
    events = []
    # NOTE: the first row on this page sometimes uses a different row class
    # ("b-statistics__table-row_type_first") than the rest -- select on the
    # table body's <tr> generally and rely on the link/date presence check
    # below to skip the blank spacer row, rather than matching on row class.
    for row in soup.select("table.b-statistics__table-events tbody tr"):
        link = row.select_one("a.b-link")
        date_span = row.select_one("span.b-statistics__date")
        tds = row.select("td.b-statistics__table-col")
        if not link or not date_span or len(tds) < 2:
            continue
        event_id = link["href"].rstrip("/").rsplit("/", 1)[-1]
        events.append(
            {
                "event_id": event_id,
                "name": _clean_text(link),
                "event_date_raw": _clean_text(date_span),
                "location": _clean_text(tds[1]),
                "source_url": link["href"],
            }
        )
    return events


def list_completed_events(client: UFCStatsClient) -> list[dict]:
    """Return every completed event: {event_id, name, event_date, location}.

    The very first row on this listing is sometimes the next *upcoming*
    event (marked with a "next" icon) rather than a completed one — callers
    should still treat event_date as authoritative and let downstream
    fight-level completeness checks (see parse_event) be the real filter.
    """
    return _list_events(client, "/statistics/events/completed")


def list_upcoming_events(client: UFCStatsClient) -> list[dict]:
    """Return every scheduled-but-not-yet-fought event: {event_id, name,
    event_date, location}. Same markup as list_completed_events, different
    listing page. Every fight_id on these events will come back from
    parse_fight with result='scheduled' -- see fetch_upcoming_card, which is
    the actual entry point for pulling a fight card before it happens.
    """
    return _list_events(client, "/statistics/events/upcoming")


def parse_event(client: UFCStatsClient, event_id: str) -> dict:
    """Fetch one event page: metadata + the list of fight_ids on the card."""
    html = client.get(f"/event-details/{event_id}")
    soup = BeautifulSoup(html, "lxml")
    name = _clean_text(soup.select_one("span.b-content__title-highlight"))
    info_items = soup.select("div.b-list__info-box li.b-list__box-list-item")
    event_date_raw, location = None, None
    for li in info_items:
        label = _clean_text(li.select_one("i.b-list__box-item-title"))
        value = li.get_text(" ", strip=True).replace(label, "", 1).strip()
        if label.startswith("Date"):
            event_date_raw = value
        elif label.startswith("Location"):
            location = value

    fight_ids = []
    for row in soup.select("tbody.b-fight-details__table-body tr.b-fight-details__table-row"):
        link = row.get("data-link")
        if link:
            fight_ids.append(link.rstrip("/").rsplit("/", 1)[-1])

    return {
        "event_id": event_id,
        "name": name,
        "event_date_raw": event_date_raw,
        "location": location,
        "source_url": f"{BASE_URL}/event-details/{event_id}",
        "fight_ids": fight_ids,
    }


def _split_pair(p_tags) -> tuple[str, str]:
    vals = [_clean_text(p) for p in p_tags]
    while len(vals) < 2:
        vals.append("")
    return vals[0], vals[1]


def _parse_landed_attempted(text: str) -> tuple[int | None, int | None]:
    m = re.match(r"(\d+)\s+of\s+(\d+)", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _parse_ctrl_time(text: str) -> int | None:
    if not text or text == "--":
        return None
    m = re.match(r"(\d+):(\d+)", text)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_fight(client: UFCStatsClient, fight_id: str) -> dict:
    """Fetch one fight page: bout metadata + fighter totals + per-round stats.

    Returns result='scheduled' for fights that haven't happened yet (no W/L
    flag present) — cleaner.py must skip those rather than writing a fake
    result, since a scheduled fight has no outcome to train or backtest on.
    A scheduled fight still carries fighter_1_id/fighter_2_id/weight_class,
    though -- that's exactly the information an upcoming-card discovery
    pass needs (see fetch_upcoming_card), and there's no reason to throw it
    away just because there's no result to report alongside it yet.
    """
    html = client.get(f"/fight-details/{fight_id}")
    soup = BeautifulSoup(html, "lxml")

    persons = soup.select("div.b-fight-details__person")
    if len(persons) != 2:
        return {"fight_id": fight_id, "result": "unknown"}

    fighter_ids = []
    statuses = []
    for person in persons:
        link = person.select_one("a.b-fight-details__person-link")
        fighter_ids.append(link["href"].rstrip("/").rsplit("/", 1)[-1] if link else None)
        statuses.append(_clean_text(person.select_one("i.b-fight-details__person-status")))

    fight_title_raw = _clean_text(soup.select_one("i.b-fight-details__fight-title"))
    title_fight = "title" in fight_title_raw.lower()
    weight_class = re.sub(r"\s*(Title\s*)?Bout$", "", fight_title_raw, flags=re.I).strip()

    if statuses[0] == "W" and statuses[1] == "L":
        result, winner_id = "fighter_1", fighter_ids[0]
    elif statuses[0] == "L" and statuses[1] == "W":
        result, winner_id = "fighter_2", fighter_ids[1]
    elif statuses[0] == "D" and statuses[1] == "D":
        result, winner_id = "draw", None
    elif statuses[0] == "NC" and statuses[1] == "NC":
        result, winner_id = "nc", None
    else:
        # No decided result yet -> fight hasn't happened (scheduled/upcoming).
        return {
            "fight_id": fight_id,
            "result": "scheduled",
            "fighter_1_id": fighter_ids[0],
            "fighter_2_id": fighter_ids[1],
            "weight_class": weight_class,
            "title_fight": title_fight,
            "source_url": f"{BASE_URL}/fight-details/{fight_id}",
        }

    labels = {}
    for item in soup.select("i.b-fight-details__text-item, i.b-fight-details__text-item_first"):
        label = _clean_text(item.select_one("i.b-fight-details__label"))
        value = item.get_text(" ", strip=True).replace(label, "", 1).strip()
        if label:
            labels[label.rstrip(":")] = value

    method = labels.get("Method")
    end_round = int(labels["Round"]) if labels.get("Round", "").isdigit() else None
    time_str = labels.get("Time", "")
    tm = re.match(r"(\d+):(\d+)", time_str)
    end_time_sec = int(tm.group(1)) * 60 + int(tm.group(2)) if tm else None
    referee = labels.get("Referee")

    fmt_match = re.search(r"(\d+)\s*Rnd", labels.get("Time format", ""))
    scheduled_rounds = int(fmt_match.group(1)) if fmt_match else None

    fight = {
        "fight_id": fight_id,
        "result": result,
        "winner_id": winner_id,
        "fighter_1_id": fighter_ids[0],
        "fighter_2_id": fighter_ids[1],
        "weight_class": weight_class,
        "title_fight": title_fight,
        "scheduled_rounds": scheduled_rounds,
        "method": method,
        "method_detail": None,
        "end_round": end_round,
        "end_time_sec": end_time_sec,
        "referee": referee,
        "source_url": f"{BASE_URL}/fight-details/{fight_id}",
        "stats": {"total": None, "per_round": []},
    }

    totals_table = soup.select_one(
        "section.js-fight-section table"
    )  # first stats table = career "Totals" section
    if totals_table:
        row = totals_table.select_one("tbody tr")
        if row:
            cols = row.select("td")
            if len(cols) >= 10:
                kd = _split_pair(cols[1].select("p"))
                sig = _split_pair(cols[2].select("p"))
                total = _split_pair(cols[4].select("p"))
                td = _split_pair(cols[5].select("p"))
                sub = _split_pair(cols[7].select("p"))
                rev = _split_pair(cols[8].select("p"))
                ctrl = _split_pair(cols[9].select("p"))
                for i in range(2):
                    sig_l, sig_a = _parse_landed_attempted(sig[i])
                    tot_l, tot_a = _parse_landed_attempted(total[i])
                    td_l, td_a = _parse_landed_attempted(td[i])
                    fight["stats"][f"fighter_{i+1}_total"] = {
                        "round": 0,
                        "knockdowns": int(kd[i]) if kd[i].isdigit() else None,
                        "sig_str_landed": sig_l,
                        "sig_str_attempted": sig_a,
                        "total_str_landed": tot_l,
                        "total_str_attempted": tot_a,
                        "takedowns_landed": td_l,
                        "takedowns_attempted": td_a,
                        "sub_attempts": int(sub[i]) if sub[i].isdigit() else None,
                        "reversals": int(rev[i]) if rev[i].isdigit() else None,
                        "control_time_sec": _parse_ctrl_time(ctrl[i]),
                    }

    per_round_table = soup.select_one("table.js-fight-table")
    if per_round_table:
        # ufcstats renders one <tr> per round, each cell holding a pair of
        # <p> (one per fighter) -- same shape as the totals table above.
        for round_idx, row in enumerate(per_round_table.select("tbody tr"), start=1):
            cols = row.select("td")
            if len(cols) < 10:
                continue
            kd = _split_pair(cols[1].select("p"))
            sig = _split_pair(cols[2].select("p"))
            total = _split_pair(cols[4].select("p"))
            td = _split_pair(cols[5].select("p"))
            sub = _split_pair(cols[7].select("p"))
            rev = _split_pair(cols[8].select("p"))
            ctrl = _split_pair(cols[9].select("p"))
            for i in range(2):
                sig_l, sig_a = _parse_landed_attempted(sig[i])
                tot_l, tot_a = _parse_landed_attempted(total[i])
                td_l, td_a = _parse_landed_attempted(td[i])
                fight["stats"]["per_round"].append(
                    {
                        "fighter_idx": i + 1,
                        "round": round_idx,
                        "knockdowns": int(kd[i]) if kd[i].isdigit() else None,
                        "sig_str_landed": sig_l,
                        "sig_str_attempted": sig_a,
                        "total_str_landed": tot_l,
                        "total_str_attempted": tot_a,
                        "takedowns_landed": td_l,
                        "takedowns_attempted": td_a,
                        "sub_attempts": int(sub[i]) if sub[i].isdigit() else None,
                        "reversals": int(rev[i]) if rev[i].isdigit() else None,
                        "control_time_sec": _parse_ctrl_time(ctrl[i]),
                    }
                )

    return fight


_HEIGHT_RE = re.compile(r"(\d+)'\s*(\d+)")


def _parse_height_in(text: str) -> float | None:
    m = _HEIGHT_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def _parse_reach_in(text: str) -> float | None:
    m = re.search(r"(\d+)", text or "")
    return float(m.group(1)) if m else None


def parse_fighter(client: UFCStatsClient, fighter_id: str) -> dict:
    """Fetch bio fields only (height/reach/stance/DOB). Deliberately does NOT
    parse ufcstats' "Career statistics" box (SLpM, win %, etc.) -- those are
    aggregated over a fighter's whole career including fights that postdate
    any given historical bout, so they are a leakage trap. All per-fight
    numbers used for modeling must come from fight_stats via features.py's
    as-of-date rollups instead.
    """
    html = client.get(f"/fighter-details/{fighter_id}")
    soup = BeautifulSoup(html, "lxml")
    name = _clean_text(soup.select_one("span.b-content__title-highlight"))
    nickname = _clean_text(soup.select_one("p.b-content__Nickname"))

    fields = {}
    for li in soup.select("div.b-list__info-box_style_small-width li.b-list__box-list-item"):
        label = _clean_text(li.select_one("i.b-list__box-item-title"))
        value = li.get_text(" ", strip=True).replace(label, "", 1).strip()
        fields[label.rstrip(":").upper()] = value

    return {
        "fighter_id": fighter_id,
        "name": name,
        "nickname": nickname or None,
        "height_in": _parse_height_in(fields.get("HEIGHT")),
        "reach_in": _parse_reach_in(fields.get("REACH")),
        "stance": fields.get("STANCE") or None,
        "dob_raw": fields.get("DOB") or None,
        "source_url": f"{BASE_URL}/fighter-details/{fighter_id}",
    }


def list_fighter_ids(client: UFCStatsClient) -> list[str]:
    """Every fighter_id, by walking the A-Z index (each letter is paginated)."""
    ids: set[str] = set()
    for letter in "abcdefghijklmnopqrstuvwxyz":
        page = 1
        while True:
            html = client.get("/statistics/fighters", params={"char": letter, "page": page})
            soup = BeautifulSoup(html, "lxml")
            links = soup.select("table.b-statistics__table a[href*='/fighter-details/']")
            if not links:
                break
            for link in links:
                ids.add(link["href"].rstrip("/").rsplit("/", 1)[-1])
            has_next_page = any(
                a.get_text(strip=True) == str(page + 1)
                for a in soup.select("a.b-statistics__paginate-link")
            )
            if not has_next_page:
                break
            page += 1
    return sorted(ids)


class BestFightOddsClient:
    """Plain rate-limited HTTP client for bestfightodds.com.

    Unlike ufcstats.com this site has no PoW gate and an explicit
    "Allow: /" robots.txt, but it has no public API either -- this still
    scrapes HTML, just without the challenge-solving step.

    NOTE: the site also exposes an internal admin panel under /cnadm/* (a
    plain login form). That is not part of the public site and is never
    requested by this client -- only the public /events/<slug> and
    /fighters/<slug> paths are used.
    """

    def __init__(self, min_interval: float = BFO_MIN_REQUEST_INTERVAL_SEC):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (ufc-edge-research-bot/0.1)"})
        self.min_interval = min_interval
        self._last_request_at = 0.0

    def get(self, path: str) -> str:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()
        resp = _get_with_retries(lambda: self.session.get(f"{BFO_BASE_URL}{path}", timeout=20))
        resp.raise_for_status()
        return resp.text


def parse_bestfightodds_event(html: str) -> dict:
    """Parse one bestfightodds.com /events/<slug> page.

    Returns per-bookmaker American moneylines for every matchup on the card,
    plus the page's own "Last change" timestamp text (see last_change_raw
    below). Caveat -- this page only ever shows the *current* line for each
    book, not a true open/close pair: a real opening line requires either
    reverse-engineering the site's line-movement chart data endpoint or
    polling this page repeatedly over time and keeping the earliest snapshot
    ourselves; still not done here.

    last_change_raw is bestfightodds' own record of when any line on this
    event's board last moved (e.g. "Jun 28th 2026 13:58 UTC") -- one value
    per event/page, not per matchup (the site doesn't expose it more
    granularly). For a fight that has already happened, this is what lets
    cleaner._derive_odds_type *verify* the current line really is a frozen
    closing price rather than just assuming so because our own clock says
    the fight is in the past. Spot-checked against UFC 196 (fought March 5,
    2016): last_change reads "Mar 6th 2016 05:54 UTC" -- the night of the
    fight, exactly as expected, still unchanged a decade later.
    """
    soup = BeautifulSoup(html, "lxml")
    event_name = _clean_text(soup.select_one(".table-header h1"))

    # The visible ".table-header-date" span (e.g. "July 25th") omits the year.
    # The <meta name="description"> tag spells out "... on July 25, 2026." --
    # prefer that as the authoritative, year-qualified date.
    event_date_raw = None
    meta_desc = soup.select_one('meta[name="description"]')
    if meta_desc:
        m = re.search(r"on ([A-Za-z]+ \d{1,2}, \d{4})\.", meta_desc.get("content", ""))
        if m:
            event_date_raw = m.group(1)
    if not event_date_raw:
        event_date_raw = _clean_text(soup.select_one(".table-header-date"))

    last_change_raw = None
    last_change_span = soup.select_one("div.table-last-changed span[title]")
    if last_change_span:
        last_change_raw = last_change_span["title"]

    book_names: dict[str, str] = {}
    header_row = soup.select_one("table.odds-table:not(.odds-table-responsive-header) thead tr")
    if header_row:
        for th in header_row.select("th[data-b]"):
            name_el = th.select_one("a") or th.select_one("span")
            book_names[th["data-b"]] = _clean_text(name_el)

    # matchup_id -> corner -> {fighter_name, odds: {book_name: american_odds}}
    matchups: dict[str, dict] = {}
    scroller_table = soup.select_one("div.table-scroller table.odds-table")
    if scroller_table:
        for row in scroller_table.select("tbody tr"):
            name_el = row.select_one("span.t-b-fcc")
            if not name_el:
                continue
            fighter_name = _clean_text(name_el)
            for cell in row.select("td[data-li]"):
                data_li = json.loads(cell["data-li"])
                if len(data_li) != 3:
                    continue  # non-moneyline prop cell, e.g. [corner, matchup_id]
                book_id, corner, matchup_id = str(data_li[0]), data_li[1], str(data_li[2])
                odds_span = cell.select_one("span")
                if not odds_span:
                    continue
                odds_text = _clean_text(odds_span)
                if not re.match(r"^[+-]\d+$", odds_text):
                    continue
                bucket = matchups.setdefault(matchup_id, {})
                fighter_bucket = bucket.setdefault(corner, {"fighter_name": fighter_name, "odds": {}})
                fighter_bucket["odds"][book_names.get(book_id, book_id)] = int(odds_text)

    return {
        "event_name": event_name,
        "event_date_raw": event_date_raw,
        "last_change_raw": last_change_raw,
        "matchups": [
            {"matchup_id": mid, "fighters": list(corners.values())}
            for mid, corners in matchups.items()
        ],
    }


def fetch_bestfightodds_candidates(client: BestFightOddsClient, max_candidates: int = 40) -> list[dict]:
    """Fetch odds for the N most recent bestfightodds.com events (any promotion).

    /archive lists event slugs only, no dates -- so matching those slugs to our
    ufcstats events requires actually fetching each candidate page (the date
    only appears there). This is why the crawl is bounded by max_candidates
    rather than unbounded: cleaner.py's matcher works through this list against
    our known events and stops needing more once every event is matched, but
    the fetch itself always pulls the full bounded window up front.
    """
    archive_html = client.get("/archive")
    soup = BeautifulSoup(archive_html, "lxml")
    slugs = []
    for a in soup.select("a[href^='/events/']"):
        href = a["href"]
        if href not in slugs:
            slugs.append(href)
        if len(slugs) >= max_candidates:
            break

    candidates = []
    for slug in slugs:
        html = client.get(slug)
        event = parse_bestfightodds_event(html)
        event["slug"] = slug
        candidates.append(event)
    return candidates


def search_bestfightodds_fighter_url(client: BestFightOddsClient, fighter_name: str) -> str | None:
    """Searches bestfightodds.com for `fighter_name` and returns their
    /fighters/<slug> profile URL (the first one found), or None. This is how
    historical events beyond /archive's recent-only window get located --
    /archive has no pagination (confirmed by inspection: no
    page=/offset=/"Older" links exist on it at all), so it's structurally
    incapable of reaching back more than the last ~20-25 events across every
    promotion combined. A fighter's own profile page, by contrast, lists
    every event bestfightodds has them fighting in (see
    bestfightodds_fighter_event_urls) -- spot-checked against Rafael
    Fiziev's page, which correctly lists his June 27 2026 fight alongside
    his entire career back to 2019.
    """
    html = client.get(f"/search?query={quote(fighter_name)}")
    soup = BeautifulSoup(html, "lxml")
    link = soup.select_one("a[href^='/fighters/']")
    return link["href"] if link else None


_BFO_ROW_DATE_RE = re.compile(r"([A-Za-z]+) (\d{1,2})(?:st|nd|rd|th) (\d{4})")


def bestfightodds_fighter_event_rows(client: BestFightOddsClient, fighter_url: str) -> list[tuple[str, str | None]]:
    """Every (/events/<slug>, event_date_iso) pair on a fighter's
    bestfightodds profile page -- their whole fight history per
    bestfightodds' own records, each row carrying an inline date (e.g. "UFC
    Fight Night Jun 27th 2026") that lets callers filter to a target window
    WITHOUT fetching every single event page first. That matters because a
    long-career fighter's profile lists 10+ years of events; fetching all of
    them to backfill just the last two years would be a lot of wasted
    requests to a site with no public API.
    """
    html = client.get(fighter_url)
    soup = BeautifulSoup(html, "lxml")
    rows = []
    seen = set()
    for tr in soup.select("tr.event-header"):
        a = tr.select_one("a[href^='/events/']")
        if not a or a["href"] in seen:
            continue
        seen.add(a["href"])
        m = _BFO_ROW_DATE_RE.search(tr.get_text(" ", strip=True))
        event_date_iso = None
        if m:
            month, day, year = m.groups()
            try:
                event_date_iso = datetime.strptime(f"{month} {day} {year}", "%b %d %Y").date().isoformat()
            except ValueError:
                pass
        rows.append((a["href"], event_date_iso))
    return rows


def fetch_bestfightodds_for_fighters(
    client: BestFightOddsClient, fighter_names: list[str], since_date: str | None = None
) -> list[dict]:
    """For each fighter name, finds their bestfightodds profile and fetches
    every event listed on it on or after `since_date` (ISO 'YYYY-MM-DD'; all
    events if None). One popular fighter's profile can surface many
    relevant events at once (a full career's worth), so this is far more
    request-efficient than searching per-event or per-fight -- callers
    typically pass one or two fighters per historical event they care about
    (any fighter from that card is enough) rather than every fighter on
    every card, since a handful of that card's other fighters' profiles
    will likely surface the same event again, which the seen-URL dedup
    below catches for free. cleaner.match_and_store_odds's own name+date
    matching handles the rest (every fight on a fetched event page, not
    just the fighter that led us there, and any wrong/irrelevant event
    pulled in gets naturally rejected there too). An event row with no
    parseable inline date is fetched regardless of `since_date` -- better to
    over-fetch a handful of ambiguous rows than silently skip real data.
    """
    seen_fighter_urls: set[str] = set()
    seen_event_urls: set[str] = set()
    events = []
    for name in fighter_names:
        fighter_url = search_bestfightodds_fighter_url(client, name)
        if not fighter_url or fighter_url in seen_fighter_urls:
            continue
        seen_fighter_urls.add(fighter_url)
        for event_url, event_date_iso in bestfightodds_fighter_event_rows(client, fighter_url):
            if event_url in seen_event_urls:
                continue
            if since_date and event_date_iso and event_date_iso < since_date:
                continue
            seen_event_urls.add(event_url)
            html = client.get(event_url)
            event = parse_bestfightodds_event(html)
            event["slug"] = event_url
            events.append(event)
    return events


def dump_raw(name: str, payload) -> Path:
    """Persist a raw fetch (list of dicts / dict) as JSON under data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _load_existing(name: str) -> list:
    path = RAW_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else []


def fetch_upcoming_card(client: UFCStatsClient, max_events: int | None = None) -> dict:
    """One-shot pull of every currently scheduled (not-yet-fought) event and
    its fight card: fighter pairs, weight class, date.

    Returns a snapshot, not something to merge incrementally with a prior
    one the way bootstrap() does for completed history -- upcoming cards
    change shape (fighters get pulled/swapped, fights added), so the
    caller (cleaner.py) is expected to replace its stored upcoming_fights
    table wholesale on each refresh rather than append to it.

    Also collects the set of fighter_ids appearing on these cards so the
    caller can fetch bios for any genuinely new fighters (a promotional
    newcomer with no prior UFC fight) -- reuses parse_fighter, same as the
    historical bootstrap.
    """
    events = list_upcoming_events(client)
    if max_events is not None:
        events = events[:max_events]

    full_events, scheduled_fights, fighter_ids = [], [], set()
    for ev in events:
        detail = parse_event(client, ev["event_id"])
        detail.update({k: v for k, v in ev.items() if k not in detail or not detail[k]})
        full_events.append(detail)
        for fight_id in detail["fight_ids"]:
            fight = parse_fight(client, fight_id)
            if fight.get("result") != "scheduled":
                continue  # already happened, or a malformed page -- not part of the upcoming snapshot
            fight["event_id"] = ev["event_id"]
            fight["event_name"] = detail.get("name")
            fight["event_date_raw"] = detail.get("event_date_raw")
            fight["location"] = detail.get("location")
            scheduled_fights.append(fight)
            fighter_ids.add(fight["fighter_1_id"])
            fighter_ids.add(fight["fighter_2_id"])

    return {"events": full_events, "fights": scheduled_fights, "fighter_ids": sorted(fighter_ids)}


def _fetch_missing_fighters(client: UFCStatsClient, fighter_ids: set[str], checkpoint_every: int = 25) -> list[dict]:
    """Loads data/raw/fighters.json, fetches bios for any of `fighter_ids`
    not already present, and checkpoints to disk periodically. Shared by
    bootstrap() (historical backfill) and the upcoming-card fetch path (a
    promotional newcomer with no prior fight needs a bio too) so there is
    one incremental/resumable fighter-fetch implementation, not two.
    """
    fighters = _load_existing("fighters")
    fetched_fighter_ids = {f["fighter_id"] for f in fighters}
    remaining_fighter_ids = sorted(fighter_ids - fetched_fighter_ids)
    print(f"fetcher: {len(fetched_fighter_ids)} fighters already fetched, {len(remaining_fighter_ids)} remaining")

    for i, fid in enumerate(remaining_fighter_ids, start=1):
        fighters.append(parse_fighter(client, fid))
        if i % checkpoint_every == 0 or i == len(remaining_fighter_ids):
            dump_raw("fighters", fighters)
            print(f"fetcher: checkpointed {i}/{len(remaining_fighter_ids)} new fighters ({len(fighters)} total)")

    return fighters


def bootstrap(
    max_events: int | None = None,
    with_odds: bool = False,
    max_odds_candidates: int = 40,
    event_checkpoint_every: int = 10,
    fighter_checkpoint_every: int = 25,
) -> None:
    """Incremental pull: completed events -> their fights -> the fighters in
    them. Writes raw JSON snapshots to data/raw/ for cleaner.py to normalize
    into SQLite.

    Resumable and incremental by default -- any events/fighters already
    present in data/raw/*.json from a prior run are skipped, and progress is
    checkpointed to disk every `event_checkpoint_every` events /
    `fighter_checkpoint_every` fighters rather than only once at the very
    end. This matters for two reasons: (1) a full historical backfill is
    thousands of requests at a polite 1 req/sec, long enough that a
    transient failure partway through is a real risk we've already hit in
    practice, not a hypothetical; losing only the last few minutes of
    progress instead of the whole run matters. (2) re-running this same
    command after new UFC events happen naturally only fetches what's new --
    which is the incremental-refresh behavior the project always wanted,
    without needing a separate code path for it.
    """
    client = UFCStatsClient()
    events = list_completed_events(client)
    if max_events is not None:
        events = events[:max_events]
    dump_raw("events_index", events)

    full_events = _load_existing("events_detail")
    all_fights = _load_existing("fights")
    done_event_ids = {e["event_id"] for e in full_events}
    fighter_ids = {f["fighter_1_id"] for f in all_fights} | {f["fighter_2_id"] for f in all_fights}

    remaining_events = [e for e in events if e["event_id"] not in done_event_ids]
    print(f"fetcher: {len(done_event_ids)} events already fetched, {len(remaining_events)} remaining")

    for i, ev in enumerate(remaining_events, start=1):
        detail = parse_event(client, ev["event_id"])
        detail.update({k: v for k, v in ev.items() if k not in detail or not detail[k]})
        full_events.append(detail)
        for fight_id in detail["fight_ids"]:
            fight = parse_fight(client, fight_id)
            if fight.get("result") in (None, "scheduled", "unknown"):
                continue
            fight["event_id"] = ev["event_id"]
            all_fights.append(fight)
            fighter_ids.add(fight["fighter_1_id"])
            fighter_ids.add(fight["fighter_2_id"])

        if i % event_checkpoint_every == 0 or i == len(remaining_events):
            dump_raw("events_detail", full_events)
            dump_raw("fights", all_fights)
            print(
                f"fetcher: checkpointed {i}/{len(remaining_events)} new events "
                f"({len(full_events)} total events, {len(all_fights)} total fights)"
            )

    fighters = _fetch_missing_fighters(client, fighter_ids, checkpoint_every=fighter_checkpoint_every)

    print(
        f"Fetched {len(full_events)} events, {len(all_fights)} completed fights, "
        f"{len(fighters)} fighters -> {RAW_DIR}"
    )

    if with_odds:
        odds_client = BestFightOddsClient()
        odds_candidates = fetch_bestfightodds_candidates(odds_client, max_candidates=max_odds_candidates)
        dump_raw("odds_bestfightodds", odds_candidates)
        print(f"Fetched {len(odds_candidates)} bestfightodds.com candidate events -> {RAW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch UFC fight/fighter data from ufcstats.com")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limit to the N most recent completed events (useful for a quick smoke test)",
    )
    parser.add_argument(
        "--with-odds",
        action="store_true",
        help="Also crawl a bounded window of recent bestfightodds.com events for later matching",
    )
    parser.add_argument(
        "--max-odds-candidates",
        type=int,
        default=40,
        help="How many recent bestfightodds.com events to fetch when --with-odds is set",
    )
    parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Fetch the current upcoming-card snapshot instead of historical completed events",
    )
    parser.add_argument(
        "--with-live-odds",
        action="store_true",
        help="With --upcoming, also fetch current bestfightodds.com lines for the card's fighters",
    )
    args = parser.parse_args()

    if args.upcoming:
        client = UFCStatsClient()
        card = fetch_upcoming_card(client, max_events=args.max_events)
        dump_raw("upcoming_events", card["events"])
        dump_raw("upcoming_fights", card["fights"])
        fighters = _fetch_missing_fighters(client, set(card["fighter_ids"]))
        print(
            f"Fetched {len(card['events'])} upcoming events, {len(card['fights'])} scheduled fights, "
            f"{len(fighters)} fighters total -> {RAW_DIR}"
        )

        if args.with_live_odds:
            fighter_by_id = {f["fighter_id"]: f["name"] for f in fighters}
            fighter_names = [fighter_by_id[fid] for fid in card["fighter_ids"] if fid in fighter_by_id]
            since_date = datetime.now().date().isoformat()
            odds_client = BestFightOddsClient()
            live_events = fetch_bestfightodds_for_fighters(odds_client, fighter_names, since_date=since_date)
            dump_raw("odds_bestfightodds_live", live_events)
            print(f"Fetched live odds from {len(live_events)} bestfightodds events -> {RAW_DIR}")
        return

    bootstrap(
        max_events=args.max_events,
        with_odds=args.with_odds,
        max_odds_candidates=args.max_odds_candidates,
    )


if __name__ == "__main__":
    main()
