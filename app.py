"""UFC Edge dashboard (V1, informational) -- Streamlit, local/personal use only.

Shows upcoming UFC cards: model win probability, de-vigged live market
probability, the gap between them, a confidence label, and the underlying
per-fighter stats (rolling form, striking/grappling rates, physical diffs)
behind each prediction. A second tab shows the last 6 COMPLETED cards with
the same model/strategy signals recomputed against the fight's own verified
closing line, plus the actual result -- so past picks can be reviewed for
correctness, not just upcoming ones.

Deliberately NOT a betting tool: edge is displayed as a "does the model
disagree with the market, and by how much" data point, with an explicit
disclaimer that larger gaps have been shown (via a real, bootstrap-tested
walk-forward backtest -- see README's Walk-forward backtest section) to be
a net-negative signal historically, not a positive one. Validating a
betting-usable edge is future work, not what this view claims to do.

Run with: streamlit run app.py
"""
from __future__ import annotations

import subprocess
import sys
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st

from src import features, market, prediction_log, report, strategy

DB_PATH = "db/ufc.db"
PAST_CARDS_LIMIT = 6

st.set_page_config(page_title="UFC Edge", layout="wide")


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource
def get_model(name: str):
    """Cached per model name -- loads once per Streamlit process, not once
    per page interaction. Uses the deliberately pinned production artifact
    (report.load_production_model), not just "whatever was trained most
    recently" -- see report.py's docstring on why that distinction matters
    for anything running unattended.
    """
    fitted_model, model_path = report.load_production_model(name)
    calibration_table = report.load_calibration_table(model_path)
    feature_columns = report.load_feature_columns(model_path)
    return fitted_model, model_path, calibration_table, feature_columns


def _fmt_pct(x: float | None) -> str:
    # pd.DataFrame(rows) turns a dict's None into NaN for a float column --
    # `x is None` alone missed that and rendered a literal "nan%" in the UI.
    return "-" if pd.isna(x) else f"{x * 100:.1f}%"


def _fmt_edge(x: float | None) -> str:
    return "-" if pd.isna(x) else f"{x * 100:+.1f}%"


def _freshness(conn: sqlite3.Connection, fighter_1_id: str, fighter_2_id: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(captured_at) AS captured_at FROM odds "
        "WHERE odds_type = 'live' AND fight_id IS NULL AND fighter_id IN (?, ?)",
        (fighter_1_id, fighter_2_id),
    ).fetchone()
    if not row or not row["captured_at"]:
        return None
    captured = datetime.fromisoformat(row["captured_at"])
    age = datetime.now(captured.tzinfo) - captured
    hours = age.total_seconds() / 3600
    if hours < 1:
        return f"odds captured {int(age.total_seconds() / 60)} min ago"
    return f"odds captured {hours:.1f} hr ago"


def add_strategy_signals(
    card_report: pd.DataFrame, show_refined: bool, show_refined_plus: bool, show_elo: bool
) -> pd.DataFrame:
    """Adds refined_signal/refined_plus_signal/elo_signal + a human-readable
    "Strategy Signal" column to a card_report DataFrame that already has
    logistic_prob/gbm_prob/f1_n_prior_fights/f2_n_prior_fights/title_fight/
    weight_class/fighter_1_decimal_odds/fighter_2_decimal_odds/
    fighter_1_stance/fighter_2_stance columns. Shared between the Upcoming
    and Past Cards tabs so the two views can never silently diverge in how
    a signal is computed.
    """
    card_report = card_report.copy()
    card_report["refined_signal"] = [
        strategy.bet_signal(lp, gp, f1_n_prior=f1n, f2_n_prior=f2n, title_fight=tf, **strategy.REFINED_RULE)
        if show_refined else None
        for lp, gp, f1n, f2n, tf in zip(
            card_report["logistic_prob"], card_report["gbm_prob"],
            card_report["f1_n_prior_fights"], card_report["f2_n_prior_fights"], card_report["title_fight"],
        )
    ]
    card_report["refined_plus_signal"] = [
        strategy.bet_signal(
            lp, gp, f1_n_prior=f1n, f2_n_prior=f2n, title_fight=tf,
            weight_class=wc, fighter_1_decimal_odds=f1o, fighter_2_decimal_odds=f2o,
            **strategy.REFINED_PLUS_RULE,
        ) if show_refined_plus else None
        for lp, gp, f1n, f2n, tf, wc, f1o, f2o in zip(
            card_report["logistic_prob"], card_report["gbm_prob"],
            card_report["f1_n_prior_fights"], card_report["f2_n_prior_fights"], card_report["title_fight"],
            card_report["weight_class"], card_report["fighter_1_decimal_odds"], card_report["fighter_2_decimal_odds"],
        )
    ]
    card_report["elo_signal"] = [
        strategy.bet_signal(
            lp, gp, f1_n_prior=f1n, f2_n_prior=f2n, title_fight=tf,
            weight_class=wc, fighter_1_decimal_odds=f1o, fighter_2_decimal_odds=f2o,
            fighter_1_stance=f1s, fighter_2_stance=f2s,
            **strategy.ELO_RULE,
        ) if show_elo else None
        for lp, gp, f1n, f2n, tf, wc, f1o, f2o, f1s, f2s in zip(
            card_report["logistic_prob"], card_report["gbm_prob"],
            card_report["f1_n_prior_fights"], card_report["f2_n_prior_fights"], card_report["title_fight"],
            card_report["weight_class"], card_report["fighter_1_decimal_odds"], card_report["fighter_2_decimal_odds"],
            card_report["fighter_1_stance"], card_report["fighter_2_stance"],
        )
    ]
    card_report["Strategy Signal"] = card_report.apply(_signal_label, axis=1)
    return card_report


def _picked_side(row) -> str | None:
    return row["refined_signal"] or row["refined_plus_signal"] or row["elo_signal"]


def _signal_label(row) -> str:
    fighter = _picked_side(row)
    if fighter is None:
        return ""
    fighter_name = row["fighter_1"] if fighter == "fighter_1" else row["fighter_2"]
    active = [
        name for name, sig in
        [("Refined", row["refined_signal"]), ("Refined+", row["refined_plus_signal"]), ("Elo", row["elo_signal"])]
        if sig is not None
    ]
    icon = "\U0001F525" if len(active) >= 3 else "⭐" if len(active) == 2 else "✅"
    return f"{icon} {' + '.join(active)} -> {fighter_name}"


def _highlight_row(row):
    label = row["Strategy Signal"]
    if label.startswith("\U0001F525"):
        color = "background-color: rgba(255, 99, 71, 0.35)"  # all 3 selected rules agree -- strongest highlight
    elif label.startswith("⭐"):
        color = "background-color: rgba(255, 215, 0, 0.30)"  # 2 of the selected rules agree
    elif label.startswith("✅"):
        color = "background-color: rgba(60, 179, 113, 0.25)"  # 1 rule only
    else:
        color = ""
    return [color] * len(row)


conn = get_connection()

st.title("UFC Edge — Card Dashboard")
st.warning(
    "**Informational only, not a betting signal.** A rigorous walk-forward backtest "
    "(bootstrap-tested, not just a raw ROI number) found that fights where this model "
    "disagrees with the market by a large margin have historically been a "
    "*statistically significant loss*, not a value opportunity. Treat the edge column "
    "below as \"where the model and market disagree,\" not as investment advice.",
    icon="⚠️",
)

model_name = st.sidebar.radio("Model (used for the Confidence label)", ["logistic", "gbm"], index=0)
n_window = st.sidebar.slider("Rolling form window (last N fights)", min_value=3, max_value=10, value=5)
logistic_model, logistic_path, logistic_calib, logistic_features = get_model("logistic")
gbm_model, gbm_path, gbm_calib, gbm_features = get_model("gbm")
st.sidebar.caption(f"Logistic artifact: `{logistic_path.name}`  \nGBM artifact: `{gbm_path.name}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Betting strategy highlight")
show_refined = st.sidebar.checkbox("Highlight Refined rule", value=True)
show_refined_plus = st.sidebar.checkbox("Highlight Refined+ rule", value=True)
show_elo = st.sidebar.checkbox("Highlight Elo rule", value=True)
st.sidebar.caption(
    "All rules need the average of the logistic + GBM probabilities past a "
    "threshold, AND neither model individually below/above a confirm floor, AND "
    "both fighters need >=3 tracked fights, with title fights excluded. Both models "
    "now also see a career-long Elo rating (100% coverage, opponent-strength- and "
    "finish-weighted) as an input feature, not just their last-5-fight rolling stats. "
    "**Refined** (avg>62%/<38%, confirm>=55%/<=45%): backtested on 351 bets, "
    "78.3% hit rate, ROI +8.7%, 95% CI [+2.6%,+14.9%] -- statistically significant. "
    "**Refined+** (adds: excludes Heavyweight/Light Heavyweight, and only bets when "
    "the picked fighter's decimal odds are below 2.0): backtested on 297 bets, 80.8% "
    "hit rate, ROI +11.4%, 95% CI [+4.7%,+18.0%]. "
    "**Elo** (adds: excludes a Southpaw-vs-Orthodox matchup on either side): "
    "backtested on 219 bets, 83.1% hit rate, ROI +14.5%, 95% CI [+7.2%,+21.5%] -- "
    "the strongest and most robust result found so far, holding up across every "
    "fold-size robustness check tested. "
    "Original/Tighter (no extra filters) were retired -- both dropped out of "
    "statistical significance as the model/data evolved. "
    "See README's Walk-forward backtest section for the full methodology."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "No scheduled auto-refresh -- click to pull the latest upcoming card and live odds "
    "on demand. Rate-limited scraping, typically takes a few minutes."
)
if st.sidebar.button("🔄 Refresh upcoming card + live odds"):
    with st.spinner("Fetching upcoming card and live odds... this can take several minutes"):
        try:
            # A hard ceiling, not just relying on the network layer's own
            # request timeouts -- one of those (the PoW-solution POST in
            # fetcher._solve_pow) previously had no timeout at all and once
            # hung this button for 12+ minutes with no way to recover short
            # of killing the process manually. This is the last line of
            # defense against the same class of bug recurring anywhere else
            # in the fetch/clean chain, not a replacement for fixing timeouts
            # at the source.
            result = subprocess.run(
                [sys.executable, "-m", "src.refresh", "--mode", "upcoming"],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            result = None
            st.sidebar.error(
                "Refresh timed out after 10 minutes -- likely a stalled request to "
                "ufcstats.com or bestfightodds.com. Safe to try again."
            )
    if result is not None:
        if result.returncode == 0:
            st.sidebar.success("Refreshed.")
            st.rerun()
        else:
            st.sidebar.error(f"Refresh failed:\n{result.stderr[-1000:]}")

tab_upcoming, tab_past = st.tabs(["📅 Upcoming Card", f"📜 Past Cards (last {PAST_CARDS_LIMIT})"])

with tab_upcoming:
    events_df = pd.read_sql_query(
        "SELECT DISTINCT event_id, event_name, event_date, location FROM upcoming_fights ORDER BY event_date",
        conn,
    )

    if events_df.empty:
        st.info(
            "No upcoming card data yet. Run `python3 -m src.fetcher --upcoming --with-live-odds` "
            "then `python3 -m src.cleaner` to populate one."
        )
        st.stop()

    event_labels = [f"{r.event_name} — {r.event_date} ({r.location})" for r in events_df.itertuples()]
    selected = st.selectbox("Upcoming card", event_labels)
    selected_event = events_df.iloc[event_labels.index(selected)]

    fights_df = pd.read_sql_query(
        """
        SELECT u.fight_id, u.fighter_1_id, u.fighter_2_id, u.weight_class, u.title_fight, u.event_date,
               f1.name AS fighter_1_name, f2.name AS fighter_2_name
        FROM upcoming_fights u
        JOIN fighters f1 ON f1.fighter_id = u.fighter_1_id
        JOIN fighters f2 ON f2.fighter_id = u.fighter_2_id
        WHERE u.event_id = ?
        """,
        conn,
        params=(selected_event["event_id"],),
    )

    matchups = list(zip(fights_df["fighter_1_id"], fights_df["fighter_2_id"]))
    as_of = selected_event["event_date"] if selected_event["event_date"] <= date.today().isoformat() else date.today().isoformat()

    # Both models are always computed -- the betting-strategy signals need both
    # probabilities together (average + individual confirm floors), regardless
    # of which single model's Confidence label the sidebar radio selects.
    logistic_report = report.build_card_report(
        conn, logistic_model, matchups, as_of_date=as_of, odds_type="live", n=n_window,
        calibration_table=logistic_calib, feature_columns=logistic_features,
    )
    gbm_report = report.build_card_report(
        conn, gbm_model, matchups, as_of_date=as_of, odds_type="live", n=n_window,
        calibration_table=gbm_calib, feature_columns=gbm_features,
    )

    card_report = logistic_report.copy()
    card_report["logistic_prob"] = logistic_report["model_prob_fighter_1"]
    card_report["gbm_prob"] = gbm_report["model_prob_fighter_1"]
    card_report["avg_prob"] = (card_report["logistic_prob"] + card_report["gbm_prob"]) / 2
    card_report["edge_fighter_1"] = card_report["avg_prob"] - card_report["market_prob_fighter_1"]
    card_report["confidence"] = logistic_report["confidence"] if model_name == "logistic" else gbm_report["confidence"]
    card_report["weight_class"] = fights_df["weight_class"]
    card_report["title_fight"] = fights_df["title_fight"].astype(bool)

    card_report = add_strategy_signals(card_report, show_refined, show_refined_plus, show_elo)

    try:
        prediction_log.log_card(conn, card_report, fights_df, selected_event)
    except Exception as exc:
        # A logging failure must never break the dashboard itself -- surfaced
        # quietly rather than silently, so a real problem (e.g. a schema drift)
        # doesn't go unnoticed indefinitely, but it's not fatal to the page.
        st.sidebar.caption(f"⚠️ Prediction logging failed: {exc}")

    display_df = card_report.copy()
    display_df["Logistic %"] = display_df["logistic_prob"].map(_fmt_pct)
    display_df["GBM %"] = display_df["gbm_prob"].map(_fmt_pct)
    display_df["Avg %"] = display_df["avg_prob"].map(_fmt_pct)
    display_df["Market %"] = display_df["market_prob_fighter_1"].map(_fmt_pct)
    display_df["Edge"] = display_df["edge_fighter_1"].map(_fmt_edge)
    display_df = display_df.rename(
        columns={"fighter_1": "Fighter 1", "fighter_2": "Fighter 2", "weight_class": "Weight class", "confidence": "Confidence"}
    )

    table_cols = ["Fighter 1", "Fighter 2", "Weight class", "Logistic %", "GBM %", "Avg %", "Market %", "Edge", "Confidence", "Strategy Signal"]

    st.dataframe(
        display_df[table_cols].style.apply(_highlight_row, axis=1),
        width="stretch",
        hide_index=True,
        height=min(35 * (len(display_df) + 1) + 3, 740),
        column_config={"Strategy Signal": st.column_config.TextColumn(width="medium")},
    )
    if show_refined or show_refined_plus or show_elo:
        st.caption(
            "Highlighted rows are the fights the selected strategy/strategies flag as bettable -- "
            "everything else gets no bet under any rule. \U0001F525 = all 3 selected rules agree, "
            "⭐ = 2 of the selected rules agree, ✅ = 1 rule only. The label spells out exactly which "
            "rule(s) fired, e.g. \"Refined + Elo\"."
        )

    st.subheader("Fight detail")
    fight_choice = st.selectbox(
        "Select a fight for the underlying stat breakdown",
        range(len(fights_df)),
        format_func=lambda i: f"{fights_df.iloc[i]['fighter_1_name']} vs. {fights_df.iloc[i]['fighter_2_name']}",
    )
    row = fights_df.iloc[fight_choice]
    report_row = card_report.iloc[fight_choice]

    status = report_row["Strategy Signal"] or "No signal from any selected rule -- not bettable under any strategy."
    st.markdown(
        f"**Logistic:** {_fmt_pct(report_row['logistic_prob'])} &nbsp;|&nbsp; "
        f"**GBM:** {_fmt_pct(report_row['gbm_prob'])} &nbsp;|&nbsp; "
        f"**Avg:** {_fmt_pct(report_row['avg_prob'])} &nbsp;|&nbsp; "
        f"**Market:** {_fmt_pct(report_row['market_prob_fighter_1'])} &nbsp;|&nbsp; "
        f"**Strategy:** {status}"
    )

    f1_roll = features.fighter_rolling_features(conn, row["fighter_1_id"], as_of, n=n_window)
    f2_roll = features.fighter_rolling_features(conn, row["fighter_2_id"], as_of, n=n_window)
    f1_phys = features.fighter_physical_features(conn, row["fighter_1_id"], as_of)
    f2_phys = features.fighter_physical_features(conn, row["fighter_2_id"], as_of)
    f1_career = features.fighter_career_features(conn, row["fighter_1_id"], as_of)
    f2_career = features.fighter_career_features(conn, row["fighter_2_id"], as_of)

    def _pct(x):
        return "-" if x is None else f"{x * 100:.0f}%"

    def _num(x, fmt="{:.0f}"):
        return "-" if x is None else fmt.format(x)

    def _raw(x):
        # Every cell in comparison_df must be a string -- mixing raw ints/floats/None
        # with _pct()'s "60%" strings in the same column made pandas produce an
        # object-dtype column pyarrow couldn't serialize (ArrowInvalid: "Could not
        # convert '60%' ... to int64"). Streamlit silently recovers by coercing
        # columns itself, but the underlying crash is real -- this avoids it.
        return "-" if x is None else str(x)

    comparison_rows = [
        ("Fights tracked (last N)", _raw(f1_roll["n_prior_fights"]), _raw(f2_roll["n_prior_fights"])),
        ("Career fights (all-time)", _raw(f1_career["total_prior_fights"]), _raw(f2_career["total_prior_fights"])),
        ("Win % (last N)", _pct(f1_roll["win_pct"]), _pct(f2_roll["win_pct"])),
        ("Current streak", _raw(f1_roll["current_streak"]), _raw(f2_roll["current_streak"])),
        ("Finish rate (career wins by KO/sub)", _pct(f1_career["finish_rate"]), _pct(f2_career["finish_rate"])),
        ("Times finished (career losses by KO/sub)", _pct(f1_career["times_finished_rate"]), _pct(f2_career["times_finished_rate"])),
        ("Sig. strike accuracy", _pct(f1_roll["sig_str_acc"]), _pct(f2_roll["sig_str_acc"])),
        ("Sig. strike defense", _pct(f1_roll["sig_str_def"]), _pct(f2_roll["sig_str_def"])),
        ("Takedown accuracy", _pct(f1_roll["td_acc"]), _pct(f2_roll["td_acc"])),
        ("Takedown defense", _pct(f1_roll["td_def"]), _pct(f2_roll["td_def"])),
        ("Days since last fight", _raw(f1_roll["days_since_last_fight"]), _raw(f2_roll["days_since_last_fight"])),
        ("Height (in)", _raw(f1_phys["height_in"]), _raw(f2_phys["height_in"])),
        ("Reach (in)", _raw(f1_phys["reach_in"]), _raw(f2_phys["reach_in"])),
        ("Stance", f1_phys["stance"] or "-", f2_phys["stance"] or "-"),
        ("Age (as of fight)", _num(f1_phys["age_years"], "{:.1f}"), _num(f2_phys["age_years"], "{:.1f}")),
    ]
    comparison_df = pd.DataFrame(comparison_rows, columns=["Stat", row["fighter_1_name"], row["fighter_2_name"]])
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.caption(
        "Numbers only, no highlighting of \"better\"/\"worse\" -- several of these "
        "(e.g. days since last fight, times finished) don't have a universally correct "
        "direction, and this view is meant to inform, not steer."
    )

    freshness = _freshness(conn, row["fighter_1_id"], row["fighter_2_id"])
    st.caption(f"Edge = model − market. {freshness or 'no live odds captured for this fight yet'}.")

with tab_past:
    st.caption(
        f"The last {PAST_CARDS_LIMIT} COMPLETED events, most recent first. Rolls automatically -- "
        "as a new card finishes and gets pulled in via `--mode full`, the oldest of these drops off. "
        "Model/market probabilities here are recomputed against each fight's own verified CLOSING "
        "line (not a live line, since the fight already happened), and the actual result is shown "
        "so a pick can be checked for correctness."
    )

    past_events_df = pd.read_sql_query(
        """
        SELECT e.event_id, e.name AS event_name, e.event_date, e.location
        FROM events e
        WHERE e.event_id IN (SELECT DISTINCT event_id FROM fights)
        ORDER BY e.event_date DESC
        LIMIT ?
        """,
        conn,
        params=(PAST_CARDS_LIMIT,),
    )

    if past_events_df.empty:
        st.info("No completed events in the database yet.")
    else:
        past_labels = [f"{r.event_name} — {r.event_date} ({r.location})" for r in past_events_df.itertuples()]
        past_selected = st.selectbox("Past card", past_labels, key="past_card_select")
        past_event = past_events_df.iloc[past_labels.index(past_selected)]

        past_fights = conn.execute(
            """
            SELECT f.fight_id, f.fighter_1_id, f.fighter_2_id, f.weight_class, f.title_fight,
                   f.event_date, f.result, f.winner_id, f.method,
                   f1.name AS fighter_1_name, f2.name AS fighter_2_name
            FROM fights f
            JOIN fighters f1 ON f1.fighter_id = f.fighter_1_id
            JOIN fighters f2 ON f2.fighter_id = f.fighter_2_id
            WHERE f.event_id = ?
            ORDER BY f.fight_id
            """,
            (past_event["event_id"],),
        ).fetchall()
        past_fights_df = pd.DataFrame([dict(r) for r in past_fights])

        past_logistic = report.build_historical_card_report(
            conn, logistic_model, past_fights, n=n_window,
            calibration_table=logistic_calib, feature_columns=logistic_features,
        )
        past_gbm = report.build_historical_card_report(
            conn, gbm_model, past_fights, n=n_window,
            calibration_table=gbm_calib, feature_columns=gbm_features,
        )

        past_report = past_logistic.copy()
        past_report["logistic_prob"] = past_logistic["model_prob_fighter_1"]
        past_report["gbm_prob"] = past_gbm["model_prob_fighter_1"]
        past_report["avg_prob"] = (past_report["logistic_prob"] + past_report["gbm_prob"]) / 2
        past_report["edge_fighter_1"] = past_report["avg_prob"] - past_report["market_prob_fighter_1"]
        past_report["confidence"] = past_logistic["confidence"] if model_name == "logistic" else past_gbm["confidence"]
        past_report["weight_class"] = past_fights_df["weight_class"]
        past_report["title_fight"] = past_fights_df["title_fight"].astype(bool)

        past_report = add_strategy_signals(past_report, show_refined, show_refined_plus, show_elo)
        past_report["picked_side"] = past_report.apply(_picked_side, axis=1)
        past_report["Result"] = past_report.apply(
            lambda r: r["fighter_1"] if r["result"] == "fighter_1" else r["fighter_2"], axis=1,
        )
        past_report["Correct"] = past_report.apply(
            lambda r: "" if r["picked_side"] is None else ("✅" if r["picked_side"] == r["result"] else "❌"),
            axis=1,
        )

        past_display = past_report.copy()
        past_display["Logistic %"] = past_display["logistic_prob"].map(_fmt_pct)
        past_display["GBM %"] = past_display["gbm_prob"].map(_fmt_pct)
        past_display["Avg %"] = past_display["avg_prob"].map(_fmt_pct)
        past_display["Market %"] = past_display["market_prob_fighter_1"].map(_fmt_pct)
        past_display["Edge"] = past_display["edge_fighter_1"].map(_fmt_edge)
        past_display = past_display.rename(
            columns={"fighter_1": "Fighter 1", "fighter_2": "Fighter 2", "weight_class": "Weight class", "confidence": "Confidence"}
        )

        past_table_cols = [
            "Fighter 1", "Fighter 2", "Result", "Weight class", "Logistic %", "GBM %", "Avg %",
            "Market %", "Edge", "Confidence", "Strategy Signal", "Correct",
        ]
        st.dataframe(
            past_display[past_table_cols].style.apply(_highlight_row, axis=1),
            width="stretch",
            hide_index=True,
            height=min(35 * (len(past_display) + 1) + 3, 740),
            column_config={"Strategy Signal": st.column_config.TextColumn(width="medium")},
        )

        picked = past_report[past_report["picked_side"].notna()]
        if not picked.empty:
            wins = (picked["Correct"] == "✅").sum()
            st.caption(
                f"This card: {wins}W-{len(picked) - wins}L on the {len(picked)} fight(s) a strategy flagged "
                f"(out of {len(past_report)} total on the card). Market % and Edge use the fight's real "
                f"closing line, not a live snapshot."
            )
        else:
            st.caption(f"No strategy flagged any fight on this card (out of {len(past_report)} total).")
