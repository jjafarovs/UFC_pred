"""UFC Edge dashboard (V1, informational) -- Streamlit, local/personal use only.

Shows upcoming UFC cards: model win probability, de-vigged live market
probability, the gap between them, a confidence label, and the underlying
per-fighter stats (rolling form, striking/grappling rates, physical diffs)
behind each prediction.

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

from src import features, market, report, strategy

DB_PATH = "db/ufc.db"

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
show_original = st.sidebar.checkbox("Highlight Original rule", value=True)
show_tighter = st.sidebar.checkbox("Highlight Tighter rule", value=True)
show_refined = st.sidebar.checkbox("Highlight Refined rule", value=True)
st.sidebar.caption(
    "All rules need the average of the logistic + GBM probabilities past a "
    "threshold, AND neither model individually below/above a confirm floor. "
    "**Original** (avg>62%/<38%, confirm>=55%/<=45%): backtested ROI +4.4%, "
    "95% CI [-0.9%,+9.7%] -- consistently positive, not yet statistically proven. "
    "**Tighter** (avg>75%/<25%, same confirm floor): backtested ROI +7.3%, "
    "95% CI [+1.0%,+13.3%] -- statistically significant, found via a threshold sweep. "
    "**Refined** (same thresholds as Original, plus: both fighters need >=3 tracked "
    "fights, and title fights are excluded): backtested ROI +10.9%, 95% CI "
    "[+3.8%,+17.6%] -- the strongest and most robust result found so far, significant "
    "across every retraining-schedule variant tested. "
    "See README's Walk-forward backtest section for the full methodology."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "No scheduled auto-refresh -- click to pull the latest upcoming card and live odds "
    "on demand. Rate-limited scraping, typically takes a few minutes."
)
if st.sidebar.button("🔄 Refresh upcoming card + live odds"):
    with st.spinner("Fetching upcoming card and live odds... this can take several minutes"):
        result = subprocess.run(
            [sys.executable, "-m", "src.refresh", "--mode", "upcoming"],
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        st.sidebar.success("Refreshed.")
        st.rerun()
    else:
        st.sidebar.error(f"Refresh failed:\n{result.stderr[-1000:]}")

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

card_report["original_signal"] = [
    strategy.bet_signal(lp, gp, **strategy.ORIGINAL_RULE) if show_original else None
    for lp, gp in zip(card_report["logistic_prob"], card_report["gbm_prob"])
]
card_report["tighter_signal"] = [
    strategy.bet_signal(lp, gp, **strategy.TIGHTER_RULE) if show_tighter else None
    for lp, gp in zip(card_report["logistic_prob"], card_report["gbm_prob"])
]
card_report["refined_signal"] = [
    strategy.bet_signal(lp, gp, f1_n_prior=f1n, f2_n_prior=f2n, title_fight=tf, **strategy.REFINED_RULE)
    if show_refined else None
    for lp, gp, f1n, f2n, tf in zip(
        card_report["logistic_prob"], card_report["gbm_prob"],
        card_report["f1_n_prior_fights"], card_report["f2_n_prior_fights"], card_report["title_fight"],
    )
]


def _signal_label(row) -> str:
    original, tighter, refined = row["original_signal"], row["tighter_signal"], row["refined_signal"]
    fighter = original or tighter or refined
    if fighter is None:
        return ""
    fighter_name = row["fighter_1"] if fighter == "fighter_1" else row["fighter_2"]
    active = [name for name, sig in [("Original", original), ("Tighter", tighter), ("Refined", refined)] if sig is not None]
    icon = "\U0001F525" if len(active) >= 3 else "⭐" if len(active) == 2 else "✅"
    return f"{icon} {' + '.join(active)} -> {fighter_name}"


card_report["Strategy Signal"] = card_report.apply(_signal_label, axis=1)

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


def _highlight_row(row):
    label = row["Strategy Signal"]
    if label.startswith("\U0001F525"):
        color = "background-color: rgba(255, 99, 71, 0.35)"  # both rules -- strongest highlight
    elif label.startswith("⭐"):
        color = "background-color: rgba(255, 215, 0, 0.30)"  # tighter rule only
    elif label.startswith("✅"):
        color = "background-color: rgba(60, 179, 113, 0.25)"  # original rule only
    else:
        color = ""
    return [color] * len(row)


st.dataframe(
    display_df[table_cols].style.apply(_highlight_row, axis=1),
    use_container_width=True,
    hide_index=True,
    height=min(35 * (len(display_df) + 1) + 3, 740),
    column_config={"Strategy Signal": st.column_config.TextColumn(width="medium")},
)
if show_original or show_tighter or show_refined:
    st.caption(
        "Highlighted rows are the fights the selected strategy/strategies flag as bettable -- "
        "everything else gets no bet under any rule. \U0001F525 = all 3 selected rules agree, "
        "⭐ = 2 of the selected rules agree, ✅ = 1 rule only. The label spells out exactly which "
        "rule(s) fired, e.g. \"Original + Refined\"."
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


comparison_rows = [
    ("Fights tracked (last N)", f1_roll["n_prior_fights"], f2_roll["n_prior_fights"]),
    ("Career fights (all-time)", f1_career["total_prior_fights"], f2_career["total_prior_fights"]),
    ("Win % (last N)", _pct(f1_roll["win_pct"]), _pct(f2_roll["win_pct"])),
    ("Current streak", f1_roll["current_streak"], f2_roll["current_streak"]),
    ("Finish rate (career wins by KO/sub)", _pct(f1_career["finish_rate"]), _pct(f2_career["finish_rate"])),
    ("Times finished (career losses by KO/sub)", _pct(f1_career["times_finished_rate"]), _pct(f2_career["times_finished_rate"])),
    ("Sig. strike accuracy", _pct(f1_roll["sig_str_acc"]), _pct(f2_roll["sig_str_acc"])),
    ("Sig. strike defense", _pct(f1_roll["sig_str_def"]), _pct(f2_roll["sig_str_def"])),
    ("Takedown accuracy", _pct(f1_roll["td_acc"]), _pct(f2_roll["td_acc"])),
    ("Takedown defense", _pct(f1_roll["td_def"]), _pct(f2_roll["td_def"])),
    ("Days since last fight", f1_roll["days_since_last_fight"], f2_roll["days_since_last_fight"]),
    ("Height (in)", f1_phys["height_in"], f2_phys["height_in"]),
    ("Reach (in)", f1_phys["reach_in"], f2_phys["reach_in"]),
    ("Stance", f1_phys["stance"], f2_phys["stance"]),
    ("Age (as of fight)", _num(f1_phys["age_years"], "{:.1f}"), _num(f2_phys["age_years"], "{:.1f}")),
]
comparison_df = pd.DataFrame(comparison_rows, columns=["Stat", row["fighter_1_name"], row["fighter_2_name"]])
st.dataframe(comparison_df, use_container_width=True, hide_index=True)
st.caption(
    "Numbers only, no highlighting of \"better\"/\"worse\" -- several of these "
    "(e.g. days since last fight, times finished) don't have a universally correct "
    "direction, and this view is meant to inform, not steer."
)

freshness = _freshness(conn, row["fighter_1_id"], row["fighter_2_id"])
st.caption(f"Edge = model − market. {freshness or 'no live odds captured for this fight yet'}.")
