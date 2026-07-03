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

import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st

from src import features, market, report

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
    return fitted_model, model_path, calibration_table


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

model_name = st.sidebar.radio("Model", ["logistic", "gbm"], index=0)
n_window = st.sidebar.slider("Rolling form window (last N fights)", min_value=3, max_value=10, value=5)
fitted_model, model_path, calibration_table = get_model(model_name)
st.sidebar.caption(f"Model artifact: `{model_path.name}`")

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
card_report = report.build_card_report(
    conn, fitted_model, matchups, as_of_date=as_of, odds_type="live", n=n_window, calibration_table=calibration_table
)
card_report["weight_class"] = fights_df["weight_class"]
card_report["title_fight"] = fights_df["title_fight"].astype(bool)

display_df = card_report.copy()
display_df["Model %"] = display_df["model_prob_fighter_1"].map(_fmt_pct)
display_df["Market %"] = display_df["market_prob_fighter_1"].map(_fmt_pct)
display_df["Edge"] = display_df["edge_fighter_1"].map(_fmt_edge)
display_df = display_df.rename(
    columns={"fighter_1": "Fighter 1", "fighter_2": "Fighter 2", "weight_class": "Weight class", "confidence": "Confidence"}
)
st.dataframe(
    display_df[["Fighter 1", "Fighter 2", "Weight class", "Model %", "Market %", "Edge", "Confidence"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Fight detail")
fight_choice = st.selectbox(
    "Select a fight for the underlying stat breakdown",
    range(len(fights_df)),
    format_func=lambda i: f"{fights_df.iloc[i]['fighter_1_name']} vs. {fights_df.iloc[i]['fighter_2_name']}",
)
row = fights_df.iloc[fight_choice]
report_row = card_report.iloc[fight_choice]

col1, col2 = st.columns(2)
f1_roll = features.fighter_rolling_features(conn, row["fighter_1_id"], as_of, n=n_window)
f2_roll = features.fighter_rolling_features(conn, row["fighter_2_id"], as_of, n=n_window)
f1_phys = features.fighter_physical_features(conn, row["fighter_1_id"], as_of)
f2_phys = features.fighter_physical_features(conn, row["fighter_2_id"], as_of)

for col, name, roll, phys in [
    (col1, row["fighter_1_name"], f1_roll, f1_phys),
    (col2, row["fighter_2_name"], f2_roll, f2_phys),
]:
    with col:
        st.markdown(f"### {name}")
        st.metric("Recent record (last N)", f"{roll['n_prior_fights']} fights tracked")
        st.write(
            {
                "Win % (last N)": None if roll["win_pct"] is None else f"{roll['win_pct']*100:.0f}%",
                "Current streak": roll["current_streak"],
                "Sig. strike accuracy": None if roll["sig_str_acc"] is None else f"{roll['sig_str_acc']*100:.0f}%",
                "Sig. strike defense": None if roll["sig_str_def"] is None else f"{roll['sig_str_def']*100:.0f}%",
                "Takedown accuracy": None if roll["td_acc"] is None else f"{roll['td_acc']*100:.0f}%",
                "Takedown defense": None if roll["td_def"] is None else f"{roll['td_def']*100:.0f}%",
                "Days since last fight": roll["days_since_last_fight"],
                "Height (in)": phys["height_in"],
                "Reach (in)": phys["reach_in"],
                "Stance": phys["stance"],
                "Age (as of fight)": None if phys["age_years"] is None else f"{phys['age_years']:.1f}",
            }
        )

freshness = _freshness(conn, row["fighter_1_id"], row["fighter_2_id"])
st.caption(f"Edge = model − market. {freshness or 'no live odds captured for this fight yet'}.")
