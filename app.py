import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="IPL Ball Outcome Prediction",
    page_icon="🏏",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/ball_outcome_model.pkl"
    )

    encoders = joblib.load(
        "models/label_encoders.pkl"
    )

    return model, encoders

model, encoders = load_model()

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "processed_data/featured_dataset.csv"
    )

df = load_data()

# ==================================================
# HEADER
# ==================================================

st.title("🏏 IPL Ball Outcome Prediction")

st.markdown(
"""
Predict the outcome of the next ball in an IPL match using
Machine Learning.
"""
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Match Situation")

# Teams

teams = sorted(
    df["batting_team"].dropna().unique()
)

batting_team = st.sidebar.selectbox(
    "Batting Team",
    teams
)

bowling_team = st.sidebar.selectbox(
    "Bowling Team",
    teams
)

# Batter

batters = sorted(
    df["batter"].dropna().unique()
)

batter = st.sidebar.selectbox(
    "Batter",
    batters
)

# Bowler

bowlers = sorted(
    df["bowler"].dropna().unique()
)

bowler = st.sidebar.selectbox(
    "Bowler",
    bowlers
)

# City

cities = sorted(
    df["city"].dropna().unique()
)

city = st.sidebar.selectbox(
    "City",
    cities
)

# Venue

venues = sorted(
    df["venue"].dropna().unique()
)

venue = st.sidebar.selectbox(
    "Venue",
    venues
)

# Phase

phase = st.sidebar.selectbox(
    "Phase",
    ["Powerplay", "Middle", "Death"]
)

# ==================================================
# NUMERIC FEATURES
# ==================================================

inning = st.sidebar.number_input(
    "Inning",
    min_value=1,
    max_value=2,
    value=1
)

over = st.sidebar.slider(
    "Over",
    0,
    19,
    10
)

ball = st.sidebar.slider(
    "Ball",
    1,
    6,
    1
)

current_score = st.sidebar.number_input(
    "Current Score",
    0,
    300,
    100
)

balls_bowled = st.sidebar.number_input(
    "Balls Bowled",
    0,
    120,
    60
)

balls_remaining = st.sidebar.number_input(
    "Balls Remaining",
    0,
    120,
    60
)

wickets_lost = st.sidebar.number_input(
    "Wickets Lost",
    0,
    10,
    3
)

current_run_rate = st.sidebar.number_input(
    "Current Run Rate",
    0.0,
    20.0,
    8.0
)

runs_last_6 = st.sidebar.number_input(
    "Runs Last 6 Balls",
    0,
    36,
    8
)

runs_last_12 = st.sidebar.number_input(
    "Runs Last 12 Balls",
    0,
    72,
    15
)

wkts_last_12 = st.sidebar.number_input(
    "Wickets Last 12 Balls",
    0,
    10,
    1
)

striker_runs = st.sidebar.number_input(
    "Striker Runs",
    0,
    200,
    30
)

striker_balls = st.sidebar.number_input(
    "Striker Balls",
    0,
    120,
    20
)

striker_sr = st.sidebar.number_input(
    "Strike Rate",
    0.0,
    400.0,
    150.0
)

batter_avg_runs = st.sidebar.number_input(
    "Batter Avg Runs/Ball",
    0.0,
    6.0,
    1.2
)

bowler_wickets = st.sidebar.number_input(
    "Bowler Wickets",
    0,
    10,
    1
)

bowler_economy = st.sidebar.number_input(
    "Bowler Economy",
    0.0,
    20.0,
    8.0
)

# ==================================================
# PREDICTION
# ==================================================

if st.sidebar.button("Predict Outcome"):

    input_df = pd.DataFrame([{

        "inning": inning,
        "over": over,
        "ball": ball,

        "batting_team":
            encoders["batting_team"].transform(
                [batting_team]
            )[0],

        "bowling_team":
            encoders["bowling_team"].transform(
                [bowling_team]
            )[0],

        "batter":
            encoders["batter"].transform(
                [batter]
            )[0],

        "bowler":
            encoders["bowler"].transform(
                [bowler]
            )[0],

        "city":
            encoders["city"].transform(
                [city]
            )[0],

        "venue":
            encoders["venue"].transform(
                [venue]
            )[0],

        "phase":
            encoders["phase"].transform(
                [phase]
            )[0],

        "current_score": current_score,
        "balls_bowled": balls_bowled,
        "balls_remaining": balls_remaining,
        "wickets_lost": wickets_lost,
        "current_run_rate": current_run_rate,
        "runs_last_6": runs_last_6,
        "runs_last_12": runs_last_12,
        "wkts_last_12": wkts_last_12,
        "striker_runs": striker_runs,
        "striker_balls": striker_balls,
        "striker_sr": striker_sr,
        "batter_avg_runs": batter_avg_runs,
        "bowler_wickets": bowler_wickets,
        "bowler_economy": bowler_economy
    }])

    prediction = model.predict(input_df)[0]

    outcome = encoders["target"].inverse_transform(
        [prediction]
    )[0]

    st.success(
        f"Predicted Ball Outcome: {outcome}"
    )

# ==================================================
# DATASET OVERVIEW
# ==================================================

st.markdown("---")

st.subheader("Dataset Overview")

st.write(df.head())

st.write(
    f"Rows: {df.shape[0]:,}"
)

st.write(
    f"Columns: {df.shape[1]}"
)

# ==================================================
# VISUALIZATIONS
# ==================================================

st.markdown("---")

st.subheader("EDA Visualizations")

try:

    st.image(
        "visualizations/ball_outcome_distribution.png",
        use_container_width=True
    )

except:
    st.info(
        "Run eda.py first to generate charts."
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
"""
Developed using:

- Python
- Pandas
- Scikit-Learn
- CatBoost
- Streamlit
"""
)