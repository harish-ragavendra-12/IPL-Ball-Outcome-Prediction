import pandas as pd
import numpy as np
import os

# ==================================================
# LOAD DATA
# ==================================================

deliveries = pd.read_csv("../data/deliveries.csv")
matches = pd.read_csv("../data/matches.csv")

print("=" * 60)
print("DATA LOADED")
print("=" * 60)

# ==================================================
# SORT DATA
# ==================================================

deliveries = deliveries.sort_values(
    ["match_id", "inning", "over", "ball"]
)

# ==================================================
# TARGET VARIABLE
# ==================================================

def create_ball_outcome(row):

    if row["is_wicket"] == 1:
        return "Wicket"

    runs = row["batsman_runs"]

    if runs == 0:
        return "Dot"
    elif runs == 1:
        return "Single"
    elif runs == 2:
        return "Double"
    elif runs == 4:
        return "Four"
    elif runs == 6:
        return "Six"
    else:
        return "Other"

deliveries["ball_outcome"] = deliveries.apply(
    create_ball_outcome,
    axis=1
)

# Remove rare class
deliveries = deliveries[
    deliveries["ball_outcome"] != "Other"
]

# ==================================================
# CURRENT SCORE
# ==================================================

deliveries["current_score"] = (
    deliveries
    .groupby(["match_id", "inning"])["total_runs"]
    .cumsum()
)

deliveries["current_score"] = (
    deliveries["current_score"]
    - deliveries["total_runs"]
)

# ==================================================
# BALLS BOWLED
# ==================================================

deliveries["balls_bowled"] = (
    deliveries
    .groupby(["match_id", "inning"])
    .cumcount()
)

# ==================================================
# WICKETS LOST
# ==================================================

deliveries["wickets_lost"] = (
    deliveries
    .groupby(["match_id", "inning"])["is_wicket"]
    .cumsum()
)

deliveries["wickets_lost"] = (
    deliveries["wickets_lost"]
    - deliveries["is_wicket"]
)

# ==================================================
# BALLS REMAINING
# ==================================================

deliveries["balls_remaining"] = (
    120 - deliveries["balls_bowled"]
)

# ==================================================
# CURRENT RUN RATE
# ==================================================

deliveries["current_run_rate"] = np.where(
    deliveries["balls_bowled"] > 0,
    deliveries["current_score"]
    / (deliveries["balls_bowled"] / 6),
    0
)

# ==================================================
# RUNS LAST 6 BALLS
# ==================================================

deliveries["runs_last_6"] = (
    deliveries
    .groupby(["match_id", "inning"])["total_runs"]
    .transform(
        lambda x: x.shift(1)
        .rolling(6, min_periods=1)
        .sum()
    )
)

deliveries["runs_last_6"] = (
    deliveries["runs_last_6"]
    .fillna(0)
)

# ==================================================
# RUNS LAST 12 BALLS
# ==================================================

deliveries["runs_last_12"] = (
    deliveries
    .groupby(["match_id", "inning"])["total_runs"]
    .transform(
        lambda x: x.shift(1)
        .rolling(12, min_periods=1)
        .sum()
    )
)

deliveries["runs_last_12"] = (
    deliveries["runs_last_12"]
    .fillna(0)
)

# ==================================================
# WICKETS LAST 12 BALLS
# ==================================================

deliveries["wkts_last_12"] = (
    deliveries
    .groupby(["match_id", "inning"])["is_wicket"]
    .transform(
        lambda x: x.shift(1)
        .rolling(12, min_periods=1)
        .sum()
    )
)

deliveries["wkts_last_12"] = (
    deliveries["wkts_last_12"]
    .fillna(0)
)

# ==================================================
# BATTER FEATURES
# ==================================================

deliveries["striker_runs"] = (
    deliveries
    .groupby(
        ["match_id", "batter"]
    )["batsman_runs"]
    .cumsum()
)

deliveries["striker_runs"] = (
    deliveries["striker_runs"]
    - deliveries["batsman_runs"]
)

deliveries["striker_balls"] = (
    deliveries
    .groupby(
        ["match_id", "batter"]
    )
    .cumcount()
)

deliveries["striker_sr"] = np.where(
    deliveries["striker_balls"] > 0,
    deliveries["striker_runs"]
    /
    deliveries["striker_balls"]
    * 100,
    0
)

deliveries["batter_avg_runs"] = np.where(
    deliveries["striker_balls"] > 0,
    deliveries["striker_runs"]
    /
    deliveries["striker_balls"],
    0
)

# ==================================================
# BOWLER FEATURES
# ==================================================

deliveries["bowler_wickets"] = (
    deliveries
    .groupby(
        ["match_id", "bowler"]
    )["is_wicket"]
    .cumsum()
)

deliveries["bowler_wickets"] = (
    deliveries["bowler_wickets"]
    - deliveries["is_wicket"]
)

deliveries["bowler_runs"] = (
    deliveries
    .groupby(
        ["match_id", "bowler"]
    )["total_runs"]
    .cumsum()
)

deliveries["bowler_runs"] = (
    deliveries["bowler_runs"]
    - deliveries["total_runs"]
)

deliveries["bowler_balls"] = (
    deliveries
    .groupby(
        ["match_id", "bowler"]
    )
    .cumcount()
)

deliveries["bowler_economy"] = np.where(
    deliveries["bowler_balls"] > 0,
    deliveries["bowler_runs"]
    /
    (deliveries["bowler_balls"] / 6),
    0
)

# ==================================================
# MATCH PHASE
# ==================================================

def get_phase(over):

    if over <= 5:
        return "Powerplay"

    elif over <= 14:
        return "Middle"

    else:
        return "Death"

deliveries["phase"] = deliveries["over"].apply(
    get_phase
)

# ==================================================
# MERGE MATCH DATA
# ==================================================

match_features = matches[
    [
        "id",
        "city",
        "venue"
    ]
]

deliveries = deliveries.merge(
    match_features,
    left_on="match_id",
    right_on="id",
    how="left"
)

# ==================================================
# FINAL DATASET
# ==================================================

final_df = deliveries[
    [
        "inning",
        "over",
        "ball",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "city",
        "venue",
        "phase",
        "current_score",
        "balls_bowled",
        "balls_remaining",
        "wickets_lost",
        "current_run_rate",
        "runs_last_6",
        "runs_last_12",
        "wkts_last_12",
        "striker_runs",
        "striker_balls",
        "striker_sr",
        "batter_avg_runs",
        "bowler_wickets",
        "bowler_economy",
        "ball_outcome"
    ]
]

# ==================================================
# SAVE
# ==================================================

os.makedirs(
    "../processed_data",
    exist_ok=True
)

final_df.to_csv(
    "../processed_data/featured_dataset.csv",
    index=False
)

print("=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("Shape:", final_df.shape)

print("\nTarget Distribution:")
print(
    final_df["ball_outcome"]
    .value_counts()
)

print(
    "\nSaved: ../processed_data/featured_dataset.csv"
)