import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# LOAD DATA
# ==================================================

matches = pd.read_csv("../data/matches.csv")
deliveries = pd.read_csv("../data/deliveries.csv")

# ==================================================
# DATASET OVERVIEW
# ==================================================

print("=" * 60)
print("MATCHES DATASET")
print("=" * 60)

print("Shape:", matches.shape)

print("\nMissing Values:")
print(matches.isnull().sum())

print("\nDuplicates:", matches.duplicated().sum())

print("\n")

print("=" * 60)
print("DELIVERIES DATASET")
print("=" * 60)

print("Shape:", deliveries.shape)

print("\nMissing Values:")
print(deliveries.isnull().sum())

print("\nDuplicates:", deliveries.duplicated().sum())

# ==================================================
# TARGET VARIABLE CREATION
# ==================================================

def ball_outcome(row):

    if row["is_wicket"] == 1:
        return "Wicket"

    elif row["batsman_runs"] == 0:
        return "Dot"

    elif row["batsman_runs"] == 1:
        return "Single"

    elif row["batsman_runs"] == 2:
        return "Double"

    elif row["batsman_runs"] == 3:
        return "Triple"

    elif row["batsman_runs"] == 4:
        return "Four"

    elif row["batsman_runs"] == 6:
        return "Six"

    else:
        return "Other"

deliveries["ball_outcome"] = deliveries.apply(
    ball_outcome,
    axis=1
)

# ==================================================
# BALL OUTCOME DISTRIBUTION
# ==================================================

plt.figure(figsize=(10,6))

sns.countplot(
    data=deliveries,
    x="ball_outcome",
    order=deliveries["ball_outcome"].value_counts().index
)

plt.title("Ball Outcome Distribution")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "../visualizations/ball_outcome_distribution.png"
)

plt.close()

# ==================================================
# RUNS DISTRIBUTION
# ==================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=deliveries,
    x="batsman_runs"
)

plt.title("Runs Scored Per Ball")

plt.tight_layout()

plt.savefig(
    "../visualizations/runs_distribution.png"
)

plt.close()

# ==================================================
# WICKET DISTRIBUTION
# ==================================================

plt.figure(figsize=(6,6))

deliveries["is_wicket"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Wicket Distribution")

plt.ylabel("")

plt.savefig(
    "../visualizations/wicket_distribution.png"
)

plt.close()

# ==================================================
# OVER WISE RUNS
# ==================================================

over_runs = (
    deliveries
    .groupby("over")["total_runs"]
    .mean()
)

plt.figure(figsize=(10,5))

sns.lineplot(
    x=over_runs.index,
    y=over_runs.values,
    marker="o"
)

plt.title("Average Runs Per Ball by Over")

plt.xlabel("Over")

plt.ylabel("Runs")

plt.tight_layout()

plt.savefig(
    "../visualizations/overwise_runs.png"
)

plt.close()

# ==================================================
# OVER WISE WICKETS
# ==================================================

over_wickets = (
    deliveries
    .groupby("over")["is_wicket"]
    .sum()
)

plt.figure(figsize=(10,5))

sns.lineplot(
    x=over_wickets.index,
    y=over_wickets.values,
    marker="o"
)

plt.title("Wickets by Over")

plt.xlabel("Over")

plt.ylabel("Total Wickets")

plt.tight_layout()

plt.savefig(
    "../visualizations/overwise_wickets.png"
)

plt.close()

# ==================================================
# TOP BATSMEN
# ==================================================

top_batsmen = (
    deliveries
    .groupby("batter")["batsman_runs"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_batsmen.values,
    y=top_batsmen.index
)

plt.title("Top 10 Run Scorers")

plt.tight_layout()

plt.savefig(
    "../visualizations/top_batsmen.png"
)

plt.close()

# ==================================================
# TOP BOWLERS
# ==================================================

top_bowlers = (
    deliveries
    .groupby("bowler")["is_wicket"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_bowlers.values,
    y=top_bowlers.index
)

plt.title("Top 10 Wicket Takers")

plt.tight_layout()

plt.savefig(
    "../visualizations/top_bowlers.png"
)

plt.close()

# ==================================================
# TEAM RUNS
# ==================================================

team_runs = (
    deliveries
    .groupby("batting_team")["total_runs"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,6))

sns.barplot(
    x=team_runs.values,
    y=team_runs.index
)

plt.title("Total Runs by Team")

plt.tight_layout()

plt.savefig(
    "../visualizations/team_runs.png"
)

plt.close()

# ==================================================
# VENUE ANALYSIS
# ==================================================

venue_matches = matches["venue"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=venue_matches.values,
    y=venue_matches.index
)

plt.title("Top 10 Venues")

plt.tight_layout()

plt.savefig(
    "../visualizations/top_venues.png"
)

plt.close()

# ==================================================
# TOSS DECISION ANALYSIS
# ==================================================

plt.figure(figsize=(6,6))

matches["toss_decision"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Toss Decisions")

plt.ylabel("")

plt.savefig(
    "../visualizations/toss_decisions.png"
)

plt.close()

# ==================================================
# CORRELATION HEATMAP
# ==================================================

numeric_cols = deliveries[
    [
        "inning",
        "over",
        "ball",
        "batsman_runs",
        "extra_runs",
        "total_runs",
        "is_wicket"
    ]
]

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "../visualizations/correlation_heatmap.png"
)

plt.close()

# ==================================================
# SUMMARY
# ==================================================

print("\nEDA COMPLETED SUCCESSFULLY")
print("Visualizations saved in visualizations folder")