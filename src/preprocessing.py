import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "../processed_data/featured_dataset.csv"
)

print("=" * 60)
print("DATA LOADED")
print("=" * 60)

print("Shape:", df.shape)

# ==================================================
# REMOVE MISSING VALUES
# ==================================================

df = df.dropna()

print("\nShape After NA Removal:", df.shape)

# ==================================================
# CATEGORICAL COLUMNS
# ==================================================

categorical_cols = [
    "batting_team",
    "bowling_team",
    "batter",
    "bowler",
    "city",
    "venue",
    "phase"
]

# ==================================================
# LABEL ENCODING
# ==================================================

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# ==================================================
# TARGET ENCODING
# ==================================================

target_encoder = LabelEncoder()

df["ball_outcome"] = target_encoder.fit_transform(
    df["ball_outcome"]
)

encoders["target"] = target_encoder

# ==================================================
# SAVE ENCODERS
# ==================================================

os.makedirs("../models", exist_ok=True)

joblib.dump(
    encoders,
    "../models/label_encoders.pkl"
)

# ==================================================
# FEATURES & TARGET
# ==================================================

X = df.drop(
    columns=["ball_outcome"]
)

y = df["ball_outcome"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================================
# SAVE DATASETS
# ==================================================

os.makedirs(
    "../processed_data",
    exist_ok=True
)

X_train.to_csv(
    "../processed_data/X_train.csv",
    index=False
)

X_test.to_csv(
    "../processed_data/X_test.csv",
    index=False
)

y_train.to_csv(
    "../processed_data/y_train.csv",
    index=False
)

y_test.to_csv(
    "../processed_data/y_test.csv",
    index=False
)

print("\nTarget Mapping")

for cls, value in zip(
    target_encoder.classes_,
    target_encoder.transform(
        target_encoder.classes_
    )
):
    print(f"{cls} --> {value}")

print("\nPreprocessing Completed")