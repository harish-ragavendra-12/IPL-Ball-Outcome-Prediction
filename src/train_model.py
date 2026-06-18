import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from catboost import CatBoostClassifier

# ==================================================
# LOAD DATA
# ==================================================

X_train = pd.read_csv(
    "../processed_data/X_train.csv"
)

X_test = pd.read_csv(
    "../processed_data/X_test.csv"
)

y_train = pd.read_csv(
    "../processed_data/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "../processed_data/y_test.csv"
).squeeze()

print("=" * 60)
print("DATA LOADED")
print("=" * 60)

# ==================================================
# MODELS
# ==================================================

models = {

    "Random Forest":

        RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "CatBoost":

        CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function="MultiClass",
            verbose=100,
            random_state=42
        )
}

# ==================================================
# TRAINING
# ==================================================

results = []

best_model = None
best_model_name = None
best_f1 = 0

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING : {name}")
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        "\nClassification Report\n"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append(
        [name, accuracy, f1]
    )

    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_model_name = name

# ==================================================
# MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "F1 Score"
    ]
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

# ==================================================
# SAVE BEST MODEL
# ==================================================

os.makedirs(
    "../models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "../models/ball_outcome_model.pkl"
)

results_df.to_csv(
    "../models/model_results.csv",
    index=False
)

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(best_model_name)

print(
    f"F1 Score : {best_f1:.4f}"
)

print(
    "\nSaved : ../models/ball_outcome_model.pkl"
)