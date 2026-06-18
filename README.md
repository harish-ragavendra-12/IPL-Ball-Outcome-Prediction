# 🏏 IPL Ball-by-Ball Outcome Prediction

## 📌 Project Overview

This project predicts the outcome of the next ball in an IPL match using Machine Learning. The model analyzes the current match situation, batter performance, bowler performance, match phase, and recent momentum to predict whether the next ball will result in:

* Dot Ball
* Single
* Double
* Four
* Six
* Wicket

The project demonstrates a complete end-to-end Machine Learning workflow including data analysis, feature engineering, preprocessing, model training, evaluation, and deployment using Streamlit.

---

## 🎯 Problem Statement

Predict the outcome of the next ball in an IPL match based on the current game situation and player statistics.

This is a multi-class classification problem where the target variable is the ball outcome.

---

## 📂 Dataset

The project uses IPL ball-by-ball and match-level datasets.

### Matches Dataset

Contains match information such as:

* Match ID
* Season
* Venue
* City
* Teams
* Toss Information
* Match Winner

### Deliveries Dataset

Contains ball-by-ball information such as:

* Match ID
* Inning
* Over
* Ball
* Batter
* Bowler
* Runs Scored
* Extras
* Wicket Information

---

## 🛠️ Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed detailed analysis on:

* Ball outcome distribution
* Runs distribution
* Wicket analysis
* Team performance
* Venue analysis
* Over-wise scoring patterns
* Top batsmen
* Top bowlers

### 2. Feature Engineering

Created match context and player performance features:

* Current Score
* Balls Bowled
* Balls Remaining
* Wickets Lost
* Current Run Rate
* Runs in Last 6 Balls
* Runs in Last 12 Balls
* Wickets in Last 12 Balls
* Striker Runs
* Striker Balls Faced
* Strike Rate
* Batter Average Runs Per Ball
* Bowler Wickets
* Bowler Economy
* Match Phase (Powerplay, Middle, Death)

### 3. Data Preprocessing

* Missing value handling
* Label Encoding
* Train-Test Split
* Dataset preparation for model training

### 4. Model Training

Models evaluated:

* Random Forest Classifier
* CatBoost Classifier

### 5. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report

---

## 🤖 Model Performance

| Model         | Accuracy | F1 Score |
| ------------- | -------- | -------- |
| Random Forest | 33.52%   | 35.12%   |
| CatBoost      | 44.37%   | 37.10%   |

### Best Model

✅ CatBoost Classifier

* Accuracy: 44.37%
* Weighted F1 Score: 37.10%

---

## 📊 Features Used

* inning
* over
* ball
* batting_team
* bowling_team
* batter
* bowler
* city
* venue
* phase
* current_score
* balls_bowled
* balls_remaining
* wickets_lost
* current_run_rate
* runs_last_6
* runs_last_12
* wkts_last_12
* striker_runs
* striker_balls
* striker_sr
* batter_avg_runs
* bowler_wickets
* bowler_economy

---

## 🚀 Streamlit Application

The project includes an interactive Streamlit web application that allows users to:

* Select match conditions
* Enter batter and bowler statistics
* Predict the outcome of the next ball
* View dataset overview
* Explore EDA visualizations

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
IPL_Ball_Outcome_Prediction/

│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
├── src/
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── train_model.py
│
├── processed_data/
│
├── visualizations/
│
├── models/
│   ├── ball_outcome_model.pkl
│   └── label_encoders.pkl
│
├── app.py
│
└── README.md
```

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* CatBoost
* Joblib
* Streamlit

---

## 🔮 Future Improvements

* Native categorical feature handling using CatBoost
* Batter vs Bowler historical statistics
* Advanced momentum features
* Real-time match prediction
* Model explainability using SHAP
* Hyperparameter tuning
* Cloud deployment

---

## 👨‍💻 Author

Harish Ragavendra

