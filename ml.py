import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DB = "matches.db"

def load_data():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("""
        SELECT
            m.fixture_id,
            m.home_goals,
            m.away_goals,
            p.result_btts,
            p.result_over25
        FROM matches m
        JOIN predictions p ON m.fixture_id = p.fixture_id
        WHERE p.result_btts IS NOT NULL
    """, conn)

    conn.close()
    return df


def make_features(df):
    # Proste featury na start:
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = abs(df["home_goals"] - df["away_goals"])

    X = df[["total_goals", "goal_diff"]]
    y_btts = df["result_btts"]
    y_over = df["result_over25"]

    return X, y_btts, y_over


def train_models():
    df = load_data()
    if len(df) < 20:
        print("❌ Za mało danych do trenowania")
        return

    X, y_btts, y_over = make_features(df)

    X_train, X_test, yb_train, yb_test = train_test_split(X, y_btts, test_size=0.2)
    X_train2, X_test2, yo_train, yo_test = train_test_split(X, y_over, test_size=0.2)

    model_btts = RandomForestClassifier(n_estimators=200)
    model_over = RandomForestClassifier(n_estimators=200)

    model_btts.fit(X_train, yb_train)
    model_over.fit(X_train2, yo_train)

    pred_btts = model_btts.predict(X_test)
    pred_over = model_over.predict(X_test2)

    print("✅ BTTS accuracy:", accuracy_score(yb_test, pred_btts))
    print("✅ Over25 accuracy:", accuracy_score(yo_test, pred_over))

    joblib.dump(model_btts, "model_btts.pkl")
    joblib.dump(model_over, "model_over.pkl")

    print("💾 Modele zapisane")


if __name__ == "__main__":
    train_models()
