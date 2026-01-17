import sys
import pandas as pd
import joblib

if len(sys.argv) != 3:
    print("Usage: python predict.py \"Team A\" \"Team B\"")
    exit(1)

home = sys.argv[1]
away = sys.argv[2]

print("Loading database...")
df = pd.read_csv("data/master_clean.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date")

def get_team_form(team, n=5):
    home_matches = df[df["home_team"] == team]
    away_matches = df[df["away_team"] == team]

    matches = pd.concat([home_matches, away_matches]).sort_values("date").tail(n)

    if len(matches) == 0:
        return None

    scored = []
    conceded = []

    for _, m in matches.iterrows():
        if m["home_team"] == team:
            scored.append(m["home_score"])
            conceded.append(m["away_score"])
        else:
            scored.append(m["away_score"])
            conceded.append(m["home_score"])

    return sum(scored)/len(scored), sum(conceded)/len(conceded)

print(f"Calculating form for {home} and {away}...")

home_form = get_team_form(home)
away_form = get_team_form(away)

if home_form is None or away_form is None:
    print("Not enough data for one of the teams.")
    exit(1)

home_avg_scored, home_avg_conceded = home_form
away_avg_scored, away_avg_conceded = away_form

X = pd.DataFrame([{
    "home_avg_scored": home_avg_scored,
    "home_avg_conceded": home_avg_conceded,
    "away_avg_scored": away_avg_scored,
    "away_avg_conceded": away_avg_conceded,
}])

print("Loading models...")

model_btts = joblib.load("models/model_btts.pkl")
model_over = joblib.load("models/model_over25.pkl")

p_btts = model_btts.predict_proba(X)[0][1]
p_over = model_over.predict_proba(X)[0][1]

print("=======================================")
print(f"MATCH: {home} vs {away}")
print("=======================================")
print(f"BTTS probability:    {p_btts*100:.1f} %")
print(f"Over 2.5 probability:{p_over*100:.1f} %")
print("=======================================")
