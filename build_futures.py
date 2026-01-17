import pandas as pd

print("Loading master_clean.csv ...")
df = pd.read_csv("data/master_clean.csv")

# sort by date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date")

# wynik
df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
df["over25"] = ((df["home_goals"] + df["away_goals"]) > 2).astype(int)

# słowniki statystyk drużyn
stats = {}

def get_team(team):
    if team not in stats:
        stats[team] = {
            "scored": [],
            "conceded": []
        }
    return stats[team]

rows = []

print("Building features...")

for _, r in df.iterrows():
    home = r["home_team"]
    away = r["away_team"]

    h = get_team(home)
    a = get_team(away)

    # rolling averages
    def avg(lst, n=5):
        if len(lst) == 0:
            return 0.0
        return sum(lst[-n:]) / min(len(lst), n)

    row = {
        "home_avg_scored": avg(h["scored"]),
        "home_avg_conceded": avg(h["conceded"]),
        "away_avg_scored": avg(a["scored"]),
        "away_avg_conceded": avg(a["conceded"]),
        "btts": r["btts"],
        "over25": r["over25"],
    }

    rows.append(row)

    # update stats AFTER match
    h["scored"].append(r["home_goals"])
    h["conceded"].append(r["away_goals"])
    a["scored"].append(r["away_goals"])
    a["conceded"].append(r["home_goals"])

features = pd.DataFrame(rows)

features = features.dropna()

features.to_csv("data/features.csv", index=False)

print("===================================")
print("FEATURES BUILT!")
print("Rows:", len(features))
print("Saved to: data/features.csv")
print("===================================")

