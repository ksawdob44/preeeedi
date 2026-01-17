import pandas as pd
import numpy as np
import glob

DATA_FILES = [
    "data/PL23.csv",
    "data/LaLiga23.csv",
    "data/Bundesliga23.csv",
    "data/SerieA23.csv",
    "data/Ligue1.csv",
]

def load_data():
    dfs = []
    for f in DATA_FILES:
        df = pd.read_csv(f)
        print(f"✅ Loaded {f} ({len(df)})")

        df = df.rename(columns={
            "HomeTeam": "home",
            "AwayTeam": "away",
            "FTHG": "hg",
            "FTAG": "ag",
            "Date": "date"
        })

        df = df[["home", "away", "hg", "ag"]]
        df = df.dropna()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)

    print("📊 Total matches:", len(df))
    return df


def team_stats(df, team, n=10):
    games = df[(df.home == team) | (df.away == team)].tail(n)
    if len(games) == 0:
        return None

    gs, ga, pts, btts, over25 = [], [], [], [], []

    for _, r in games.iterrows():
        if r.home == team:
            gfor, gag = r.hg, r.ag
        else:
            gfor, gag = r.ag, r.hg

        gs.append(gfor)
        ga.append(gag)
        btts.append(1 if r.hg > 0 and r.ag > 0 else 0)
        over25.append(1 if (r.hg + r.ag) > 2.5 else 0)

        if gfor > gag: pts.append(3)
        elif gfor == gag: pts.append(1)
        else: pts.append(0)

    return {
        "gs": np.mean(gs),
        "ga": np.mean(ga),
        "gd": np.mean(gs) - np.mean(ga),
        "btts": np.mean(btts),
        "over25": np.mean(over25),
        "ppg": np.mean(pts),
        "form": np.mean(pts[-5:]) if len(pts) >= 5 else np.mean(pts)
    }


def build_features(df):
    rows = []

    for i in range(len(df)):
        r = df.iloc[i]
        home = r.home
        away = r.away

        h = team_stats(df.iloc[:i], home)
        a = team_stats(df.iloc[:i], away)

        if h is None or a is None:
            continue

        feats = {
            "h_gs": h["gs"], "h_ga": h["ga"], "h_gd": h["gd"], "h_ppg": h["ppg"],
            "a_gs": a["gs"], "a_ga": a["ga"], "a_gd": a["gd"], "a_ppg": a["ppg"],
            "form_diff": h["form"] - a["form"],
            "attack_diff": h["gs"] - a["gs"],
            "defense_diff": a["ga"] - h["ga"],
            "goal_expectancy": (h["gs"] + a["gs"]) / 2,

            "btts": 1 if (r.hg > 0 and r.ag > 0) else 0,
            "over25": 1 if (r.hg + r.ag) > 2.5 else 0,
        }

        rows.append(feats)

    df_feat = pd.DataFrame(rows)
    return df_feat
