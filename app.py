import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Starting app")

# =========================
# LOAD CSV FILES
# =========================

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print("📂 Found files:", files)

dfs = []

def try_load_csv(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {path} ({len(df)} rows)")
        return df
    except Exception as e:
        try:
            df = pd.read_csv(path, sep=";")
            print(f"✅ Loaded {path} with ; ({len(df)} rows)")
            return df
        except Exception as e2:
            print(f"❌ Failed to load {path}: {e}")
            return None

for f in files:
    df = try_load_csv(f)
    if df is None:
        continue
    dfs.append(df)

print(f"📊 Total raw tables: {len(dfs)}")

# =========================
# NORMALIZE MATCH TABLES
# =========================

def normalize_matches(df):
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    # find columns
    home_cols = ["home_team", "home", "home_team_name", "team1"]
    away_cols = ["away_team", "away", "away_team_name", "team2"]
    hg_cols = ["home_goals", "fulltime_home", "home_score", "score1", "fthg"]
    ag_cols = ["away_goals", "fulltime_away", "away_score", "score2", "ftag"]

    def find(col_list):
        for c in col_list:
            if c in df.columns:
                return c
        return None

    home = find(home_cols)
    away = find(away_cols)
    hg = find(hg_cols)
    ag = find(ag_cols)

    if not all([home, away, hg, ag]):
        return None

    out = pd.DataFrame()
    out["home"] = df[home]
    out["away"] = df[away]
    out["home_goals"] = pd.to_numeric(df[hg], errors="coerce")
    out["away_goals"] = pd.to_numeric(df[ag], errors="coerce")

    # date
    date_cols = [c for c in df.columns if "date" in c]
    if len(date_cols) > 0:
        out["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
    else:
        out["date"] = pd.NaT

    out = out.dropna(subset=["home", "away", "home_goals", "away_goals"])
    return out

matches = []

for df in dfs:
    norm = normalize_matches(df)
    if norm is not None and len(norm) > 0:
        matches.append(norm)

df = pd.concat(matches, ignore_index=True)

print("📊 Total match rows:", len(df))

# remove duplicate columns if any madness happened
df = df.loc[:, ~df.columns.duplicated()]

# =========================
# BASIC CLEANING
# =========================

df = df.dropna(subset=["home_goals", "away_goals"])
df["total_goals"] = df["home_goals"] + df["away_goals"]
df["over25"] = (df["total_goals"] > 2.5).astype(int)

# =========================
# FEATURE ENGINEERING
# =========================

print("🛠️ Feature engineering...")

df = df.sort_values("date")

teams = pd.concat([df["home"], df["away"]]).unique()

stats = {}

def get_stats(team):
    if team not in stats:
        stats[team] = {
            "gs": [],
            "ga": [],
        }
    return stats[team]

features = []

rows = []

for _, row in df.iterrows():
    home = row["home"]
    away = row["away"]

    hs = get_stats(home)
    as_ = get_stats(away)

    def avg(lst):
        if len(lst) == 0:
            return 1.5
        return np.mean(lst[-10:])

    h_gs = avg(hs["gs"])
    h_ga = avg(hs["ga"])
    a_gs = avg(as_["gs"])
    a_ga = avg(as_["ga"])

    feat = {
        "h_gs": h_gs,
        "h_ga": h_ga,
        "a_gs": a_gs,
        "a_ga": a_ga,
        "attack_diff": h_gs - a_gs,
        "defense_diff": a_ga - h_ga,
        "goal_expectancy": h_gs + a_gs,
    }

    rows.append(feat)

    # update stats
    hs["gs"].append(row["home_goals"])
    hs["ga"].append(row["away_goals"])
    as_["gs"].append(row["away_goals"])
    as_["ga"].append(row["home_goals"])

X = pd.DataFrame(rows)
y = df["over25"].values

print("📐 Feature matrix:", X.shape)

# =========================
# TRAIN MODEL
# =========================

print("🤖 Training Over2.5 model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, pred)
roc = roc_auc_score(y_test, proba)

print(f"✅ OVER2.5 Accuracy: {acc:.4f}")
print(f"✅ OVER2.5 ROC-AUC: {roc:.4f}")

print("\n📄 OVER classification report:")
print(classification_report(y_test, pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/model_over25.pkl")
joblib.dump(list(X.columns), "models/features.pkl")
print("💾 Saved models/model_over25.pkl")

# =========================
# FEATURE IMPORTANCE
# =========================

fi = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n📊 FEATURE IMPORTANCE:")
print(fi)

fi.to_csv("feature_importance.csv", index=False)
print("💾 Saved feature_importance.csv")

print("\n🏁 TRAINING DONE.")

