import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

print("📊 Loading features...")
df = pd.read_csv("data/features.csv")

# =========================
# FEATURE MATRIX
# =========================
X = df.drop(columns=["btts", "over25"])
y_btts = df["btts"]
y_over = df["over25"]

# =========================
# SPLIT
# =========================
X_train, X_test, yb_train, yb_test = train_test_split(X, y_btts, test_size=0.2, random_state=42)
_, _, yo_train, yo_test = train_test_split(X, y_over, test_size=0.2, random_state=42)

# =========================
# BTTS MODEL
# =========================
print("🤖 Training BTTS model...")
model_btts = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model_btts.fit(X_train, yb_train)

pred_btts = model_btts.predict(X_test)
proba_btts = model_btts.predict_proba(X_test)[:,1]

print("✅ BTTS Accuracy:", accuracy_score(yb_test, pred_btts))
print("✅ BTTS ROC-AUC:", roc_auc_score(yb_test, proba_btts))
print("\n📄 BTTS REPORT:\n", classification_report(yb_test, pred_btts))

joblib.dump(model_btts, "models/model_btts.pkl")

# =========================
# OVER 2.5 MODEL
# =========================
print("🤖 Training Over2.5 model...")
model_over = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model_over.fit(X_train, yo_train)

pred_over = model_over.predict(X_test)
proba_over = model_over.predict_proba(X_test)[:,1]

print("✅ OVER Accuracy:", accuracy_score(yo_test, pred_over))
print("✅ OVER ROC-AUC:", roc_auc_score(yo_test, proba_over))
print("\n📄 OVER REPORT:\n", classification_report(yo_test, pred_over))

joblib.dump(model_over, "models/model_over25.pkl")

# =========================
# FEATURE IMPORTANCE
# =========================
imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model_over.feature_importances_
}).sort_values("importance", ascending=False)

imp.to_csv("models/feature_importance.csv", index=False)

print("\n🏁 TRAINING DONE.")
print("Saved:")
print(" - models/model_btts.pkl")
print(" - models/model_over25.pkl")
print(" - models/feature_importance.csv")
