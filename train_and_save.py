# ============================================================
#  train_and_save.py
#  Run this ONCE to train the model and save it as model.pkl
#  Usage: python train_and_save.py
#  Requires: cancer_reg.csv in the same directory
# ============================================================

import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("cancer_reg.csv")

# ── 2. FEATURE ENGINEERING ──────────────────────────────────
# Derived feature: Case Fatality Rate
df["cfr"] = df["avgdeathsperyear"] / (df["avganncount"] + 1)

# Drop non-numeric / identifier columns
drop_cols = ["Geography", "binnedInc", "target_deathrate"]
feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]

X = df[feature_cols].copy()
MEDIAN_VAL = df["target_deathrate"].median()
y = (df["target_deathrate"] > MEDIAN_VAL).astype(int)  # 1 = High Risk, 0 = Low Risk

print(f"Features used ({len(feature_cols)}): {feature_cols}")
print(f"Median death-rate threshold: {MEDIAN_VAL:.2f}")

# ── 3. IMPUTE & SCALE ───────────────────────────────────────
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ── 4. TRAIN / TEST SPLIT ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. BEST MODEL (HistGradientBoosting — typically top-1) ──
print("\nTraining model...")
model = HistGradientBoostingClassifier(
    max_iter=600,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=15,
    l2_regularization=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {acc:.4f}")

# ── 6. SAVE MODEL + SCALER + IMPUTER + META ─────────────────
bundle = {
    "model":        model,
    "scaler":       scaler,
    "imputer":      imputer,
    "feature_cols": feature_cols,
    "median_val":   MEDIAN_VAL,
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n✅  model.pkl saved successfully!")
print("    Now run:  streamlit run app.py")
