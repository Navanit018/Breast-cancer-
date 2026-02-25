import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from scipy.stats import loguniform, randint

# 1) Load data
df = pd.read_csv("breast_cancer.csv")
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Label: M=1 (malignant), B=0 (benign)
df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
feature_cols = X.columns.tolist()

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3) Candidate models (advanced: compare + tune)
candidates = []

# --- SVM pipeline + tuning space ---
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True, random_state=42))
])
svm_params = {
    "model__C": loguniform(1e-3, 1e3),
    "model__gamma": loguniform(1e-4, 1e0),
    "model__kernel": ["rbf"]
}
candidates.append(("SVM", svm_pipe, svm_params))

# --- RandomForest (no scaling needed, but keep pipeline uniform) ---
rf_pipe = Pipeline([
    ("model", RandomForestClassifier(random_state=42))
])
rf_params = {
    "model__n_estimators": randint(200, 1200),
    "model__max_depth": randint(2, 20),
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 10),
    "model__max_features": ["sqrt", "log2", None],
    "model__class_weight": [None, "balanced"]
}
candidates.append(("RandomForest", rf_pipe, rf_params))

# If you install xgboost, you can add XGBoost later (best for tabular).

best_name, best_search, best_auc = None, None, -1

for name, pipe, params in candidates:
    print(f"\n=== Tuning {name} ===")
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    auc = search.best_score_
    print("Best CV ROC-AUC:", auc)
    print("Best params:", search.best_params_)

    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_search = search

# 4) Evaluate best model on test set
best_model = best_search.best_estimator_
proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, proba)

print("\n====================")
print("BEST MODEL:", best_name)
print("Test ROC-AUC:", test_auc)

# Choose threshold: prioritize recall (cancer detection)
threshold = 0.5  # you can optimize this later
pred = (proba >= threshold).astype(int)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nReport:\n", classification_report(y_test, pred, digits=4))

# 5) Save everything needed for deployment
artifact = {
    "model": best_model,
    "feature_cols": feature_cols,
    "threshold": threshold,
    "best_model_name": best_name,
    "test_auc": float(test_auc),
}
with open("best_pipeline.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved: best_pipeline.pkl")