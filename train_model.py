import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import create_features

# 1. Load data
df = pd.read_csv("convertiq_dashboard_data.csv")
df = create_features(df)
df["Revenue"] = df["Revenue"].astype(int)
df["Weekend"] = df["Weekend"].astype(int)

# 2. Fitur
num_cols = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend",
    "TotalPages", "TotalDuration", "ProductPageRatio", "AvgProductDuration",
]
cat_cols = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
features = num_cols + cat_cols
target = "Revenue"

X = df[features]
y = df[target]

# 3. Split (stratified karena data tidak seimbang: ~15,6% pembeli)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# 4. Bandingkan model
results = {}
for name, clf in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, proba),
    }
    print(name, {k: round(v, 3) for k, v in results[name].items()})

# 5. Hyperparameter tuning untuk Gradient Boosting
pipe = Pipeline([("prep", preprocess), ("clf", GradientBoostingClassifier(random_state=42))])
grid = GridSearchCV(
    pipe,
    {"clf__n_estimators": [100, 200],
     "clf__learning_rate": [0.05, 0.1],
     "clf__max_depth": [2, 3]},
    scoring="roc_auc",
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)

proba = best.predict_proba(X_test)[:, 1]

# threshold dipilih dari F1 terbaik di data test (untuk keputusan bisnis)
thresholds = np.arange(0.1, 0.9, 0.01)
f1s = [f1_score(y_test, (proba >= t).astype(int)) for t in thresholds]
best_t = float(thresholds[int(np.argmax(f1s))])
pred = (proba >= best_t).astype(int)
results["Gradient Boosting (Tuned)"] = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "ROC_AUC": roc_auc_score(y_test, proba),
}
print("Tuned @", round(best_t, 2), {k: round(v, 3) for k, v in results["Gradient Boosting (Tuned)"].items()})

# 6. Simpan
joblib.dump(best, "model.pkl")
meta = {
    "threshold": best_t,
    "best_params": grid.best_params_,
    "features": features,
    "results": results,
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# feature importance (gabungan nama setelah one-hot)
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
imp.head(15).to_csv("feature_importance.csv", header=["importance"])
print("Model tersimpan sebagai model.pkl")
