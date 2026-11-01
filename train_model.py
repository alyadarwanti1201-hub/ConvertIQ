"""Mereproduksi model final notebook ConvertIQ_Project.ipynb:
Logistic Regression Balanced (tanpa PageValues), threshold 0,50."""
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from feature_engineering import create_features

df = pd.read_csv("convertiq_dashboard_data.csv")

# PageValues dikeluarkan (risiko data leakage), sama dengan notebook
X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

numerical = ["Administrative", "Administrative_Duration", "Informational",
             "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
             "BounceRates", "ExitRates", "SpecialDay"]
categorical = ["Month", "OperatingSystems", "Browser", "Region",
               "TrafficType", "VisitorType", "Weekend"]
numerical_fe = numerical + ["TotalPages", "TotalDuration",
                            "ProductPageRatio", "AvgProductDuration"]

X_train_fe, X_test_fe = create_features(X_train), create_features(X_test)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), numerical_fe),
    ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])


def pipe(clf):
    return Pipeline([("preprocessing", preprocessor), ("classifier", clf)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Regression Balanced": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Decision Tree Balanced": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest Balanced": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
}
cv_table = {}
for name, clf in models.items():
    s = cross_validate(pipe(clf), X_train_fe, y_train, cv=cv, scoring=metrics, n_jobs=-1)
    cv_table[name] = {m: float(s[f"test_{m}"].mean()) for m in metrics}
    print(name, {k: round(v, 4) for k, v in cv_table[name].items()})

grid = GridSearchCV(
    pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    {"classifier__C": [0.01, 0.1, 1, 10, 100], "classifier__solver": ["lbfgs"]},
    scoring="f1", cv=cv, n_jobs=-1)
grid.fit(X_train_fe, y_train)
final_model = grid.best_estimator_
print("Best params:", grid.best_params_, "CV F1:", round(grid.best_score_, 4))

THRESHOLD = 0.50
prob = final_model.predict_proba(X_test_fe)[:, 1]
pred = (prob >= THRESHOLD).astype(int)
test = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob)}
cm = confusion_matrix(y_test, pred).tolist()
print("TEST", {k: round(v, 4) for k, v in test.items()}, cm)
print("Sample prob X_test[0]:", round(float(final_model.predict_proba(X_test_fe.iloc[[0]])[0, 1]), 6))

oof = cross_val_predict(final_model, X_train_fe, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
thr_rows = []
for t in np.arange(0.10, 0.91, 0.05):
    p = (oof >= t).astype(int)
    thr_rows.append({"Threshold": round(float(t), 2),
                     "Precision": float(precision_score(y_train, p, zero_division=0)),
                     "Recall": float(recall_score(y_train, p, zero_division=0)),
                     "F1": float(f1_score(y_train, p, zero_division=0))})

names = final_model.named_steps["preprocessing"].get_feature_names_out()
coef = pd.Series(final_model.named_steps["classifier"].coef_[0], index=names)
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)

config = {
    "model": final_model, "threshold": THRESHOLD,
    "input_features": X_train.columns.tolist(),
    "model_features": X_train_fe.columns.tolist(),
    "model_name": "ConvertIQ - Logistic Regression Balanced", "target": "Revenue",
}
joblib.dump(config, "convertiq_model.joblib")
meta = {"sklearn_version": sklearn.__version__, "threshold": THRESHOLD,
        "best_params": grid.best_params_, "n_train": len(X_train), "n_test": len(X_test),
        "cv_table": cv_table, "test": test, "confusion": cm, "thresholds": thr_rows,
        "top_coef": top.to_dict()}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Tersimpan: convertiq_model.joblib, model_meta.json")
