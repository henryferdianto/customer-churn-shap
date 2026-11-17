"""Train a churn classifier with class-imbalance handling."""
import argparse
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", default="churn")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    X = pd.get_dummies(df.drop(columns=[args.target]), drop_first=True)
    y = df[args.target]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        eval_metric="auc", random_state=42,
    )
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"hold-out ROC-AUC: {auc:.4f}")
    joblib.dump(model, "models/churn.joblib")


if __name__ == "__main__":
    main()
