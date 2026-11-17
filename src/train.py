"""Train a churn classifier with class-imbalance handling."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def prepare(df: pd.DataFrame, target: str = "churn"):
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    return X, df[target]


def train(df: pd.DataFrame, target: str = "churn"):
    X, y = prepare(df, target)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    models = {
        "rf_balanced": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=42
        ),
        "histgb": HistGradientBoostingClassifier(random_state=42),
    }
    best, best_auc = None, -1.0
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        print(f"{name}: hold-out ROC-AUC = {auc:.4f}")
        if auc > best_auc:
            best, best_auc = model, auc
    return best, list(X.columns), best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/churn.csv")
    ap.add_argument("--target", default="churn")
    args = ap.parse_args()
    if not Path(args.input).exists():
        from src.sample_data import main as make_sample

        make_sample()
    model, columns, auc = train(pd.read_csv(args.input), args.target)
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": columns, "auc": auc}, "models/churn.joblib")
    print(f"best model saved (AUC {auc:.4f})")


if __name__ == "__main__":
    main()
