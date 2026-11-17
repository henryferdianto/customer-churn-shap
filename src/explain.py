"""Feature-importance explanations with a SHAP-optional path."""
import argparse
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance


def explain(payload: dict, df: pd.DataFrame, target: str = "churn", n_repeats: int = 5):
    model = payload["model"]
    columns = payload["columns"]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True).reindex(
        columns=columns, fill_value=0
    )
    y = df[target]
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42,
                                    scoring="roc_auc")
    return (
        pd.DataFrame({"feature": columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/churn.joblib")
    ap.add_argument("--input", default="data/raw/churn.csv")
    args = ap.parse_args()
    payload = joblib.load(args.model)
    ranking = explain(payload, pd.read_csv(args.input))
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
