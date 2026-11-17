"""Generate SHAP explanations for the churn model."""
import argparse
import joblib
import pandas as pd
import shap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    model = joblib.load(args.model)
    X = pd.get_dummies(pd.read_csv(args.input).drop(columns=["churn"]), drop_first=True)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    shap.summary_plot(values, X, show=False)
    print("saved SHAP summary plot")


if __name__ == "__main__":
    main()
