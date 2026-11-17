import joblib
import pandas as pd

from src.explain import explain
from src.retention import campaign_roi
from src.sample_data import make
from src.train import train


def main():
    df = make()
    df.to_csv("data/raw/churn.csv", index=False)
    model, columns, auc = train(df)
    payload = {"model": model, "columns": columns, "auc": auc}
    print("\nTop churn drivers:")
    print(explain(payload, df).head(5).to_string(index=False))
    probs = model.predict_proba(
        pd.get_dummies(df.drop(columns=["churn"]), drop_first=True).reindex(columns=columns, fill_value=0)
    )[:, 1]
    print("\nCampaign ROI:", campaign_roi(probs, clv=500, contact_cost=20,
                                          success_rate=0.25, n_contacted=200))


if __name__ == "__main__":
    main()
