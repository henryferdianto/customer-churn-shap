"""Generate a synthetic customer churn dataset with real signal."""
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 120, n)
    support_calls = rng.poisson(1.2, n)
    late_payments = rng.poisson(0.8, n)
    contract = rng.choice(["monthly", "yearly", "biennial"], n, p=[0.6, 0.3, 0.1])

    logit = (
        -1.2
        - 0.045 * tenure
        + 0.018 * monthly
        + 0.55 * support_calls
        + 0.6 * late_payments
        + np.where(contract == "monthly", 1.1, np.where(contract == "yearly", 0.2, -0.6))
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": np.round(monthly, 2),
        "support_calls": support_calls,
        "late_payments": late_payments,
        "contract": contract,
        "churn": churn,
    })


def main() -> Path:
    out = Path("data/raw/churn.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"wrote churn data -> {out}")
    return out


if __name__ == "__main__":
    main()
