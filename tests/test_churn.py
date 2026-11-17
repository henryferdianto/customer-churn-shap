import pandas as pd

from src.explain import explain
from src.retention import campaign_roi
from src.sample_data import make
from src.train import prepare, train


def test_retention_roi_math():
    out = campaign_roi([0.9, 0.1], clv=100, contact_cost=10, success_rate=0.5)
    assert out["contacted"] == 2
    assert out["clv_saved"] == 50.0
    assert out["net_roi"] == 30.0


def test_model_learns_signal():
    df = make(800)
    _, _, auc = train(df)
    assert auc > 0.7


def test_explain_returns_ranked_features():
    df = make(500)
    model, columns, _ = train(df)
    ranking = explain({"model": model, "columns": columns}, df)
    assert list(ranking.columns) == ["feature", "importance"]
    assert ranking["importance"].is_monotonic_decreasing
