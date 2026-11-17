"""Simulate retention campaign ROI from churn probabilities."""


def campaign_roi(probabilities, clv, contact_cost, success_rate, n_contacted=None):
    at_risk = sorted(probabilities, reverse=True)
    if n_contacted is not None:
        at_risk = at_risk[:n_contacted]
    saved_clv = sum(p * clv * success_rate for p in at_risk)
    cost = len(at_risk) * contact_cost
    return {
        "contacted": len(at_risk),
        "clv_saved": round(saved_clv, 2),
        "cost": round(cost, 2),
        "net_roi": round(saved_clv - cost, 2),
    }
