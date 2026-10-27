"""The saved models are good enough and handle new values safely. Needs `python -m opsinsight.models.train` first."""
import json

import pandas as pd
import pytest

from opsinsight.models.config import DOMAINS, MODELS_DIR
from opsinsight.models.predict import predict


@pytest.mark.parametrize("domain", ["incidents", "sales", "sales_pickup"])
def test_model_reaches_target(domain):
    m = json.loads((MODELS_DIR / f"{domain}_meta.json").read_text())["metrics"]
    assert m["test_auc"] >= 0.70  # the goal in the Test Plan (R-01, R-02)
    assert m["lift_top_10pct"] > 1.5  # the top 10% must find clearly more problems than random


@pytest.mark.parametrize("domain", ["incidents", "sales", "sales_pickup"])
def test_prediction_has_risk_and_three_reasons(domain):
    rows = pd.read_parquet(DOMAINS[domain]["features_file"]).head(5)
    out = predict(domain, rows)
    assert len(out) == 5
    for p in out:
        assert 0 <= p["prediction"]["risk"] <= 1
        assert p["prediction"]["risk_level"] in ("high", "medium", "low")
        assert len(p["top_reasons"]) == 3


def test_unknown_team_does_not_crash():
    rows = pd.read_parquet(DOMAINS["incidents"]["features_file"]).head(1).copy()
    rows["assignment_group_cat"] = "A team we have never seen"
    assert 0 <= predict("incidents", rows)[0]["prediction"]["risk"] <= 1

def test_recency_weights_halve_every_half_life():
    from opsinsight.models.train import recency_weights
    times = pd.Series(pd.to_datetime(["2016-01-01", "2016-01-31", "2016-03-01", None]))
    w = recency_weights(times, 30)
    assert w[2] == 1.0 and abs(w[1] - 0.5 ** (30 / 30)) < 1e-9 and abs(w[0] - 0.5 ** (60 / 30)) < 1e-9
    assert w[3] == 1.0                                   # a missing date is not dropped
    assert recency_weights(times, None) is None          # no half-life: every row counts the same


def test_pickup_model_uses_only_what_is_known_at_pickup():
    """Stage 2 may use dispatch facts (known once the courier has the order) but nothing from the delivery."""
    m = json.loads((MODELS_DIR / "sales_pickup_meta.json").read_text())
    assert {"dispatch_days", "days_left_at_pickup", "handed_over_late"} <= set(m["features"])
    assert not {"delivered_at", "days_late", "review_score", "carrier_at", "estimated_at"} & set(m["features"])
    assert m["metrics"]["test_auc"] >= 0.80                 # clearly better than predicting at approval
