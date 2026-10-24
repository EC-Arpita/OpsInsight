"""The saved models are good enough and handle new values safely. Needs `python -m opsinsight.models.train` first."""
import json

import pandas as pd
import pytest

from opsinsight.models.config import DOMAINS, MODELS_DIR
from opsinsight.models.predict import predict


@pytest.mark.parametrize("domain", ["incidents", "sales"])
def test_model_reaches_target(domain):
    m = json.loads((MODELS_DIR / f"{domain}_meta.json").read_text())["metrics"]
    assert m["test_auc"] >= 0.70  # the goal in the Test Plan (R-01, R-02)
    assert m["lift_top_10pct"] > 1.5  # the top 10% must find clearly more problems than random


@pytest.mark.parametrize("domain", ["incidents", "sales"])
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