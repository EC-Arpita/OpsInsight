"""Train the risk models and save them to models/<domain>.joblib.

Run from the api folder:   python -m opsinsight.models.train
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from .config import DOMAINS, LGBM_PARAMS, MODELS_DIR


def split_xy(df, cfg):
    """X = feature columns only, y = the target."""
    X = df.drop(columns=cfg["non_feature_cols"])
    return X, df[cfg["target"]]


def train_domain(domain: str, params: dict = None, save=True) -> dict:
    """Train on the 'train' split, evaluate on 'test', save model + metadata."""
    cfg = DOMAINS[domain]
    df = pd.read_parquet(cfg["features_file"])
    X, y = split_xy(df, cfg)
    is_train = (df["split"] == "train").values

    model = lgb.LGBMClassifier(**(params or LGBM_PARAMS))
    model.fit(X[is_train], y[is_train])                       # learn from the past only

    p_test = model.predict_proba(X[~is_train])[:, 1]          # risk for each test row
    p_train = model.predict_proba(X[is_train])[:, 1]
    y_test = y[~is_train].values
    hi, med = np.quantile(p_test, 0.90), np.quantile(p_test, 0.60)
    top10 = p_test >= hi

    meta = {
        "domain": domain,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "algorithm": "LightGBM (gradient boosted trees)",
        "params": params or LGBM_PARAMS,
        "features": list(X.columns),
        "categories": {c: [str(v) for v in X[c].cat.categories] for c in X.columns
                       if isinstance(X[c].dtype, pd.CategoricalDtype)},
        "typical_values": {c: (str(X.loc[is_train, c].mode().iloc[0]) if isinstance(X[c].dtype, pd.CategoricalDtype)
                               else float(X.loc[is_train, c].median())) for c in X.columns},
        "risk_thresholds": {"high": float(hi), "medium": float(med),
                            "rule": "high = top 10% of test-period risk, medium = top 40%"},
        "test_risk_distribution": np.sort(p_test).round(5).tolist()[::max(1, len(p_test) // 1000)],
        "metrics": {
            "test_auc": round(float(roc_auc_score(y_test, p_test)), 3),
            "train_auc": round(float(roc_auc_score(y[is_train], p_train)), 3),
            "base_rate_test": round(float(y_test.mean()), 3),
            "precision_top_10pct": round(float(y_test[top10].mean()), 3),
            "lift_top_10pct": round(float(y_test[top10].mean() / y_test.mean()), 2),
            "train_rows": int(is_train.sum()), "test_rows": int((~is_train).sum()),
        },
    }
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": model, "meta": meta}, MODELS_DIR / f"{domain}.joblib")
        (MODELS_DIR / f"{domain}_meta.json").write_text(
            json.dumps({k: v for k, v in meta.items() if k != "test_risk_distribution"}, indent=2))
    return meta


def main():
    for d in DOMAINS:
        meta = train_domain(d)
        print(f"{d:10s} test AUC {meta['metrics']['test_auc']}  "
              f"top-10% precision {meta['metrics']['precision_top_10pct']}  -> models/{d}.joblib")


if __name__ == "__main__":
    main()
