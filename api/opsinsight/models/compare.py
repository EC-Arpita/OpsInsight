"""Compare models and settings fairly, and pick the best one.

Run from the api folder:   python -m opsinsight.models.compare

Fair = we pick using a VALIDATION period (the newest 20% of the training rows),
and only then look at the TEST period once, to report the result.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DOMAINS, LGBM_PARAMS
from .train import split_xy

BASE = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
CANDIDATES = {
    "LightGBM, default settings": dict(random_state=42, verbose=-1),
    "LightGBM, cautious (current)": LGBM_PARAMS,
    "LightGBM, bigger trees": dict(BASE, n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, reg_lambda=1),
    "LightGBM, slower + more trees": dict(BASE, n_estimators=600, learning_rate=0.015, num_leaves=15, min_child_samples=300, reg_lambda=5),
    "LightGBM, very small trees": dict(BASE, n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=400, reg_lambda=10),
}


def logistic_model(X):
    """A simple, classic model: numbers scaled, categories one-hot encoded, then logistic regression."""
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in X.columns if c not in cats]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=2000, C=0.5))


def top10_precision(y, p):
    top = p >= np.quantile(p, 0.9)
    return float(y[top].mean())


def compare(domain: str) -> pd.DataFrame:
    cfg = DOMAINS[domain]
    df = pd.read_parquet(cfg["features_file"])
    X, y = split_xy(df, cfg)
    train_idx = np.where(df["split"] == "train")[0]
    test_idx = np.where(df["split"] == "test")[0]
    # validation = newest 20% of the training rows (rows are already in time order)
    cut = int(len(train_idx) * 0.8)
    fit_idx, val_idx = train_idx[:cut], train_idx[cut:]

    models = {"Logistic regression": lambda: logistic_model(X)}
    models.update({name: (lambda p=p: lgb.LGBMClassifier(**p)) for name, p in CANDIDATES.items()})

    rows = []
    for name, make in models.items():
        m = make().fit(X.iloc[fit_idx], y.iloc[fit_idx])                       # 1) choose on validation
        val_auc = roc_auc_score(y.iloc[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1])
        m = make().fit(X.iloc[train_idx], y.iloc[train_idx])                   # 2) refit on all training rows
        p = m.predict_proba(X.iloc[test_idx])[:, 1]                            # 3) report on test
        yt = y.iloc[test_idx].values
        rows.append({"model": name, "validation_auc": round(val_auc, 3),
                     "test_auc": round(roc_auc_score(yt, p), 3),
                     "test_top10_precision": round(top10_precision(yt, p), 3)})
    out = pd.DataFrame(rows).sort_values("validation_auc", ascending=False).reset_index(drop=True)
    out["picked"] = ["yes"] + [""] * (len(out) - 1)
    return out


def main():
    for d in DOMAINS:
        print(f"\n=== {d} ===")
        print(compare(d).to_string(index=False))


if __name__ == "__main__":
    main()
