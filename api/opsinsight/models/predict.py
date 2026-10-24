"""Load a saved model and score rows: risk + top reasons, in the contract format.

    from opsinsight.models.predict import predict
    results = predict("incidents", features_df)     # list of dicts, one per row
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import DOMAINS, LABELS, MODELS_DIR, UPLOAD_MODELS_DIR


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """name = a domain ("incidents", "sales": the reference models) or an upload id (that company's own model)."""
    for folder in (MODELS_DIR, UPLOAD_MODELS_DIR):
        if (folder / f"{name}.joblib").exists():
            return joblib.load(folder / f"{name}.joblib")
    raise FileNotFoundError(f"{name}.joblib not found - run `python -m opsinsight.models.train` first")


def _align(rows: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Same columns, order and category values as in training; unseen categories -> 'Other'."""
    missing = [c for c in meta["features"] if c not in rows.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = rows[meta["features"]].copy()
    for c, cats in meta["categories"].items():
        vals = X[c].astype(str)
        fallback = "Other" if "Other" in cats else cats[0]
        X[c] = pd.Categorical(vals.where(vals.isin(cats), fallback), categories=cats)
    return X


def _fmt(v):
    """1149.0 -> '1,149'   10.3245 -> '10.32'   'RJ' -> 'RJ'"""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def predict(domain: str, rows: pd.DataFrame, top_k: int = 3, model: str = None) -> list:
    """Score each row. `rows` must contain the feature columns (and ideally the id column).
    model = an upload id to use that company's own model; None = the reference model."""
    bundle = load(model or domain)
    model, meta, cfg = bundle["model"], bundle["meta"], DOMAINS[domain]
    labels = LABELS.get(domain, {})
    X = _align(rows, meta)
    risk = model.predict_proba(X)[:, 1]
    if bundle.get("lr") is not None:                          # blend (Step 5): average with logistic regression
        risk = (risk + bundle["lr"].predict_proba(X)[:, 1]) / 2

    contrib = model.predict(X, pred_contrib=True)[:, :-1]      # SHAP values (log-odds); last column = bias
    dist = np.array(meta["test_risk_distribution"])
    hi, med = meta["risk_thresholds"]["high"], meta["risk_thresholds"]["medium"]
    ids = rows[cfg["id_col"]].astype(str).tolist() if cfg["id_col"] in rows else [str(i) for i in rows.index]

    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(contrib[i]))[:top_k]          # biggest pushes first
        reasons = []
        for j in order:
            f = meta["features"][j]
            is_cat = f in meta["categories"]
            direction = "increases risk" if contrib[i, j] > 0 else "decreases risk"
            value, typical = X.iloc[i, j], meta["typical_values"][f]
            label = labels.get(f, f)
            reasons.append({
                "feature": f, "label": label,
                "value": str(value) if is_cat else (None if pd.isna(value) else round(float(value), 2)),
                "typical_value": typical if is_cat else round(float(typical), 2),
                "effect": round(float(contrib[i, j]), 3), "direction": direction,
                "sentence": f"{label}: {_fmt(value)} ({'most common' if is_cat else 'usual'}: {_fmt(typical)}), {direction}",
            })
        level = "high" if risk[i] >= hi else "medium" if risk[i] >= med else "low"
        ups = [r["sentence"] for r in reasons if r["direction"] == "increases risk"]
        summary = (f"{level.capitalize()} risk ({risk[i]:.0%}) of {cfg['target_text']}"
                   + (", mainly because: " + "; ".join(ups) + "." if ups else "."))
        out.append({
            "id": ids[i], "item_type": cfg["item_type"],
            "prediction": {"target": cfg["target_name"], "risk": round(float(risk[i]), 3), "risk_level": level,
                           "risk_percentile": round(float((dist < risk[i]).mean() * 100), 1),
                           "prediction_moment": cfg["prediction_moment"]},
            "top_reasons": reasons,
            "explanation_summary": summary,
        })
    return out


def model_info(domain: str, model: str = None) -> dict:
    meta = load(model or domain)["meta"]
    return {k: v for k, v in meta.items() if k not in ("test_risk_distribution", "typical_values", "categories")}
