"""The `predictions` and `model_performance` sections for an uploaded file, in the dashboard's format."""
import pandas as pd

from ..models.predict import model_info, predict

ACTIONS = {"incidents": "Fast-track: senior agent picks it up within 1 hour",
           "sales": "Contact seller to ship within 24h and use express carrier",
           "sales_pickup": "Already with the courier: upgrade to express delivery and tell the customer the new date"}


def _ts(v):
    return None if pd.isna(v) else f"{pd.Timestamp(v):%Y-%m-%dT%H:%M}"


def _v(v):
    return None if pd.isna(v) else v                       # empty value -> null (JSON has no NaN)


def _context(domain, row):
    if domain == "incidents":
        return {"opened_at": _ts(row["opened_at"]), "team": row["assignment_group"], "category": row["category"],
                "subcategory": row["subcategory"], "priority": None if pd.isna(row["priority_n"]) else int(row["priority_n"]),
                "location": row["location"], "contact_type": row["contact_type"]}
    return {"approved_at": _ts(row["approved_at"]), "promised_by": _ts(row["estimated_at"])[:10] if _ts(row["estimated_at"]) else None,
            "customer_state": _v(row["customer_state"]), "customer_city": _v(row["customer_city"]),
            "seller_id": _v(row["main_seller_id"]), "seller_state": _v(row["seller_state"]), "category": _v(row["main_category"]),
            "order_value": None if pd.isna(row["payment_total"]) else round(float(row["payment_total"]), 2),
            "n_items": None if pd.isna(row["n_items"]) else int(row["n_items"]),         # orders-only files have neither
            "payment_type": _v(row["main_payment_type"])}


def _similar(domain, row, history):
    if domain == "incidents":
        same = history[(history["category"] == row["category"]) & (history["assignment_group"] == row["assignment_group"])]
        return {"rule": "same category and team, training period", "count": int(len(same)),
                "breach_rate": round(float(same["breached"].mean()), 3) if len(same) else None,
                "median_resolution_hours": round(float(same["hours_to_resolve"].median()), 1) if len(same) else None}
    if pd.isna(row["seller_state"]):                       # orders-only files have no seller: compare by customer state
        same, rule = history[history["customer_state"] == row["customer_state"]], "same customer state, training period"
    else:
        same = history[(history["seller_state"] == row["seller_state"]) & (history["customer_state"] == row["customer_state"])]
        rule = "same seller state → customer state route, training period"
    days = (same["delivered_at"] - same["purchased_at"]).dt.total_seconds() / 86400
    return {"rule": rule, "count": int(len(same)),
            "late_rate": round(float(same["late"].mean()), 3) if len(same) else None,
            "avg_delivery_days": round(float(days.mean()), 1) if len(same) else None}


def predictions_section(domain: str, table: pd.DataFrame, features: pd.DataFrame, top: int = 10, model: str = None) -> dict:
    """Score the current period (the 'test' rows) and keep the riskiest items, with context and reasons."""
    id_col, target = ("number", "breached") if domain == "incidents" else ("order_id", "late")
    current = features[features["split"] == "test"]
    scored = pd.DataFrame(predict(domain, current, model=model))
    scored["risk"] = scored["prediction"].map(lambda p: p["risk"])
    best = scored.sort_values("risk", ascending=False).head(top)
    rows = table.set_index(id_col)
    rows.index = rows.index.astype(str)                    # predictions use text ids; some files have numbers
    train_ids = set(features.loc[features["split"] == "train", id_col])
    history = table[table[id_col].isin(train_ids)]
    if domain != "incidents":
        history = history[history["is_model_row"] == 1]
    meta = model_info(domain, model)
    items = []
    for _, p in best.iterrows():
        row = rows.loc[p["id"]]
        items.append({"id": p["id"], "item_type": p["item_type"], "context": _context(domain, row),
                      "prediction": p["prediction"], "top_reasons": p["top_reasons"],
                      "explanation_summary": p["explanation_summary"], "similar_past_cases": _similar(domain, row, history),
                      "recommended_action": {"recommendation_id": "REC3", "action": ACTIONS[domain],
                                             "urgency": "now" if p["prediction"]["risk_level"] == "high" else "today"},
                      "actual_outcome": {target: bool(row[target]),
                                         "note": "known only because this is historical data; hidden in a live system"}})
    hi, med = meta["risk_thresholds"]["high"], meta["risk_thresholds"]["medium"]
    return {"count_shown": len(items), "sorted_by": "risk",
            "risk_level_rules": {"high": f"risk ≥ {hi:.3f} (top 10%)", "medium": f"risk ≥ {med:.3f} (top 40%)", "low": "below medium"},
            "items": items}


def reference_fit(domain: str, features: pd.DataFrame) -> dict:
    """How well the reference model (trained on another company) fits this company's own labelled rows."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from ..models.config import DOMAINS
    from ..models.predict import _align, load
    bundle = load(domain)
    X, y = _align(features, bundle["meta"]), features[DOMAINS[domain]["target"]].astype(int).to_numpy()
    p = bundle["model"].predict_proba(X)[:, 1]
    if bundle.get("lr") is not None:
        w = bundle.get("w", 0.5)
        p = w * p + (1 - w) * bundle["lr"].predict_proba(X)[:, 1]
    cats = bundle["meta"]["categories"]
    known = float(np.mean([features[c].astype(str).isin(set(v) - {"Other"}).mean() for c, v in cats.items()]))
    out = {"rows": len(y), "problems": int(y.sum()), "known_values_share": round(known, 2), "auc": None, "lift_top_10pct": None}
    if 0 < y.sum() < len(y):
        top = p >= np.quantile(p, 0.9)
        out.update(auc=round(float(roc_auc_score(y, p)), 3), lift_top_10pct=round(float(y[top].mean() / y.mean()), 2))
    return out


def model_section(domain: str, model: str = None, features: pd.DataFrame = None) -> dict:
    meta = model_info(domain, model)
    extra = {}
    if model is None and features is not None:       # a company too small for its own model: measure the borrowed one
        fit = reference_fit(domain, features)
        auc = "not measurable" if fit["auc"] is None else f"AUC {fit['auc']}"
        extra = {"on_your_data": fit, "warning": (
            f"Your file has too little history for its own model ({fit['rows']} items, {fit['problems']} problems; "
            f"it needs at least 100 older items to learn from and 10 problems). Risk scores come from a reference model trained on another company's data; "
            f"on your data it ranks with {auc}. Only {fit['known_values_share']:.0%} of your team and category values "
            "are known to it. Treat risk levels as rough.")}
    return {**extra, "algorithm": meta["algorithm"], **meta["metrics"], "n_features": len(meta["features"]),
            "validation": "time-based split: trained on older items, tested on the newest 20%",
            "trained_on": "this company's own history" if model else "the reference dataset",
            "note": ("trained on this file's older items and tested on its newest 20%" if model else
                     "the saved model was trained on the reference dataset; retrain on this company's data for its own scores")}


def pickup_section(orders: pd.DataFrame, pickup: pd.DataFrame, model: str = None, uploaded=True) -> dict | None:
    """Stage 2 for sales: orders already with the courier, scored again with the pickup model (dispatch facts known).
    None when the file has no pickup (ship) dates in its current period. uploaded=False for the reference data itself."""
    if pickup is None or not (pickup["split"] == "test").any():
        return None
    out = predictions_section("sales_pickup", orders, pickup, model=model)
    out["model_performance"] = model_section("sales_pickup", model, pickup if (uploaded and model is None) else None)
    out["prediction_moment"] = "when the carrier picks the order up"
    return out
