"""Pick the right model for an uploaded file, using Arpita's domain detection (prep/detect.py).

    from opsinsight.models.route import choose_model
    choice = choose_model(df)
    choice["status"], choice["model"], choice["message"]
"""
from ..prep.detect import inspect_upload

# Arpita's detection names the data type; our model files are named differently
DOMAIN_TO_MODEL = {"incidents": "incidents", "sales_orders": "sales"}


def choose_model(df, analyst_confirmed=False) -> dict:
    """Which saved model (models/<name>.joblib) should score this file? Never guesses."""
    r = inspect_upload(df)
    base = {"model": None, "message": r["message"], "detection_score": r["best_score"]}
    if r["decision"] == "unknown":
        return {**base, "status": "refused"}
    missing = r["mapping"]["missing_required"]
    if missing:
        return {**base, "status": "cannot_run", "missing_required": missing,
                "message": r["message"] + f" Missing required columns: {', '.join(missing)}."}
    if r["decision"] == "ask_analyst" and not analyst_confirmed:
        return {**base, "status": "needs_confirmation"}
    return {**base, "status": "ok", "model": DOMAIN_TO_MODEL[r["domain"]],
            "mapping": r["mapping"]["mapping"], "shape": r["shape"]}
