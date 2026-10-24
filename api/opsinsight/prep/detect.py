"""Domain detection: look at an uploaded file and work out what kind of data it is.

    from opsinsight.prep.detect import inspect_upload
    report = inspect_upload(df)
    report["domain"], report["decision"], report["shape"], report["message"]
"""
import pandas as pd

from .mapping import SCHEMAS, suggest_mapping

CONFIDENT = 0.80      # best score at least this ...
MARGIN = 0.20         # ... and this far ahead of the runner-up -> decide automatically
POSSIBLE = 0.50       # between POSSIBLE and CONFIDENT -> ask the analyst


def detect_domain(columns, df: pd.DataFrame = None) -> dict:
    """Score every known data type by how many of its REQUIRED columns we can find, and how surely."""
    results = []
    for domain, schema in SCHEMAS.items():
        s = suggest_mapping(list(columns), domain, df)
        required = [name for name, (req, _, _) in schema.items() if req]
        found = [name for name in required if name not in s["missing_required"]]     # a days column counts too
        confidence = {x["standard"]: x["score"] for x in s["suggestions"] if x["standard"] in found and x["score"]}
        coverage = len(found) / len(required)                        # share of required columns found
        avg_conf = sum(confidence.values()) / len(found) if found else 0   # how sure each match is
        results.append({"domain": domain, "score": round(coverage * avg_conf, 2),
                        "required_found": f"{len(found)} of {len(required)}",
                        "missing_required": [r for r in required if r not in found]})
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    runner_up = results[1]["score"] if len(results) > 1 else 0
    if best["score"] >= CONFIDENT and best["score"] - runner_up >= MARGIN:
        decision = "confident"
    elif best["score"] >= POSSIBLE:
        decision = "ask_analyst"
    else:
        decision = "unknown"
    return {"domain": best["domain"] if decision != "unknown" else None, "decision": decision,
            "best_score": best["score"], "scores": results}


def detect_shape(df: pd.DataFrame, id_col: str) -> dict:
    """Is this one row per item, or an event log with several rows (updates) per item?"""
    n_rows, n_ids = len(df), df[id_col].nunique()
    per_item = n_rows / n_ids if n_ids else 0
    shape = "event_log" if per_item > 1.05 else "one_row_per_item"
    return {"shape": shape, "rows": n_rows, "unique_items": n_ids, "rows_per_item": round(per_item, 2)}


def inspect_upload(df: pd.DataFrame) -> dict:
    """Everything the upload screen needs to show the analyst before anything is processed."""
    d = detect_domain(df.columns, df)
    report = {**d, "mapping": None, "shape": None, "message": ""}
    if d["decision"] == "unknown":
        report["message"] = ("Couldn't recognise this file. It doesn't look like support tickets or sales orders. "
                             "Check that it has an ID, a date and an outcome column.")
        return report
    m = suggest_mapping(list(df.columns), d["domain"], df)
    report["mapping"] = m
    id_col = m["mapping"].get("number") or m["mapping"].get("order_id")
    if id_col:
        report["shape"] = detect_shape(df, id_col)
    names = {"incidents": "support tickets", "sales_orders": "sales orders"}
    how = "Looks like" if d["decision"] == "confident" else "This might be"
    shape_text = ""
    if report["shape"]:
        s = report["shape"]
        shape_text = (f" It's an event log: {s['rows']:,} rows for {s['unique_items']:,} items "
                      f"(about {s['rows_per_item']} updates each), so it will be collapsed to one row per item."
                      if s["shape"] == "event_log" else f" One row per item ({s['unique_items']:,} items).")
    report["message"] = f"{how} {names[d['domain']]} ({d['best_score']:.0%} match). Please confirm.{shape_text}"
    return report