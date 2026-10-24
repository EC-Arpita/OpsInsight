"""Column mapping: make another company's file fit the pipeline.

Flow:  suggest_mapping(columns)  ->  analyst checks / overrides  ->  apply_mapping(df, mapping)
"""
import re
from difflib import SequenceMatcher

import pandas as pd

# name -> (required?, other names people use, value to fill when the column is missing)
SCHEMAS = {
    "incidents": {
        "number": (True, ["ticket_id", "ticket_no", "ticket_number", "incident_id", "incident_number", "case_id", "id"], None),
        "opened_at": (True, ["created_at", "created_date", "open_date", "opened_date", "created", "creation_time", "reported_at"], None),
        "sys_updated_at": (False, ["updated_at", "updated", "last_updated", "modified_at", "event_time", "timestamp"], None),
        "sys_mod_count": (False, ["update_number", "event_number", "update_count", "version", "mod_count", "sequence"], 0),
        "made_sla": (True, ["sla_met", "within_sla", "sla_achieved", "met_sla", "sla_ok", "sla_status", "sla"], None),
        "assignment_group": (True, ["team", "assigned_team", "support_group", "queue", "group", "resolver_group"], None),
        "priority": (True, ["priority_level", "prio", "severity_priority"], None),
        "category": (True, ["ticket_category", "issue_category", "type", "issue_type"], None),
        "resolved_at": (True, ["resolved_date", "resolution_date", "resolved", "fixed_at", "solved_at"], None),
        "closed_at": (False, ["closed_date", "close_date", "closed", "completed_at"], None),
        "incident_state": (False, ["state", "status", "ticket_status", "stage"], "Unknown"),
        "subcategory": (False, ["sub_category", "subcat", "ticket_subcategory"], None),
        "u_symptom": (False, ["symptom", "issue", "problem_description"], None),
        "impact": (False, ["business_impact", "impact_level"], None),
        "urgency": (False, ["urgency_level"], None),
        "contact_type": (False, ["channel", "source", "contact_channel", "reported_via"], "Unknown"),
        "location": (False, ["site", "city", "region", "office"], None),
        "caller_id": (False, ["caller", "customer_id", "requester", "user_id", "reported_by"], None),
        "opened_by": (False, ["created_by", "agent_opened", "logged_by"], None),
        "reassignment_count": (False, ["reassignments", "times_reassigned", "transfer_count", "hops"], 0),
        "reopen_count": (False, ["reopens", "times_reopened", "reopened_count"], 0),
        "knowledge": (False, ["kb_used", "knowledge_used", "kb_article"], False),
        "u_priority_confirmation": (False, ["priority_confirmed"], False),
        "notify": (False, ["notification"], "Do Not Notify"),
        "closed_code": (False, ["resolution_code", "close_code", "closure_code"], None),
        "sys_created_at": (False, ["system_created_at"], None),
        "caused_by": (False, [], None), "vendor": (False, [], None), "cmdb_ci": (False, ["configuration_item", "ci"], None),
        "rfc": (False, ["change_request"], None), "problem_id": (False, ["problem"], None),
    },
    "sales_orders": {
        "order_id": (True, ["id", "order_no", "order_number"], None),
        "customer_id": (True, ["customer", "client_id", "buyer_id"], None),
        "order_status": (True, ["status", "state"], None),
        "order_purchase_timestamp": (True, ["purchase_date", "order_date", "ordered_at", "created_at", "purchased_at"], None),
        "order_approved_at": (False, ["approved_at", "approval_date", "paid_at", "payment_date"], None),
        "order_delivered_carrier_date": (False, ["shipped_at", "ship_date", "dispatch_date", "handed_to_carrier"], None),
        "order_delivered_customer_date": (True, ["delivered_at", "delivery_date", "received_at"], None),
        "order_estimated_delivery_date": (True, ["promised_date", "estimated_delivery", "eta", "expected_delivery"], None),
        "customer_state": (False, ["region", "state", "customer_region", "order_region", "delivery_region", "zone"], None),
    },
}
MIN_SCORE = 0.6     # below this, no automatic match: the analyst picks by hand


def normalise(name: str) -> str:
    """'Ticket No.' -> 'ticket_no', 'createdDate' -> 'created_date'"""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _score(column: str, standard: str, synonyms: list) -> tuple:
    c = normalise(column)
    if c == standard:
        return 1.0, "exact name"
    if c in synonyms:
        return 0.95, "known synonym"
    best = max([standard] + synonyms, key=lambda s: SequenceMatcher(None, c, s).ratio())
    return round(SequenceMatcher(None, c, best).ratio(), 2), f"similar to '{best}'"


def suggest_mapping(columns, domain="incidents") -> dict:
    """For each standard column, the best-matching uploaded column (each uploaded column used once)."""
    schema = SCHEMAS[domain]
    candidates = []
    for col in columns:
        for std, (required, syn, _) in schema.items():
            score, why = _score(col, std, syn)
            if score >= MIN_SCORE:
                candidates.append((score, col, std, why))
    mapping, used, reasons = {}, set(), {}
    for score, col, std, why in sorted(candidates, key=lambda x: -x[0]):   # best matches first
        if std not in mapping and col not in used:
            mapping[std], reasons[std] = col, (score, why)
            used.add(col)
    suggestions = []
    for std, (required, _, default) in schema.items():
        score, why = reasons.get(std, (None, "no match"))
        suggestions.append({"standard": std, "required": required, "uploaded_column": mapping.get(std),
                            "score": score, "reason": why,
                            "needs_review": std not in mapping or score < 0.95,
                            "if_missing": "cannot run" if required else f"filled with {default!r}"})
    missing_required = [s["standard"] for s in suggestions if s["required"] and s["uploaded_column"] is None]
    return {"mapping": mapping, "suggestions": suggestions, "missing_required": missing_required,
            "unused_columns": [c for c in columns if c not in used], "ready": not missing_required}


def apply_mapping(df: pd.DataFrame, mapping: dict, domain="incidents") -> pd.DataFrame:
    """Rename uploaded columns to standard names and add missing optional columns with defaults."""
    schema = SCHEMAS[domain]
    missing = [s for s, (req, _, _) in schema.items() if req and s not in mapping]
    if missing:
        raise ValueError(f"required columns not mapped: {missing}")
    out = df.rename(columns={src: std for std, src in mapping.items()})
    keep = [s for s in schema if s in out.columns]
    out = out[keep].copy()
    for std, (req, _, default) in schema.items():
        if std not in out.columns:
            out[std] = default
    return out