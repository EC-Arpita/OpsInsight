"""Column mapping: make another company's file fit the pipeline.

Flow:  suggest_mapping(columns)  ->  analyst checks / overrides  ->  apply_mapping(df, mapping)

    from opsinsight.prep.mapping import suggest_mapping, apply_mapping
    s = suggest_mapping(df.columns, "incidents")
    s["suggestions"]                     # what the app shows the analyst to confirm
    clean = apply_mapping(df, s["mapping"], "incidents")
"""
import re
from difflib import SequenceMatcher

import pandas as pd

# name -> (required?, other names people use, value to fill when the column is missing)
SCHEMAS = {
    "incidents": {
        "number": (True, ["ticket_id", "ticket_no", "ticket_number", "incident_id", "incident_number", "case_id", "id",
                           "issue_key", "issue_id", "key", "ticket", "case_number", "request_id", "reference", "ticket_nr"], None),
        "opened_at": (True, ["created_at", "created_date", "open_date", "opened_date", "created", "creation_time", "reported_at",
                              "created_time", "opened", "open_time", "date_opened", "submitted_at", "logged_at"], None),
        "sys_updated_at": (False, ["updated_at", "updated", "last_updated", "modified_at", "event_time", "timestamp"], None),
        "sys_mod_count": (False, ["update_number", "event_number", "update_count", "version", "mod_count", "sequence"], 0),
        "made_sla": (True, ["sla_met", "within_sla", "sla_achieved", "met_sla", "sla_ok", "sla_status", "sla",
                             "sla_breached", "sla_breach", "sla_violated", "sla_missed", "breached"], None),
        "assignment_group": (True, ["team", "assigned_team", "support_group", "queue", "group", "resolver_group",
                                     "assignment_group_name", "department", "assigned_group"], "All teams"),
        "priority": (True, ["priority_level", "prio", "severity_priority", "severity", "sev"], "3 - Moderate"),
        "category": (True, ["ticket_category", "issue_category", "type", "issue_type", "ticket_type", "request_type"], "Unknown"),
        "resolved_at": (True, ["resolved_date", "resolution_date", "resolved", "fixed_at", "solved_at", "resolved_time",
                                "resolution_time", "date_resolved"], None),
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
        "order_id": (True, ["id", "order_no", "order_number", "order", "order_ref", "invoice_no", "invoice_id"], None),
        "customer_id": (True, ["customer", "client_id", "buyer_id", "customer_no", "customer_number", "client"], None),
        "order_status": (True, ["status", "state", "delivery_status_text"], "delivered"),
        "order_purchase_timestamp": (True, ["purchase_date", "order_date", "ordered_at", "created_at", "purchased_at",
                                             "order_time", "order_datetime", "date_ordered", "invoice_date"], None),
        "order_approved_at": (False, ["approved_at", "approval_date", "paid_at", "payment_date"], None),
        "order_delivered_carrier_date": (False, ["shipped_at", "ship_date", "dispatch_date", "handed_to_carrier"], None),
        "order_delivered_customer_date": (True, ["delivered_at", "delivery_date", "received_at", "delivered_on", "delivered_date",
                                                  "actual_delivery_date", "date_delivered"], None),
        "order_estimated_delivery_date": (True, ["promised_date", "estimated_delivery", "eta", "expected_delivery",
                                                  "expected_delivery_date", "promised_delivery_date", "due_date",
                                                  "estimated_delivery_date", "delivery_due"], None),
        "customer_state": (False, ["region", "state", "customer_region", "order_region", "delivery_region", "zone"], None),
        "order_value": (False, ["sales", "amount", "order_total", "total", "revenue", "order_amount", "value", "total_amount",
                                "payment_value", "price", "grand_total", "net_sales"], None),
        "promised_days": (False, ["days_for_shipment_scheduled", "scheduled_days", "days_promised", "sla_days",
                                  "promised_delivery_days", "days_scheduled"], None),
        "actual_days": (False, ["days_for_shipping_real", "delivery_days", "days_taken", "actual_delivery_days",
                                "days_to_deliver", "shipping_days"], None),
    },
}
MIN_SCORE = 0.6     # below this, no automatic match: the analyst picks by hand

# date fields may only match columns that really hold dates, day counts only columns that hold numbers
DATE_FIELDS = {"opened_at", "sys_updated_at", "resolved_at", "closed_at", "sys_created_at", "order_purchase_timestamp",
               "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"}
DAY_FIELDS = {"promised_days", "actual_days"}
MONEY_FIELDS = {"order_value"}
TEXT_FIELDS = {"order_status", "incident_state", "customer_state", "assignment_group", "category"}   # never a money column
# a required date can instead come from a number of days after the order date
ALTERNATIVES = {"order_delivered_customer_date": "actual_days", "order_estimated_delivery_date": "promised_days"}
# when a column is missing, the nearest one stands in: closed for resolved, shipped for delivered
FALLBACKS = {"resolved_at": "closed_at", "order_delivered_customer_date": "order_delivered_carrier_date"}
# what the app truly cannot run without; other 'required' columns help recognise the file and get a default.
# SLA met / promised date can also come from rules typed on the upload screen.
NEEDED = {"incidents": ["number", "opened_at", "resolved_at"],
          "sales_orders": ["order_id", "order_purchase_timestamp", "order_delivered_customer_date"]}
PLAIN = {"number": "a ticket ID", "opened_at": "when the ticket was opened", "resolved_at": "when it was resolved or closed",
         "order_id": "an order ID", "order_purchase_timestamp": "the order date",
         "order_delivered_customer_date": "the delivery (or ship) date"}


def cannot_run_without(mapping: dict, domain: str) -> list:
    return [s for s in NEEDED[domain] if s not in mapping and ALTERNATIVES.get(s) not in mapping]


def check_needed(mapping: dict, domain: str):
    """Stop early, in plain words, when a column the app cannot do without is not mapped."""
    missing = cannot_run_without(mapping, domain)
    if missing:
        raise ValueError("Your file needs " + ", ".join(PLAIN[m] for m in missing) +
                         ". Pick the matching column in step 2 (Check what we found).")


def _holds_dates(values: pd.Series) -> bool:
    v = values.dropna().head(200)
    if v.empty or pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
        return False
    return pd.to_datetime(v.astype(str), errors="coerce", format="mixed").notna().mean() > 0.8


def _fits(std: str, values: pd.Series) -> bool:
    if std in DATE_FIELDS:
        return _holds_dates(values)
    if std in DAY_FIELDS:
        return pd.api.types.is_numeric_dtype(values)
    if std in MONEY_FIELDS:
        return pd.api.types.is_numeric_dtype(values)
    if std in TEXT_FIELDS:
        return not pd.api.types.is_float_dtype(values)
    return True


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


def suggest_mapping(columns, domain="incidents", df: pd.DataFrame = None) -> dict:
    """For each standard column, the best-matching uploaded column (each uploaded column used once).
    With df, the values are checked too: a date field never matches a text column like 'Order State'."""
    schema = SCHEMAS[domain]
    candidates = []
    for col in columns:
        for std, (required, syn, _) in schema.items():
            score, why = _score(col, std, syn)
            if score >= MIN_SCORE and (df is None or _fits(std, df[col])):
                candidates.append((score, col, std, why))
    mapping, used, reasons = {}, set(), {}
    for score, col, std, why in sorted(candidates, key=lambda x: -x[0]):   # best matches first
        if std not in mapping and col not in used:
            mapping[std], reasons[std] = col, (score, why)
            used.add(col)
    for std, other in FALLBACKS.items():
        if std in schema and std not in mapping and ALTERNATIVES.get(std) not in mapping and other in mapping:
            mapping[std] = mapping.pop(other)
            reasons[std] = (reasons.pop(other)[0], f"no {std} column: using {mapping[std]!r} instead")
    suggestions = []
    for std, (required, _, default) in schema.items():
        score, why = reasons.get(std, (None, "no match"))
        suggestions.append({"standard": std, "required": required, "uploaded_column": mapping.get(std),
                            "score": score, "reason": why,
                            "needs_review": std not in mapping or score < 0.95,
                            "if_missing": ("cannot run" if std in NEEDED[domain] else
                                           "enter rules on this screen" if std in ("made_sla", "order_estimated_delivery_date") else
                                           "left empty" if default is None else f"filled with {default!r}")})
    missing_required = [s["standard"] for s in suggestions if s["required"] and s["uploaded_column"] is None
                        and ALTERNATIVES.get(s["standard"]) not in mapping]
    return {"mapping": mapping, "suggestions": suggestions, "missing_required": missing_required,
            "cannot_run_without": cannot_run_without(mapping, domain),
            "unused_columns": [c for c in columns if c not in used], "ready": not missing_required}


def apply_mapping(df: pd.DataFrame, mapping: dict, domain="incidents") -> pd.DataFrame:
    """Rename uploaded columns to standard names and add missing optional columns with defaults."""
    schema = SCHEMAS[domain]
    check_needed(mapping, domain)
    out = df.rename(columns={src: std for std, src in mapping.items()})
    keep = [s for s in schema if s in out.columns]
    out = out[keep].copy()
    for std, (req, _, default) in schema.items():
        if std not in out.columns:
            out[std] = default
    return out
