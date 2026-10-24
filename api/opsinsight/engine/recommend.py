"""Recommended actions: each important finding becomes an action card from the playbook.

The playbook is the list of proven actions for each type of finding, with blanks to fill. The ids stay the
same every time (REC1 = routing for tickets, and so on), so a manager's decision always stays with its card.
"""
PLAYBOOK = {
    "incidents": {
        "process": {"rec_id": "REC1", "estimate": "route_first_time", "type": "process",
                    "title": "Route tickets to the right team first time",
                    "description": "Use the model's category/symptom signals to suggest the correct team at opening; "
                                   "require a reason when reassigning.",
                    "target": {"tickets": "all new tickets", "focus": "categories with most reassignments"},
                    "effort": "medium", "owner": "Helpdesk lead"},
        "capacity": {"rec_id": "REC2", "estimate": "rebalance_team", "type": "capacity",
                     "title": "Rebalance workload away from {subject}",
                     "description": "Move part of {subject}'s incoming tickets to teams with spare capacity, or add temporary staff.",
                     "target": {"team": "{subject}"}, "effort": "low", "owner": "Operations manager"},
        "high_risk": {"rec_id": "REC3", "estimate": "fast_track_top10", "type": "prioritisation",
                      "title": "Fast-track the 10% highest-risk tickets",
                      "description": "Every new ticket in the model's top 10% risk goes to a senior agent within 1 hour.",
                      "target": {"tickets": "the model's top 10% risk"}, "effort": "low", "owner": "Shift supervisor"},
        "overload": {"rec_id": "REC4", "estimate": "cover_overload", "type": "capacity",
                     "title": "Cover {subject} on its overloaded days",
                     "description": "{plan}", "target": {"team": "{subject}"}, "effort": "low", "owner": "Team lead"},
    },
    "sales": {
        "supplier": {"rec_id": "REC1", "estimate": "seller_dispatch_deadline", "type": "supplier",
                     "title": "Enforce a dispatch deadline for slow sellers",
                     "description": "Alert sellers whose dispatch time is in the slowest 10%; escalate after 2 days; "
                                    "deprioritise repeat offenders in search.",
                     "target": {"sellers": "slowest 10% by dispatch time"}, "effort": "medium", "owner": "Seller operations"},
        "logistics": {"rec_id": "REC2", "estimate": "region_buffer_days", "type": "policy",
                      "title": "Add buffer days to promises for North-East routes",
                      "description": "Add 2-3 days to the promised date for high-risk routes so promises match reality.",
                      "target": {"customer_states": "North-East"}, "effort": "low", "owner": "Logistics manager"},
        "high_risk": {"rec_id": "REC3", "estimate": "proactive_top10", "type": "prioritisation",
                      "title": "Proactively handle the 10% highest-risk orders",
                      "description": "For each order in the model's top 10%: nudge the seller, upgrade carrier, and send "
                                     "the customer a delivery update.",
                      "target": {"orders": "the model's top 10% risk"}, "effort": "medium", "owner": "Customer experience team"},
        "busy": {"rec_id": "REC4", "estimate": "warn_busy_region", "type": "capacity",
                 "title": "Warn sellers and couriers in {subject} before its busy days",
                 "description": "{plan}", "target": {"region": "{subject}"}, "effort": "low", "owner": "Logistics manager"},
    },
}
KEYS = {"incidents": ("breaches_avoided_per_period", "range_breaches_avoided", "sla_breach_rate"),
        "sales": ("late_orders_avoided_per_period", "range_late_orders_avoided", "late_delivery_rate")}


def _fill(value, subject, plan_text):
    if isinstance(value, str):
        return value.replace("{subject}", subject or "").replace("{plan}", plan_text or "")
    if isinstance(value, dict):
        return {k: _fill(v, subject, plan_text) for k, v in value.items()}
    return value


def build_recommendations(domain: str, findings: dict, estimates: dict, plan_texts: dict) -> list:
    """findings = rank_findings() output; estimates = {estimate key: result}; plan_texts = {subject: sentence}."""
    avoided_key, range_key, metric = KEYS[domain]
    cards, used = [], set()
    for f in findings["items"]:
        entry = PLAYBOOK[domain].get(f["type"])
        if entry is None or entry["rec_id"] in used:
            continue                                        # no proven action for this type, or already proposed
        est = estimates.get(entry["estimate"])
        if est is None:
            continue
        used.add(entry["rec_id"])
        subject = f.get("subject")
        evidence = [f"{f['id']}: {f['finding']}"]
        cards.append({
            "id": entry["rec_id"], "title": _fill(entry["title"], subject, None), "type": entry["type"],
            "description": _fill(entry["description"], subject, plan_texts.get(subject)),
            "target": _fill(entry["target"], subject, None), "evidence": evidence,
            "expected_impact": {"metric": metric, "baseline": est["baseline_rate"], "expected": est["new_rate"],
                                "change_pct_points": round((est["new_rate"] - est["baseline_rate"]) * 100, 1),
                                avoided_key: est["saved"], range_key: est["range"]},
            "estimate_method": est["method"], "assumptions": est["assumptions"],
            "confidence": f["confidence"], "effort": entry["effort"], "suggested_owner": entry["owner"],
            "linked_findings": [f["id"]] + ([f["related"]] if f.get("related") else []),
            "linked_root_causes": [f["id"]] if f["source"] == "root_causes" else [],
            "status": "pending", "approval": {"decided_by": None, "decided_at": None, "decision": None, "comment": None}})
    # order: most problems avoided first; when close, the easier action first
    effort_rank = {"low": 0, "medium": 1, "high": 2}
    cards.sort(key=lambda c: (-c["expected_impact"][avoided_key], effort_rank[c["effort"]]))
    top = cards[0]["expected_impact"][avoided_key] if cards else 0
    for c in cards:
        c["quick_win"] = c["effort"] == "low" and c["expected_impact"][avoided_key] >= 0.4 * top
    return cards
