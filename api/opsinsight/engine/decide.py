"""Add every decision section to an analysis: ranked findings, the plan, recommendations, impact.

    add_decisions(analysis, "incidents", {"tickets": tickets, "features": features})
    add_decisions(analysis, "sales", {"orders": orders, "features": features})
"""
from .findings import rank_findings
from .optimiser import action_queue, priority_queue, workload_plan
from .recommend import KEYS, build_recommendations
from .simulate import replay_dispatch, replay_queues, sales_estimates, ticket_estimates

HALF = 0.5     # assumption for cover cards: half the tickets that would wait overnight because of overload breach


def _ticket_plan_text(plan, team):
    moves = sorted((m for m in plan["moves"] if m["from_team"] == team), key=lambda m: m["date"])
    gaps = [g for g in plan["staffing_gaps"] if g["team"] == team]
    parts = [f"move {m['tickets']} low-risk tickets from {m['date']} to {m['to_date']}" if m["to_team"] == team
             else f"give {m['tickets']} tickets on {m['date']} to {m['to_team']}" for m in moves]
    parts += [f"add cover for about {g['tickets']} tickets on {g['date']}" for g in gaps]
    return (f"For {team}: " + "; ".join(parts) + ".") if parts else ""


def add_decisions(analysis: dict, domain: str, tables: dict) -> dict:
    findings = rank_findings(analysis, domain)
    analysis["ranked_findings"] = findings
    avoided_key, _, _ = KEYS[domain]
    if domain == "incidents":
        tickets, features = tables["tickets"], tables["features"]
        worst = next((f["subject"] for f in findings["items"] if f["type"] == "capacity"), None)
        est = ticket_estimates(tickets, features, worst)
        plan = workload_plan(analysis["forecast"], tickets)
        replay = replay_queues(analysis["forecast"], plan, cover=True)
        base = est["_baseline"]
        texts = {}
        for f in findings["items"]:
            if f["type"] == "overload":
                r = next(t for t in replay["teams"] if t["team"] == f["subject"])
                saved = (r["baseline"]["ticket_nights_waiting"] - r["with_plan"]["ticket_nights_waiting"]) * HALF
                new_rate = (base["breaches"] - saved) / base["tickets"]
                est["cover_overload"] = {
                    "saved": round(saved), "range": [round(saved * 0.5), round(saved)], "baseline_rate": base["breach_rate"],
                    "new_rate": round(new_rate, 3),
                    "method": "Day-by-day replay of next week with the plan and extra cover: tickets left waiting overnight "
                              f"fall from {r['baseline']['ticket_nights_waiting']:.0f} to {r['with_plan']['ticket_nights_waiting']:.0f}; "
                              "half of them are assumed to breach.",
                    "assumptions": ["Half the tickets that wait overnight because of overload would breach (to be validated)",
                                    "Next week's forecast is compared with this period's breach rate"]}
                texts[f["subject"]] = _ticket_plan_text(plan, f["subject"])
        analysis["priority_queue"] = priority_queue(analysis, domain)
        analysis["workload_plan"] = plan
        replays = {"queues": replay}
        rate_key = "breach_rate"
    else:
        orders, features = tables["orders"], tables["features"]
        est = sales_estimates(orders, features)
        replay = replay_dispatch(orders, features)
        if "seller_dispatch_deadline" in est:
            est["seller_dispatch_deadline"]["method"] += (f" Replay check: replaying each order with a typical dispatch time "
                                                          f"avoids {replay['late_orders_avoided']} late orders (optimistic upper end).")
        base = est["_baseline"]
        cur_late = base["late_rate"]
        texts = {}
        for f in findings["items"]:
            if f["type"] == "busy":
                extra = sum(d["expected_orders"] - d["usual_orders"] for d in f["evidence"])
                saved = extra * cur_late * 0.30
                est["warn_busy_region"] = {
                    "saved": round(saved), "range": [round(saved * 0.5), round(saved)], "baseline_rate": base["late_rate"],
                    "new_rate": round((base["late_orders"] - saved) / base["orders"], 3),
                    "method": f"About {extra:.0f} extra orders expected; they are assumed late at this period's rate "
                              f"({cur_late:.1%}), and an early warning is assumed to prevent 30% of those.",
                    "assumptions": ["Early warning prevents 30% of the extra lateness (to be validated)"]}
                texts[f["subject"]] = ("Tell sellers and couriers in " + f["subject"] + " a few days ahead: about "
                                       f"{extra:.0f} more orders than usual are expected on "
                                       + ", ".join(d["date"] for d in f["evidence"]) + ".")
        analysis["priority_queue"] = action_queue(analysis)
        replays = {"dispatch": replay}
        rate_key = "late_rate"
    recs = build_recommendations(domain, findings, est, texts)
    analysis["recommendations"] = recs
    analysis["impact_simulation"] = {
        "period": "current", "baseline": base,
        "scenarios": [{"recommendation_id": r["id"], rate_key: r["expected_impact"]["expected"],
                       avoided_key.replace("_per_period", ""): r["expected_impact"][avoided_key]} for r in recs],
        "replays": replays,
        "note": "Scenarios are estimated separately and are not additive (they overlap)."}
    return analysis
