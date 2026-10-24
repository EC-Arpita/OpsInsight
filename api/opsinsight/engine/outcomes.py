"""Outcome tracking: after the next upload, did the approved actions work?

For each approved action we compare the period before it with the period after it, on two numbers:
    the overall problem rate (breach rate or late rate), compared with what the action was expected to achieve;
    the action's own target (for routing: the share of tickets reassigned), so improvement that came from
    somewhere else is not credited to the action.
Before/after is not proof (seasons, staff and volume change too); the result says so.
"""
OVERALL = {"incidents": "sla_breach_rate", "sales": "late_delivery_rate"}
TARGET_KPI = {"incidents": {"REC1": "reassignment_rate", "REC3": "sla_breach_rate", "REC4": "sla_breach_rate"},
              "sales": {"REC1": "avg_dispatch_days", "REC3": "late_delivery_rate", "REC4": "late_delivery_rate"}}


def kpis(analysis: dict, which: str = "value") -> dict:
    """{kpi id: value} for the current period ('value') or the period before it ('previous_value')."""
    return {k["id"]: k[which] for k in analysis["kpis"]}


def _team_rate(analysis, team):
    for b in analysis.get("breakdowns", []):
        if b["dimension"] == "assignment_group":
            for row in b["rows"]:
                if row["value"] == team:
                    return row["breach_rate"]
    return None


def verdict(before, after, expected):
    """All three are rates where lower is better."""
    change, hoped = after - before, expected - before
    if hoped < 0 and change <= 0.5 * hoped:
        return "worked"
    if change < 0:
        return "partly worked"
    return "did not work"


def compare_outcomes(domain: str, recommendations: list, decisions: dict,
                     before_kpis: dict, after_kpis: dict, before_analysis=None, after_analysis=None) -> dict:
    overall = OVERALL[domain]
    rows = []
    for rec in recommendations:
        d = decisions.get(rec["id"])
        if not d or d["decision"] not in ("approved", "edited"):
            continue
        b, a = before_kpis[overall], after_kpis[overall]
        exp = round(b + rec["expected_impact"]["change_pct_points"] / 100, 3)   # the card's expected change, from "before"
        row = {"recommendation_id": rec["id"], "title": rec["title"], "approved_at": d["decided_at"],
               "overall": {"metric": overall, "before": b, "expected": exp, "after": a, "verdict": verdict(b, a, exp)}}
        key = TARGET_KPI[domain].get(rec["id"])
        if key and key != overall:
            tb, ta = before_kpis[key], after_kpis[key]
            row["own_target"] = {"metric": key, "before": tb, "after": ta, "moved_the_right_way": ta < tb}
        if domain == "incidents" and rec["id"] == "REC2" and before_analysis and after_analysis:
            team = rec["target"]["team"]
            tb, ta = _team_rate(before_analysis, team), _team_rate(after_analysis, team)
            if tb is not None and ta is not None:
                row["own_target"] = {"metric": f"{team} breach rate", "before": tb, "after": ta, "moved_the_right_way": ta < tb}
        if row.get("own_target") and not row["own_target"]["moved_the_right_way"] and row["overall"]["verdict"] != "did not work":
            row["note"] = "The overall rate improved but the action's own target did not, so something else caused it."
        rows.append(row)
    return {"rule": "worked = at least half the expected improvement; partly = some improvement; else did not work",
            "caution": "Before/after is not proof: seasons, staff and volume also change. A stronger test applies an "
                       "action to some teams or regions only and compares them with the others.",
            "actions": rows}
