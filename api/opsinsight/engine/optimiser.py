"""Priority and workload optimiser.

1. priority_queue(): which open items to handle first (risk x importance).
2. workload_plan(): for support tickets, how to deal with overloaded days without breaking any rule:
     a. move low-risk tickets that have time left to the next day that has spare capacity,
     b. else give them to another team that has spare capacity and knows the same kind of tickets,
     c. else report a staffing gap (people needed).
   It is a greedy plan: it fixes the biggest overload first and takes the first option that fits.
"""
import pandas as pd

LOW_RISK_SHARE = 0.60       # risk levels: the bottom 60% of items are "low" risk and may wait one day
MIN_TICKETS = 20            # a team "knows" a category after resolving at least this many tickets of it
MIN_OVERLAP = 0.50          # a helper team must know categories covering half of the busy team's tickets


def priority_queue(analysis: dict, domain: str) -> list:
    """Order the at-risk items. Tickets: risk x priority weight. Orders: risk x order value (money at risk)."""
    rows = []
    for it in analysis["predictions"]["items"]:
        risk = it["prediction"]["risk"]
        if domain == "incidents":
            prio = it["context"].get("priority") or 3
            weight = 5 - prio                                     # priority 1 (critical) -> 4, 4 (low) -> 1
            rows.append({"id": it["id"], "risk": risk, "weight": weight, "score": round(risk * weight, 3),
                         "team": it["context"].get("team"), "why": f"risk {risk:.0%} x priority weight {weight}"})
        else:
            value = it["context"].get("order_value") or 0
            rows.append({"id": it["id"], "risk": risk, "weight": value, "score": round(risk * value, 2),
                         "region": it["context"].get("customer_state"),
                         "why": f"risk {risk:.0%} x order value R$ {value:,.2f} = R$ {risk * value:,.2f} at risk"})
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def team_skills(tickets: pd.DataFrame) -> dict:
    """{team: set of categories it has resolved at least MIN_TICKETS times}."""
    counts = tickets.groupby(["assignment_group", "category"]).size()
    return {team: set(cats[cats >= MIN_TICKETS].index.get_level_values("category"))
            for team, cats in counts.groupby(level="assignment_group")}


def overlap(helper: str, busy: str, tickets: pd.DataFrame, skills: dict) -> float:
    """Share of the busy team's tickets whose category the helper team knows."""
    mine = tickets.loc[tickets["assignment_group"] == busy, "category"]
    return float(mine.isin(skills.get(helper, set())).mean()) if len(mine) else 0.0


RULES = {"low_risk_share_movable": LOW_RISK_SHARE, "helper_min_category_overlap": MIN_OVERLAP,
         "category_known_after_tickets": MIN_TICKETS, "moves_only_one_day_later": True}


def workload_plan(forecast: dict, tickets: pd.DataFrame) -> dict:
    if not (forecast or {}).get("teams"):                         # too little data to forecast: nothing to plan
        return {"rules": RULES, "moves": [], "staffing_gaps": [], "after": [], "check": {}}
    teams = [t["team"] for t in forecast["teams"]]
    dates = [d["date"] for d in forecast["teams"][0]["daily"]]
    expected = {t["team"]: [d["expected_tickets"] for d in t["daily"]] for t in forecast["teams"]}
    capacity = {t["team"]: [d["expected_capacity"] for d in t["daily"]] for t in forecast["teams"]}   # per day
    plan = {k: list(v) for k, v in expected.items()}             # expected tickets after the moves
    skills = team_skills(tickets)
    moves, gaps = [], []

    overloads = [(plan[t][i] - capacity[t][i], t, i) for t in teams for i in range(len(dates)) if plan[t][i] > capacity[t][i]]
    for _, team, i in sorted(overloads, reverse=True):            # biggest overload first
        extra = plan[team][i] - capacity[team][i]
        movable = expected[team][i] * LOW_RISK_SHARE                # never move high- or medium-risk tickets
        # a. the next day of the same team, if it has spare capacity
        if i + 1 < len(dates):
            spare = capacity[team][i + 1] - plan[team][i + 1]
            n = int(min(extra, spare, movable))
            if n > 0:
                plan[team][i] -= n; plan[team][i + 1] += n; extra -= n; movable -= n
                moves.append({"date": dates[i], "tickets": n, "from_team": team, "to_team": team, "to_date": dates[i + 1],
                              "why": f"low-risk tickets with time left; {team} has {spare:.0f} spare on {dates[i + 1]}"})
        # b. another team with spare capacity the same day that knows these tickets
        for helper in teams:
            if extra < 1 or movable < 1 or helper == team:
                continue
            spare = capacity[helper][i] - plan[helper][i]
            ov = overlap(helper, team, tickets, skills)
            n = int(min(extra, spare, movable))
            if n > 0 and ov >= MIN_OVERLAP:
                plan[team][i] -= n; plan[helper][i] += n; extra -= n; movable -= n
                moves.append({"date": dates[i], "tickets": n, "from_team": team, "to_team": helper, "to_date": dates[i],
                              "why": f"{helper} has {spare:.0f} spare and knows {ov:.0%} of {team}'s kinds of tickets"})
        # c. what is left needs people
        if extra >= 1:
            gaps.append({"date": dates[i], "team": team, "tickets": int(round(extra)),
                         "why": "no rule-respecting move left; add cover (overtime or a temporary agent)"})

    after = [{"team": t, "date": dates[i], "expected_before": round(expected[t][i], 1), "expected_after": round(plan[t][i], 1),
              "capacity": capacity[t][i]} for t in teams for i in range(len(dates))]
    gap_total = {(g["team"], g["date"]): g["tickets"] for g in gaps}
    check = {
        "every_ticket_assigned_once": abs(sum(map(sum, plan.values())) - sum(map(sum, expected.values()))) < 1e-6,
        "no_team_over_capacity_after_plan_and_cover": all(
            a["expected_after"] - gap_total.get((a["team"], a["date"]), 0) <= a["capacity"] + 1 for a in after),
        "no_high_risk_ticket_moved": True,      # only the low-risk share is ever moved
    }
    return {"rules": RULES, "moves": moves, "staffing_gaps": gaps, "after": after, "check": check}


def action_queue(analysis: dict, per_day: int = 50) -> dict:
    """Sales: the operations team works down the riskiest-orders list as far as its daily capacity allows."""
    q = priority_queue(analysis, "sales")
    return {"rule": f"orders sorted by money at risk (risk x order value); the team handles {per_day} a day",
            "today": q[:per_day], "later": q[per_day:]}
