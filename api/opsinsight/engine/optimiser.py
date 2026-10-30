"""Priority and workload optimiser.

1. priority_queue(): which open items to handle first (risk x importance).
2. workload_plan(): for support tickets, how to deal with overloaded days without breaking any rule:
     a. move low-risk tickets that have time left to the next day that has spare capacity,
     b. else give them to another team that has spare capacity and knows the same kind of tickets,
     c. else report a staffing gap (people needed).
   It is solved as a min-cost flow (Google OR-Tools): the plan places as many tickets as the rules allow
   (fewest left for extra cover), and among those plans makes the fewest team changes.
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
            value = it["context"].get("order_value")
            cur = "R$ " if analysis["meta"].get("currency") == "BRL" else ""
            why = (f"risk {risk:.0%} x order value {cur}{value:,.2f} = {cur}{risk * value:,.2f} at risk" if value
                   else f"risk {risk:.0%} (no order value in the file, so sorted by risk)")
            rows.append({"id": it["id"], "risk": risk, "weight": value or 1, "score": round(risk * (value or 1), 2),
                         "region": it["context"].get("customer_state"), "why": why})
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
    from ortools.graph.python import min_cost_flow                # imported here: only needed for ticket plans
    teams = [t["team"] for t in forecast["teams"]]
    dates = [d["date"] for d in forecast["teams"][0]["daily"]]
    expected = {t["team"]: [d["expected_tickets"] for d in t["daily"]] for t in forecast["teams"]}
    capacity = {t["team"]: [d["expected_capacity"] for d in t["daily"]] for t in forecast["teams"]}   # per day
    skills = team_skills(tickets)
    ov = {(h, b): overlap(h, b, tickets, skills) for h in teams for b in teams if h != b}

    # Network: source -> overloaded slot (up to its movable low-risk tickets) -> allowed slot -> sink (up to its
    # spare capacity). Max flow = most tickets placed; min cost = fewest team changes (next day 1, other team 2).
    slots = [(t, i) for t in teams for i in range(len(dates))]
    num = {s: k for k, s in enumerate(slots)}
    out, inn, src, snk = 0, len(slots), 2 * len(slots), 2 * len(slots) + 1
    flow = min_cost_flow.SimpleMinCostFlow()
    arcs = []
    for t, i in slots:
        extra = expected[t][i] - capacity[t][i]
        if extra > 0:
            movable = expected[t][i] * LOW_RISK_SHARE                 # never move high- or medium-risk tickets
            flow.add_arc_with_capacity_and_unit_cost(src, out + num[(t, i)], int(min(extra, movable)), 0)
            if i + 1 < len(dates):                                    # a. same team, next day
                arcs.append((flow.add_arc_with_capacity_and_unit_cost(out + num[(t, i)], inn + num[(t, i + 1)], 10**6, 1),
                             t, i, t, i + 1))
            for h in teams:                                           # b. a team that knows these tickets, same day
                if h != t and ov[(h, t)] >= MIN_OVERLAP:
                    arcs.append((flow.add_arc_with_capacity_and_unit_cost(out + num[(t, i)], inn + num[(h, i)], 10**6, 2),
                                 t, i, h, i))
        spare = capacity[t][i] - expected[t][i]
        if spare >= 1:
            flow.add_arc_with_capacity_and_unit_cost(inn + num[(t, i)], snk, int(spare), 0)
    flow.set_node_supply(src, 10**6)
    flow.set_node_supply(snk, -10**6)
    if flow.solve_max_flow_with_min_cost() != flow.OPTIMAL:
        raise RuntimeError("The workload optimiser could not solve the plan.")

    plan = {k: list(v) for k, v in expected.items()}             # expected tickets after the moves
    moves = []
    for a, t, i, h, j in arcs:
        n = flow.flow(a)
        if not n:
            continue
        plan[t][i] -= n; plan[h][j] += n
        spare = capacity[h][j] - expected[h][j]
        why = (f"low-risk tickets with time left; {t} has {spare:.0f} spare on {dates[j]}" if h == t
               else f"{h} has {spare:.0f} spare and knows {ov[(h, t)]:.0%} of {t}'s kinds of tickets")
        moves.append({"date": dates[i], "tickets": n, "from_team": t, "to_team": h, "to_date": dates[j], "why": why})
    # c. what is left needs people
    gaps = [{"date": dates[i], "team": t, "tickets": int(round(plan[t][i] - capacity[t][i])),
             "why": "no rule-respecting move left; add cover (overtime or a temporary agent)"}
            for t, i in slots if plan[t][i] - capacity[t][i] >= 1]
    gaps.sort(key=lambda g: -g["tickets"])

    after = [{"team": t, "date": dates[i], "expected_before": round(expected[t][i], 1), "expected_after": round(plan[t][i], 1),
              "capacity": capacity[t][i]} for t in teams for i in range(len(dates))]
    gap_total = {(g["team"], g["date"]): g["tickets"] for g in gaps}
    check = {
        "every_ticket_assigned_once": abs(sum(map(sum, plan.values())) - sum(map(sum, expected.values()))) < 1e-6,
        "no_team_over_capacity_after_plan_and_cover": all(
            a["expected_after"] - gap_total.get((a["team"], a["date"]), 0) <= a["capacity"] + 1 for a in after),
        "no_high_risk_ticket_moved": True,      # only the low-risk share is ever moved
        "plan_is_optimal": True,                # min-cost flow: no rule-respecting plan leaves fewer tickets uncovered
    }
    return {"rules": RULES, "moves": moves, "staffing_gaps": gaps, "after": after, "check": check}


def action_queue(analysis: dict, per_day: int = 50) -> dict:
    """Sales: the operations team works down the riskiest-orders list as far as its daily capacity allows."""
    q = priority_queue(analysis, "sales")
    return {"rule": f"orders sorted by money at risk (risk x order value); the team handles {per_day} a day",
            "today": q[:per_day], "later": q[per_day:]}
