"""Impact simulation: what would each action have changed in the current period?

Level 1 (estimates): one stated assumption per action, then simple arithmetic on the real tickets or orders.
Level 2 (replays):   replay day by day (ticket queues) or order by order (delivery timelines).
Every result says how it was estimated, so a reader can question the assumption instead of the number.
"""
import pandas as pd

from ..models.config import DOMAINS
from ..models.predict import load, predict

ASSUMED_PREVENTION = 0.30     # share of problems prevented by fast-tracking the riskiest items (to check in a pilot)


def _current(table, features, id_col):
    ids = set(features.loc[features["split"] == "test", id_col])
    return table[table[id_col].isin(ids)]


def _top10(domain, features):
    """Current-period items at or above the model's high-risk cut-off."""
    current = features[features["split"] == "test"]
    risk = pd.Series([p["prediction"]["risk"] for p in predict(domain, current)], index=current.index)
    return current[risk >= load(domain)["meta"]["risk_thresholds"]["high"]]


def _result(saved, n, bad, method, assumptions, low_share=0.5):
    new_rate = (bad - saved) / n
    return {"saved": round(float(saved)), "range": [round(float(saved) * low_share), round(float(saved))],
            "baseline_rate": round(bad / n, 3), "new_rate": round(float(new_rate), 3),
            "method": method, "assumptions": assumptions}


# --------------------------------------------------------------------------- support tickets
def ticket_estimates(tickets: pd.DataFrame, features: pd.DataFrame, worst_team: str | None) -> dict:
    cur = _current(tickets, features, "number")
    n, bad = len(cur), int(cur["breached"].sum())
    out = {}
    r1 = tickets.loc[tickets["final_reassignment_count"] == 1, "breached"].mean()   # whole data, as in the root cause
    r2 = cur[cur["final_reassignment_count"] >= 2]
    out["route_first_time"] = _result(r2["breached"].sum() - len(r2) * r1, n, bad,
        "Tickets with 2+ reassignments in the current period assumed to breach at the 1-reassignment rate.",
        ["Better routing cuts repeat reassignments, not the first one"])
    if worst_team:
        w = cur[cur["assignment_group"] == worst_team]
        out["rebalance_team"] = _result(max(0.0, w["breached"].sum() - len(w) * bad / n), n, bad,
            "Team assumed to reach the overall breach rate after rebalancing.",
            ["Receiving teams have the skills and spare capacity"])
    top = _top10("incidents", features)
    hit = cur[cur["number"].isin(top["number"])]
    out["fast_track_top10"] = _result(hit["breached"].sum() * ASSUMED_PREVENTION, n, bad,
        f"Assumes fast-tracking prevents {ASSUMED_PREVENTION:.0%} of breaches among flagged tickets (to be validated in a pilot).",
        [f"{ASSUMED_PREVENTION:.0%} prevention rate is an assumption"])
    out["_baseline"] = {"tickets": n, "breaches": bad, "breach_rate": round(bad / n, 3)}
    return out


def replay_queues(forecast: dict, plan: dict, cover: bool) -> dict:
    """Day-by-day replay of next week for each team: arrivals join a queue, the team finishes up to its
    capacity, the rest waits overnight. Compares the forecast as it is with the plan (and optional cover)."""
    after = {(a["team"], a["date"]): a["expected_after"] for a in plan["after"]}
    gaps = {(g["team"], g["date"]): g["tickets"] for g in plan["staffing_gaps"]}
    out = []
    for t in forecast["teams"]:
        res = {}
        for label, use_plan in (("baseline", False), ("with_plan", True)):
            waiting, nights, peak = 0.0, 0.0, 0.0
            for d in t["daily"]:
                arrivals = after[(t["team"], d["date"])] if use_plan else d["expected_tickets"]
                cap = d["expected_capacity"] + (gaps.get((t["team"], d["date"]), 0) if (use_plan and cover) else 0)
                queue = waiting + arrivals
                waiting = max(0.0, queue - cap)
                nights += waiting
                peak = max(peak, queue / cap if cap else 0)
            res[label] = {"ticket_nights_waiting": round(nights, 1), "peak_load_vs_capacity": round(peak, 2)}
        out.append({"team": t["team"], **res})
    return {"rule": "each day: queue = left over + arrivals; finished = capacity; the rest waits overnight",
            "with_cover": cover, "teams": out}


# --------------------------------------------------------------------------- sales orders
NORTHEAST = {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}


def _dispatch_days(orders):
    return ((orders["carrier_at"] - orders["approved_at"]).dt.total_seconds() / 86400).clip(lower=0)


def sales_estimates(orders: pd.DataFrame, features: pd.DataFrame) -> dict:
    d = orders[orders["is_model_row"] == 1].copy()
    d["late"] = d["late"].astype(int)
    d["dispatch_days"] = _dispatch_days(d)
    cur = _current(d, features, "order_id")
    n, bad = len(cur), int(cur["late"].sum())
    q50, q90 = d["dispatch_days"].quantile([0.5, 0.9])
    medium_rate = d.loc[d["dispatch_days"].between(q50, q90), "late"].mean()
    slow = cur[cur["dispatch_days"] >= q90]
    out = {"seller_dispatch_deadline": _result(slow["late"].sum() - len(slow) * medium_rate, n, bad,
        "Slow-dispatch orders assumed to be late at the rate of medium-dispatch orders.", ["Sellers respond to alerts"])}
    ne = cur[cur["customer_state"].isin(NORTHEAST)]
    out["region_buffer_days"] = _result(ne["late"].sum() * 0.40, n, bad,
        "Assumes 40% of North-East late orders arrive within the added buffer (validate with delivery-day distribution).",
        ["Slightly longer promises don't reduce conversion much"])
    top = _top10("sales", features)
    hit = cur[cur["order_id"].isin(top["order_id"])]
    out["proactive_top10"] = _result(hit["late"].sum() * ASSUMED_PREVENTION, n, bad,
        f"Assumes proactive handling prevents {ASSUMED_PREVENTION:.0%} of late deliveries among flagged orders (to be validated).",
        [f"{ASSUMED_PREVENTION:.0%} prevention rate is an assumption"])
    out["_baseline"] = {"orders": n, "late_orders": bad, "late_rate": round(bad / n, 3)}
    return out


def replay_dispatch(orders: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Order-by-order replay of the dispatch deadline: slow sellers ship as fast as a typical seller,
    the delivery date moves earlier by the time saved, and each order is checked against its promise again."""
    d = orders[orders["is_model_row"] == 1].copy()
    d["dispatch_days"] = _dispatch_days(d)
    typical = d["dispatch_days"].median()
    q90 = d["dispatch_days"].quantile(0.9)
    cur = _current(d, features, "order_id").copy()
    saved_days = (cur["dispatch_days"] - typical).where(cur["dispatch_days"] >= q90, 0).clip(lower=0)
    new_delivery = cur["delivered_at"] - pd.to_timedelta(saved_days, unit="D")
    late_before = int(cur["late"].sum())
    late_after = int((new_delivery > cur["estimated_at"]).sum())            # the same rule as Arpita's 'late' column
    return {"rule": f"orders whose seller took {q90:.1f}+ days to ship are replayed as if shipped in {typical:.1f} days",
            "orders": len(cur), "late_before": late_before, "late_after": late_after,
            "late_orders_avoided": late_before - late_after,
            "late_rate_before": round(late_before / len(cur), 3), "late_rate_after": round(late_after / len(cur), 3)}
