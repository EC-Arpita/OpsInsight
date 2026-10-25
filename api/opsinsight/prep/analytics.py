"""Turn clean tables into the analysis JSON sections (the work of notebook 08).

    from opsinsight.prep import prepare_incidents
    from opsinsight.prep.analytics import incident_analysis
    out = prepare_incidents("data/raw/incidents/incident_event_log.csv")
    analysis = incident_analysis(out["tickets"], out["features"], out["quality"])
    analysis["kpis"], analysis["breakdowns"], analysis["root_causes"], ...
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .sales import REGION


# --------------------------------------------------------------------------- small helpers
def num(x, d=3):
    """Round, and turn numpy numbers into plain Python numbers (JSON can't store numpy types)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return round(float(x), d)


def pct_change(now, before):
    return None if not before else num((now - before) / before * 100, 1)


def kpi(id, label, now, before, unit, definition, higher_is_better, warn=None, crit=None, d=3, mode="level"):
    """One KPI card. higher_is_better=None means a volume (status 'info').
    mode='level': thresholds on the value; mode='change': thresholds on % change in the bad direction."""
    change = None if before is None else now - before
    chg_pct = pct_change(now, before)
    if higher_is_better is None:
        status, trend = "info", (None if change is None else "up" if change > 0 else "down" if change < 0 else "flat")
    else:
        if mode == "change":
            bad = 0 if chg_pct is None else (-chg_pct if higher_is_better else chg_pct)   # no earlier value: no alarm
            status = "critical" if bad >= crit else "warning" if bad >= warn else "good"
        elif higher_is_better:
            status = "critical" if now <= crit else "warning" if now <= warn else "good"
        else:
            status = "critical" if now >= crit else "warning" if now >= warn else "good"
        trend = None if change is None else ("flat" if change == 0 else
                                             "improving" if (change > 0) == higher_is_better else "worsening")
    return {"id": id, "label": label, "value": num(now, d), "unit": unit,
            "previous_value": num(before, d), "change": num(change, d), "change_pct": chg_pct,
            "trend": trend, "higher_is_better": higher_is_better, "status": status,
            "status_rule": None if higher_is_better is None else
                           {"mode": mode, "warning": warn, "critical": crit,
                            "meaning": "thresholds on % change in the bad direction" if mode == "change" else "thresholds on the value"},
            "definition": definition}


def split_periods(df, time_col, cutoff, end):
    """Current period = cutoff..end; previous = the same length just before."""
    length = end - cutoff
    prev_start = cutoff - length
    cur = df[df[time_col] >= cutoff]
    prev = df[(df[time_col] >= prev_start) & (df[time_col] < cutoff)]
    return cur, prev, prev_start


def breakdown(df, col, label, target, id_col, names, extras, top=8, min_n=50):
    """'Where is it worst?' table for one column, sorted by number of problems.
    names  = output key names, e.g. {"count": "tickets", "rate": "breach_rate", "problems": "breaches",
             "share": "share_of_all_breaches", "vs": "vs_overall_breach_rate"}
    extras = {"output_name": (column, "mean" | "median" | "sum", decimals)}"""
    total = df[target].sum()
    overall = df[target].mean()
    agg = {"_n": (id_col, "size"), "_rate": (target, "mean"), "_problems": (target, "sum")}
    agg.update({k: (c, how) for k, (c, how, _) in extras.items()})
    g = df.groupby(col).agg(**agg)
    g = g[g["_n"] >= min_n].sort_values("_problems", ascending=False).head(top)
    rows = []
    for value, r in g.iterrows():
        row = {"value": str(value), names["count"]: int(r["_n"]), names["rate"]: num(r["_rate"]),
               names["problems"]: int(r["_problems"]), names["share"]: num(r["_problems"] / total),
               names["vs"]: num(r["_rate"] - overall)}
        row.update({k: num(r[k], dec) for k, (_, _, dec) in extras.items()})
        row["flag"] = "above_average" if r["_rate"] > overall * 1.2 else "normal"
        rows.append(row)
    return {"dimension": col, "label": label, "sorted_by": names["problems"], "rows": rows}


def anomalies(series, metric, label):
    """Weeks far from normal: |value - median| / (1.4826 * MAD) > 2 (a robust z-score)."""
    med = series.median()
    mad = (series - med).abs().median() * 1.4826
    out = []
    for d, v in series.items():
        z = (v - med) / mad if mad else 0
        if abs(z) > 2:
            out.append({"week_ending": d.strftime("%Y-%m-%d"), "metric": metric, "label": label, "actual": num(v),
                        "expected": num(med), "deviation_score": num(z, 1),
                        "severity": "critical" if abs(z) > 4 else "warning",
                        "direction": "above_normal" if v > med else "below_normal"})
    return out


def quality_section(quality, rows_in, rows_used, item_name, warnings, modules):
    steps = [{"step": r.step, "rows_before": int(r.rows_before), "rows_after": int(r.rows_after),
              "rows_removed": int(r.rows_removed), "note": "" if pd.isna(r.note) else r.note}
             for r in quality.itertuples()]
    return {"rows_in_raw_file": rows_in, f"{item_name}_used_for_analysis": rows_used,
            "cleaning_steps": steps, "warnings": warnings, "modules": modules}


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- incidents
INC_NAMES = {"count": "tickets", "rate": "breach_rate", "problems": "breaches",
             "share": "share_of_all_breaches", "vs": "vs_overall_breach_rate"}
INC_EXTRAS = {"median_resolution_hours": ("hours_to_resolve", "median", 1),
              "avg_reassignments": ("final_reassignment_count", "mean", 2)}


def incident_analysis(tickets, features, quality, dataset_name="Incident log"):
    """meta, data_quality, kpis, trends, breakdowns, root_causes, anomalies for support tickets."""
    inc = tickets.merge(features[["number", "split"]], on="number")
    cutoff = features.loc[features["split"] == "test", "opened_at"].min()
    end = features["opened_at"].max()
    cur, prev, prev_start = split_periods(inc, "opened_at", cutoff, end)

    def k(df):
        return {"tickets": len(df), "breach": df["breached"].mean(), "median_h": df["hours_to_resolve"].median(),
                "reassigned": (df["final_reassignment_count"] > 0).mean(),
                "reopened": (df["final_reopen_count"] > 0).mean(), "high_prio": (df["priority_n"] <= 2).mean()}
    a, b = k(cur), k(prev)
    kpis = [
        kpi("ticket_volume", "Tickets received", a["tickets"], b["tickets"], "tickets",
            "Number of tickets opened in the period", None, d=0),
        kpi("sla_breach_rate", "SLA breach rate", a["breach"], b["breach"], "ratio",
            "Share of tickets that missed their SLA deadline", False, 0.25, 0.35),
        kpi("median_resolution_hours", "Median time to resolve", a["median_h"], b["median_h"], "hours",
            "Median hours from opening to resolution", False, 24, 48, d=1),
        kpi("reassignment_rate", "Tickets reassigned at least once", a["reassigned"], b["reassigned"], "ratio",
            "Share of tickets passed to another team at least once", False, 0.35, 0.50),
        kpi("reopen_rate", "Reopened tickets", a["reopened"], b["reopened"], "ratio",
            "Share of tickets reopened after being resolved", False, 0.02, 0.05),
        kpi("high_priority_share", "Critical + High priority share", a["high_prio"], b["high_prio"], "ratio",
            "Share of tickets with priority 1 or 2", False, 0.05, 0.10),
    ]

    w = inc.set_index("opened_at").resample("W-SUN")
    weekly = pd.DataFrame({"tickets": w.size(), "breach_rate": w["breached"].mean(),
                           "median_resolution_hours": w["hours_to_resolve"].median()}).dropna()
    trends = {"granularity": "week", "series": [
        {"week_ending": d.strftime("%Y-%m-%d"), "tickets": int(r.tickets), "breach_rate": num(r.breach_rate),
         "median_resolution_hours": num(r.median_resolution_hours, 1)} for d, r in weekly.iterrows()]}

    def bd(col, label, top=8, min_n=50, data=cur):
        return breakdown(data, col, label, "breached", "number", INC_NAMES, INC_EXTRAS, top, min_n)
    breakdowns = [bd("assignment_group", "Team (assignment group)"), bd("category", "Category"),
                  bd("priority_n", "Priority (1 = Critical)", top=4, min_n=1)]

    # root causes
    r0 = inc.loc[inc["final_reassignment_count"] == 0, "breached"].mean()
    r1 = inc.loc[inc["final_reassignment_count"] == 1, "breached"].mean()
    r2 = inc[inc["final_reassignment_count"] >= 2]
    r5 = inc.loc[inc["final_reassignment_count"] >= 5, "breached"].mean()
    cur_r2 = cur[cur["final_reassignment_count"] >= 2]
    ro1 = inc.loc[inc["final_reopen_count"] > 0, "breached"].mean()
    ro0 = inc.loc[inc["final_reopen_count"] == 0, "breached"].mean()
    teams = bd("assignment_group", "", top=30)["rows"] or bd("assignment_group", "", top=30, min_n=1)["rows"]   # small files: no team has 50
    big = [t for t in teams if t["tickets"] >= 150] or teams      # small files: fall back to all teams
    worst = max(big, key=lambda t: t["breach_rate"])
    root_causes = [
        {"id": "RC1", "title": "Tickets bounced between teams (reassignment)", "category": "process",
         "finding": f"Breach rate rises from {r0:.0%} with no reassignment to {r1:.0%} after one and {r5:.0%} after five or more.",
         "evidence": [{"segment": "0 reassignments", "breach_rate": num(r0)},
                      {"segment": "1 reassignment", "breach_rate": num(r1)},
                      {"segment": "2+ reassignments", "breach_rate": num(r2["breached"].mean()), "tickets": len(r2)},
                      {"segment": "5+ reassignments", "breach_rate": num(r5)}],
         "affected_tickets_current_period": len(cur_r2),
         "share_of_breaches_current_period": num(cur_r2["breached"].sum() / cur["breached"].sum()),
         "confidence": "high", "linked_recommendations": ["REC1"],
         "note": "Correlation from historical data; reassignment is an outcome, so it is used for root-cause analysis, not as a model input."},
        {"id": "RC2", "title": f"{worst['value']} is under strain", "category": "capacity",
         "finding": f"{worst['value']} breached {worst['breach_rate']:.0%} of its {worst['tickets']} tickets this period vs {cur['breached'].mean():.0%} overall.",
         "evidence": [{"segment": worst["value"], "breach_rate": worst["breach_rate"], "tickets": worst["tickets"]},
                      {"segment": "all teams", "breach_rate": num(cur["breached"].mean()), "tickets": len(cur)}],
         "affected_tickets_current_period": worst["tickets"],
         "share_of_breaches_current_period": worst["share_of_all_breaches"],
         "confidence": "medium", "linked_recommendations": ["REC2"]},
        {"id": "RC3", "title": "Reopened tickets", "category": "quality",
         "finding": f"Reopened tickets breach {ro1:.0%} of the time vs {ro0:.0%} for others.",
         "evidence": [{"segment": "reopened", "breach_rate": num(ro1)}, {"segment": "not reopened", "breach_rate": num(ro0)}],
         "affected_tickets_current_period": int((cur["final_reopen_count"] > 0).sum()),
         "share_of_breaches_current_period": num(cur.loc[cur["final_reopen_count"] > 0, "breached"].sum() / cur["breached"].sum()),
         "confidence": "medium", "linked_recommendations": []},
    ]
    # a file without reassignment or reopen counts (filled with 0) can't show those causes or KPIs
    gone = {"RC1", "reassignment_rate"} if pd.isna(r1) else set()
    gone |= {"RC3", "reopen_rate"} if pd.isna(ro1) else set()
    root_causes = [rc for rc in root_causes if rc["id"] not in gone]
    kpis = [k for k in kpis if k["id"] not in gone]

    return {
        "schema_version": "1.0",
        "meta": {"analysis_id": "inc-" + _now()[:10], "generated_at": _now(), "domain": "support_tickets",
                 "dataset_name": dataset_name, "item_type": "ticket",
                 "target": {"name": "sla_breach", "definition": "ticket missed its SLA deadline (made_sla = False on the last event)"},
                 "prediction_moment": "when the ticket is opened",
                 "data_period": {"start": f"{inc['opened_at'].min():%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"},
                 "current_period": {"start": f"{cutoff:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"},
                 "previous_period": {"start": f"{prev_start:%Y-%m-%d}", "end": f"{cutoff - pd.Timedelta(minutes=1):%Y-%m-%d}"},
                 "currency": None, "timezone": "local time of source system"},
        "data_quality": quality_section(quality, int(quality["rows_before"].iloc[0]), len(inc), "tickets", warnings=[], modules=[
            {"module": "kpis", "enabled": True}, {"module": "root_cause", "enabled": True},
            {"module": "revenue_impact", "enabled": False, "reason": "dataset has no cost or revenue column"}]),
        "kpis": kpis, "trends": trends, "breakdowns": breakdowns, "root_causes": root_causes,
        "anomalies": anomalies(weekly["breach_rate"], "breach_rate", "Weekly SLA breach rate"),
    }


# --------------------------------------------------------------------------- sales
SALES_NAMES = {"count": "orders", "rate": "late_rate", "problems": "late_orders",
               "share": "share_of_all_late", "vs": "vs_overall_late_rate"}
SALES_EXTRAS = {"avg_delivery_days": ("delivery_days", "mean", 1), "avg_review_score": ("review_score", "mean", 2),
                "revenue": ("revenue", "sum", 0)}


def sales_analysis(orders, features, quality, dataset_name="Sales orders"):
    """meta, data_quality, kpis, trends, breakdowns, root_causes, anomalies for sales orders."""
    d = orders[orders["is_model_row"] == 1].merge(features[["order_id", "split"]], on="order_id")
    d["late"] = d["late"].astype(int)
    d["delivery_days"] = (d["delivered_at"] - d["purchased_at"]).dt.total_seconds() / 86400
    d["dispatch_days"] = ((d["carrier_at"] - d["approved_at"]).dt.total_seconds() / 86400).clip(lower=0)
    d["revenue"] = d["items_price"] + d["items_freight"]
    d["region"] = d["customer_state"].map(REGION).fillna(d["customer_state"])   # Brazil: 5 regions; elsewhere as given
    cutoff = features.loc[features["split"] == "test", "purchased_at"].min()
    end = features["purchased_at"].max()
    cur, prev, prev_start = split_periods(d, "purchased_at", cutoff, end)

    def k(df):
        return {"orders": len(df), "late": df["late"].mean(), "days": df["delivery_days"].mean(),
                "review": df["review_score"].mean(), "revenue": df["revenue"].sum(), "dispatch": df["dispatch_days"].mean()}
    a, b = k(cur), k(prev)
    kpis = [
        kpi("orders_delivered", "Orders delivered", a["orders"], b["orders"], "orders", "Delivered orders purchased in the period", None, d=0),
        kpi("revenue", "Revenue (items + freight)", a["revenue"], b["revenue"], "BRL", "Sum of item price and freight", True, 5, 20, d=0, mode="change"),
        kpi("late_delivery_rate", "Late delivery rate", a["late"], b["late"], "ratio", "Share delivered after the promised date", False, 0.07, 0.10),
        kpi("avg_delivery_days", "Average delivery time", a["days"], b["days"], "days", "Days from purchase to delivery", False, 12, 15, d=1),
        kpi("avg_dispatch_days", "Average seller dispatch time", a["dispatch"], b["dispatch"], "days", "Days from approval to carrier pickup", False, 3, 4, d=2),
        kpi("avg_review_score", "Average review score", a["review"], b["review"], "stars (1-5)", "Mean review score of delivered orders", True, 4.0, 3.5, d=2),
    ]

    w = d.set_index("purchased_at").resample("W-SUN")
    sw = pd.DataFrame({"orders": w.size(), "late_rate": w["late"].mean(), "revenue": w["revenue"].sum(),
                       "avg_delivery_days": w["delivery_days"].mean()}).dropna()
    trends = {"granularity": "week", "series": [
        {"week_ending": i.strftime("%Y-%m-%d"), "orders": int(r.orders), "late_rate": num(r.late_rate),
         "revenue": num(r.revenue, 0), "avg_delivery_days": num(r.avg_delivery_days, 1)} for i, r in sw.iterrows()]}

    def bd(col, label, min_n=100):
        return breakdown(cur, col, label, "late", "order_id", SALES_NAMES, SALES_EXTRAS, 8, min_n)
    breakdowns = [bd("customer_state", "Customer state"), bd("main_category", "Product category"),
                  bd("main_seller_id", "Seller", min_n=50)]

    q = d["dispatch_days"].quantile([0.5, 0.9])
    fast, slow = d[d["dispatch_days"] <= q[0.5]], d[d["dispatch_days"] >= q[0.9]]
    promised = (d["estimated_at"] - d["approved_at"]).dt.total_seconds() / 86400
    short, long_ = d[promised <= promised.quantile(0.25)], d[promised >= promised.quantile(0.75)]
    regions = d.groupby("region")["late"].agg(["mean", "size"]).query("size >= 100")
    worst_r, best_r = (regions["mean"].idxmax(), regions["mean"].idxmin()) if len(regions) >= 2 else (None, None)
    wr, br = d[d["region"] == worst_r], d[d["region"] == best_r]
    on_time_rev, late_rev = d.loc[d["late"] == 0, "review_score"].mean(), d.loc[d["late"] == 1, "review_score"].mean()
    cur_slow = cur[cur["dispatch_days"] >= q[0.9]]
    cur_w = cur["region"] == worst_r
    root_causes = [
        {"id": "RC1", "title": "Slow seller dispatch", "category": "supplier",
         "finding": f"Orders the seller took {q[0.9]:.1f}+ days to ship are late {slow['late'].mean():.0%} of the time vs {fast['late'].mean():.0%} when shipped within {q[0.5]:.1f} days.",
         "evidence": [{"segment": f"dispatch ≤ {q[0.5]:.1f} days", "late_rate": num(fast["late"].mean()), "orders": len(fast)},
                      {"segment": f"dispatch ≥ {q[0.9]:.1f} days", "late_rate": num(slow["late"].mean()), "orders": len(slow)}],
         "affected_orders_current_period": len(cur_slow),
         "share_of_late_current_period": num(cur_slow["late"].sum() / cur["late"].sum()),
         "confidence": "high", "linked_recommendations": ["REC1"]},
        {"id": "RC2", "title": f"Late deliveries concentrate in {worst_r}", "category": "logistics",
         "finding": f"{worst_r} customers get late orders {wr['late'].mean():.0%} of the time vs {br['late'].mean():.0%} in {best_r}.",
         "evidence": [{"segment": worst_r, "late_rate": num(wr["late"].mean()), "orders": len(wr)},
                      {"segment": best_r, "late_rate": num(br["late"].mean()), "orders": len(br)}],
         "affected_orders_current_period": int(cur_w.sum()),
         "share_of_late_current_period": num(cur.loc[cur_w, "late"].sum() / cur["late"].sum()),
         "confidence": "high", "linked_recommendations": ["REC2"]},
        {"id": "RC3", "title": "Tight delivery promises", "category": "policy",
         "finding": f"Orders with the shortest 25% of promises are late {short['late'].mean():.0%} vs {long_['late'].mean():.0%} for the longest 25%.",
         "evidence": [{"segment": "shortest 25% promises", "late_rate": num(short["late"].mean())},
                      {"segment": "longest 25% promises", "late_rate": num(long_["late"].mean())}],
         "affected_orders_current_period": None, "share_of_late_current_period": None,
         "confidence": "medium", "linked_recommendations": ["REC2"]},
        {"id": "RC4", "title": "Business impact: late orders damage reviews", "category": "customer",
         "finding": f"Late orders average {late_rev:.1f} stars vs {on_time_rev:.1f} on time.",
         "evidence": [{"segment": "on time", "avg_review_score": num(on_time_rev, 2)}, {"segment": "late", "avg_review_score": num(late_rev, 2)}],
         "affected_orders_current_period": int(cur["late"].sum()), "share_of_late_current_period": 1.0,
         "confidence": "high", "linked_recommendations": []},
    ]
    # an orders file without items, ship dates, reviews or regions can't show those parts
    gone = set()
    if not d["revenue"].sum():
        gone.add("revenue")
    if d["dispatch_days"].isna().all():
        gone |= {"avg_dispatch_days", "RC1"}
    if d["review_score"].isna().all():
        gone |= {"avg_review_score", "RC4"}
    if worst_r is None or wr["late"].mean() < 1.2 * br["late"].mean():   # needs 2 regions and a real gap
        gone.add("RC2")
    kpis = [k for k in kpis if k["id"] not in gone]
    root_causes = [rc for rc in root_causes if rc["id"] not in gone]

    return {
        "schema_version": "1.0",
        "meta": {"analysis_id": "sales-" + _now()[:10], "generated_at": _now(), "domain": "sales_orders",
                 "dataset_name": dataset_name, "item_type": "order",
                 "target": {"name": "late_delivery", "definition": "delivered after the estimated (promised) delivery date"},
                 "prediction_moment": "when the order is approved",
                 "data_period": {"start": f"{d['purchased_at'].min():%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"},
                 "current_period": {"start": f"{cutoff:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"},
                 "previous_period": {"start": f"{prev_start:%Y-%m-%d}", "end": f"{cutoff - pd.Timedelta(minutes=1):%Y-%m-%d}"},
                 "currency": "BRL", "timezone": "America/Sao_Paulo"},
        "data_quality": quality_section(quality, int(quality["rows_before"].iloc[0]), len(d), "orders", warnings=[], modules=[
            {"module": "kpis", "enabled": True}, {"module": "root_cause", "enabled": True},
            {"module": "revenue_impact", "enabled": True}]),
        "kpis": kpis, "trends": trends, "breakdowns": breakdowns, "root_causes": root_causes,
        "anomalies": anomalies(sw["late_rate"], "late_rate", "Weekly late delivery rate"),
    }
