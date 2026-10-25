"""SLA rules: when a ticket file has no 'SLA met' column, work it out from rules the analyst types in.

rules = {"hours": {"1 - Critical": 4, "2 - High": 8, "3 - Moderate": 24, "4 - Low": 72}, "default_hours": 24}
A ticket met its SLA if it was resolved within the hours allowed for its priority.
"""
import math

import pandas as pd

from .uploads import has_times, to_dates


def apply_sla_rules(df: pd.DataFrame, mapping: dict, rules: dict, order: str = None) -> pd.DataFrame:
    """Return a copy of the uploaded table with a new 'made_sla' column (same value on every row of a ticket).

    mapping : the confirmed mapping {standard name: company's column}; needs number, opened_at,
              resolved_at and priority.
    """
    col = {k: mapping[k] for k in ("number", "opened_at", "resolved_at", "priority")}
    out = df.copy()
    opened = to_dates(out[col["opened_at"]], order)
    resolved = to_dates(out[col["resolved_at"]], order)
    per_ticket = pd.DataFrame({"id": out[col["number"]], "opened": opened, "resolved": resolved,
                               "priority": out[col["priority"]].astype(str)})
    t = per_ticket.groupby("id").agg(opened=("opened", "min"), resolved=("resolved", "max"), priority=("priority", "first"))
    by_word = {k.split("-")[-1].strip().lower(): v for k, v in rules["hours"].items()}   # "4 - Low" also matches "Low"
    by_word.update({"medium": by_word.get("moderate"), "urgent": by_word.get("critical")})
    allowed = (t["priority"].map(rules["hours"]).fillna(t["priority"].str.strip().str.lower().map(by_word))
               .fillna(rules.get("default_hours", 24)).astype(float))
    end_of_data = per_ticket["opened"].max()
    if has_times(opened) or has_times(resolved):
        took = (t["resolved"].fillna(end_of_data) - t["opened"]).dt.total_seconds() / 3600   # still open: time so far
        t["made_sla"] = took <= allowed
    else:   # dates without a time of day: compare whole days, e.g. 4 or 8 hours allowed = resolved by the next day
        days = (t["resolved"].fillna(end_of_data) - t["opened"]).dt.days
        t["made_sla"] = days <= allowed.map(lambda h: math.ceil(h / 24))
    out["made_sla"] = out[col["number"]].map(t["made_sla"])
    return out


def sla_summary(df_with_sla: pd.DataFrame, number_col: str) -> dict:
    """What the rules produced, for the analyst to check before the analysis runs."""
    t = df_with_sla.groupby(number_col)["made_sla"].first()
    return {"tickets": int(len(t)), "breach_rate": round(float(1 - t.mean()), 3)}
