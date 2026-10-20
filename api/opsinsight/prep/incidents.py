"""Incident (support ticket) pipeline: raw event log -> clean tickets -> model features.

Same logic as notebooks 02 and 03, packaged as functions the app can call.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FeatureDictionary, QualityLog, count_in_window, past_mean, top_n_or_other

DATE_COLS = ["opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at"]
DATE_FORMAT = "%d/%m/%Y %H:%M"
EMPTY_COLS = ["caused_by", "vendor", "cmdb_ci", "rfc", "problem_id"]
DEFAULT_WINDOW = ("2016-02-29", "2016-06-01")   # main period of the UCI dataset


def clean_events(raw: pd.DataFrame, window=DEFAULT_WINDOW, log: QualityLog = None) -> pd.DataFrame:
    """Clean the raw event log (one row per ticket update). Notebook 02, steps 2-9."""
    log = log or QualityLog()
    df = raw.copy()
    log.add("load raw file", len(df), len(df), f"{df['number'].nunique():,} tickets")

    # broken state value
    before = len(df)
    df = df[df["incident_state"] != "-100"]
    log.add("drop incident_state == -100", before, len(df))

    # exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.add("drop exact duplicate rows", before, len(df))

    # text -> real dates and numbers
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], format=DATE_FORMAT, errors="coerce")
    for c in ["impact", "urgency", "priority"]:
        df[f"{c}_n"] = df[c].str.extract(r"(\d+)", expand=False).astype(float)
    df["notify_email"] = df["notify"].eq("Send Email")
    df = df.sort_values(["number", "sys_mod_count", "sys_updated_at"]).reset_index(drop=True)

    # keep the main time window
    if window:
        before, t_before = len(df), df["number"].nunique()
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        df = df[(df["opened_at"] >= start) & (df["opened_at"] < end)]
        log.add(f"keep tickets opened {window[0]} to {window[1]}", before, len(df),
                f"tickets {t_before:,} -> {df['number'].nunique():,}")

    # impossible timelines
    bad = df.loc[(df["resolved_at"] < df["opened_at"]) | (df["closed_at"] < df["opened_at"]), "number"].unique()
    before = len(df)
    df = df[~df["number"].isin(bad)].copy()
    log.add("drop tickets with impossible timelines", before, len(df), f"{len(bad)} tickets")

    # almost-empty columns -> yes/no flags, then drop them
    for c in EMPTY_COLS:
        df[f"has_{c}"] = df[c].notna()
    return df.drop(columns=EMPTY_COLS + ["notify"]).reset_index(drop=True)

TEXT_COLS = ["caller_id", "opened_by", "contact_type", "location", "category", "subcategory",
             "u_symptom", "assignment_group", "final_assignment_group", "final_closed_code"]


def collapse_to_tickets(events: pd.DataFrame, log: QualityLog = None) -> pd.DataFrame:
    """One row per ticket: first event = known at opening, last event = outcome. Notebook 02, steps 10-12."""
    log = log or QualityLog()
    # drop_duplicates keeps the REAL first/last row; groupby().first() would mix values from later rows
    first = events.drop_duplicates("number", keep="first").set_index("number")
    last = events.drop_duplicates("number", keep="last").set_index("number")

    at_open = ["opened_at", "caller_id", "opened_by", "contact_type", "location", "category",
               "subcategory", "u_symptom", "impact_n", "urgency_n", "priority_n", "assignment_group",
               "knowledge", "u_priority_confirmation", "notify_email", "has_cmdb_ci", "has_problem_id",
               "has_rfc", "has_vendor", "has_caused_by", "incident_state"]
    t = first[at_open].rename(columns={"incident_state": "first_state_seen"})

    # outcomes: for the target and root-cause analysis only, never model inputs
    t["final_made_sla"] = last["made_sla"]
    t["final_reassignment_count"] = last["reassignment_count"]
    t["final_reopen_count"] = last["reopen_count"]
    t["final_sys_mod_count"] = last["sys_mod_count"]
    t["final_assignment_group"] = last["assignment_group"]
    t["final_closed_code"] = last["closed_code"]
    t["resolved_at"] = last["resolved_at"]
    t["closed_at"] = last["closed_at"]
    t["n_events"] = events.groupby("number").size()

    # THE TARGET: 1 = breached SLA, 0 = made SLA
    t["breached"] = (~t["final_made_sla"].astype(bool)).astype(int)
    t["hours_to_resolve"] = (t["resolved_at"] - t["opened_at"]).dt.total_seconds() / 3600
    t = t.reset_index()
    t[TEXT_COLS] = t[TEXT_COLS].fillna("Unknown")

    if not t["number"].is_unique:
        raise ValueError("duplicate tickets after collapsing")
    log.add("final: one row per ticket", len(events), len(t), "events collapsed into tickets")
    return t

CATEGORY_PLAN = {"contact_type": 5, "category": 30, "subcategory": 50, "u_symptom": 50,
                 "location": 30, "assignment_group": 30, "opened_by": 30}


def build_features(tickets: pd.DataFrame, test_share=0.2, m=20):
    """Model-ready features, all known when the ticket is opened. Notebook 03.
    Returns (features_df, FeatureDictionary)."""
    t = tickets.sort_values("opened_at").reset_index(drop=True).copy()
    fd = FeatureDictionary("ticket_opening")

    # 1. time-based split, decided first
    cutoff = t["opened_at"].quantile(1 - test_share)
    t["split"] = np.where(t["opened_at"] < cutoff, "train", "test")
    train = t["split"] == "train"

    # 2. time
    t["open_hour"] = t["opened_at"].dt.hour
    t["open_weekday"] = t["opened_at"].dt.dayofweek
    t["is_weekend"] = (t["open_weekday"] >= 5).astype(int)
    t["is_business_hours"] = (t["open_hour"].between(8, 17) & (t["is_weekend"] == 0)).astype(int)
    fd.add("open_hour", "hour of opened_at (0-23)", "time")
    fd.add("open_weekday", "day of week of opened_at (0=Mon)", "time")
    fd.add("is_weekend", "1 if opened on Sat/Sun", "time")
    fd.add("is_business_hours", "1 if opened Mon-Fri 08:00-17:59", "time")

    # 3. severity
    t["impact_x_urgency"] = t["impact_n"] * t["urgency_n"]
    for c, how in [("priority_n", "priority number, 1=Critical 4=Low"), ("impact_n", "impact number, 1=High"),
                   ("urgency_n", "urgency number, 1=High"), ("impact_x_urgency", "impact_n * urgency_n")]:
        fd.add(c, how, "severity")

    # 4. flags
    for c in ["knowledge", "u_priority_confirmation", "notify_email", "has_cmdb_ci", "has_problem_id"]:
        t[c] = t[c].astype(int)
        fd.add(c, f"1 if {c} was set on the first event", "flags")

    # 5. workload: how busy was the team when the ticket arrived?
    t["_done_at"] = t["resolved_at"].fillna(t["closed_at"])
    for c in ["group_opened_last_24h", "group_opened_last_7d", "group_backlog_at_open"]:
        t[c] = 0
    for _, idx in t.groupby("assignment_group").groups.items():
        sub = t.loc[idx]
        opens, dones, q = np.sort(sub["opened_at"].values), np.sort(sub["_done_at"].values), sub["opened_at"].values
        t.loc[idx, "group_opened_last_24h"] = count_in_window(opens, q, np.timedelta64(24, "h"))
        t.loc[idx, "group_opened_last_7d"] = count_in_window(opens, q, np.timedelta64(7, "D"))
        t.loc[idx, "group_backlog_at_open"] = np.maximum(
            np.searchsorted(opens, q, side="left") - np.searchsorted(dones, q, side="right"), 0)
    t["all_opened_last_24h"] = count_in_window(np.sort(t["opened_at"].values), t["opened_at"].values,
                                               np.timedelta64(24, "h"))
    fd.add("group_opened_last_24h", "tickets the same assignment_group received in the previous 24h", "workload")
    fd.add("group_opened_last_7d", "tickets the same assignment_group received in the previous 7 days", "workload")
    fd.add("group_backlog_at_open", "tickets of the same group opened earlier and not yet resolved", "workload")
    fd.add("all_opened_last_24h", "tickets the whole helpdesk received in the previous 24h", "workload")

    # 6. history: only outcomes already known when this ticket opened
    prior = t.loc[train, "breached"].mean()
    for key in ["assignment_group", "category", "location", "caller_id"]:
        t[f"{key}_past_breach_rate"], _ = past_mean(t, key, "breached", "_done_at", "opened_at", prior, m)
        fd.add(f"{key}_past_breach_rate",
               f"smoothed breach rate of earlier tickets with the same {key} resolved before this one opened (m={m})",
               "history")
    week = np.timedelta64(7, "D")
    t["recent_breach_rate_7d"], t["recent_resolved_7d"] = past_mean(t, None, "breached", "_done_at", "opened_at",
                                                                    prior, m, window=week)
    t["group_recent_breach_rate_7d"], _ = past_mean(t, "assignment_group", "breached", "_done_at", "opened_at",
                                                    prior, m, window=week)
    fd.add("recent_breach_rate_7d", "smoothed breach rate of ALL tickets resolved in the 7 days before this one opened", "history")
    fd.add("recent_resolved_7d", "number of tickets resolved helpdesk-wide in the 7 days before opening", "history")
    fd.add("group_recent_breach_rate_7d", "same as recent_breach_rate_7d but only for the same assignment_group", "history")

    t["_log_hours"] = np.log1p(t["hours_to_resolve"])
    prior_hours = t.loc[train, "_log_hours"].mean()
    valid = t["_log_hours"].notna()
    t["category_past_log_hours"] = np.nan
    t.loc[valid, "category_past_log_hours"] = past_mean(t[valid], "category", "_log_hours", "_done_at",
                                                        "opened_at", prior_hours, m)[0]
    t["category_past_log_hours"] = t["category_past_log_hours"].fillna(prior_hours)
    fd.add("category_past_log_hours", "smoothed mean log(1+hours to resolve) of earlier resolved tickets in the same category", "history")

    t["caller_tickets_last_30d"] = 0
    for _, idx in t.groupby("caller_id").groups.items():
        opens = np.sort(t.loc[idx, "opened_at"].values)
        t.loc[idx, "caller_tickets_last_30d"] = count_in_window(opens, t.loc[idx, "opened_at"].values,
                                                                np.timedelta64(30, "D"))
    fd.add("caller_tickets_last_30d", "tickets the same caller opened in the previous 30 days", "history")

    # 7. text categories -> top N + "Other" (learned on training rows only)
    for col, n in CATEGORY_PLAN.items():
        t[f"{col}_cat"] = top_n_or_other(t, col, n, train)
        fd.add(f"{col}_cat", f"{col}, top {n} values from training period, rest = 'Other'", "category")

    # 8. final table + leakage guard
    features = t[["number", "opened_at", "split"] + fd.names + ["breached"]].copy()
    leaky = [c for c in features.columns if c.startswith("final_") or c in ("hours_to_resolve", "resolved_at", "closed_at")]
    if leaky:
        raise ValueError(f"leaky columns in features: {leaky}")
    return features, fd

def prepare_incidents(csv_path, window=DEFAULT_WINDOW, test_share=0.2, save_to=None):
    """Run the whole incident pipeline on one CSV file.

    csv_path : path to incident_event_log.csv (missing values written as '?')
    window   : (start, end) dates to keep, or None to keep everything
    save_to  : project 'data' folder; if given, saves interim/ and processed/ parquet files
    Returns a dict with events, tickets, features, quality, feature_dictionary.
    """
    log = QualityLog()
    raw = pd.read_csv(csv_path, na_values="?", low_memory=False)
    events = clean_events(raw, window=window, log=log)
    tickets = collapse_to_tickets(events, log=log)
    features, fd = build_features(tickets, test_share=test_share)
    out = {"events": events, "tickets": tickets, "features": features,
           "quality": log.to_frame(), "feature_dictionary": fd.to_frame()}
    if save_to:
        data = Path(save_to)
        (data / "interim").mkdir(parents=True, exist_ok=True)
        (data / "processed").mkdir(parents=True, exist_ok=True)
        events.to_parquet(data / "interim/incident_events_clean.parquet", index=False)
        tickets.to_parquet(data / "interim/incidents_clean.parquet", index=False)
        features.to_parquet(data / "processed/incidents_features.parquet", index=False)
    return out