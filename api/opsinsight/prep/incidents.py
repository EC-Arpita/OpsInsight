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