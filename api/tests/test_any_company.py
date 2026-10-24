"""Other companies' files: simple ticket exports and single orders files. Run from the api folder: pytest tests -v"""
import pandas as pd

from opsinsight.engine.pipeline import standardise_orders, standardise_tickets
from opsinsight.prep.mapping import apply_mapping, suggest_mapping


def test_simple_ticket_export_is_accepted():
    cols = ["Ticket_ID", "Created_Date", "Resolved_Date", "Assigned_Team", "Priority", "Issue_Type", "SLA_Status"]
    s = suggest_mapping(cols, "incidents")
    assert s["ready"], s["missing_required"]               # no event-log columns needed any more
    assert s["mapping"]["made_sla"] == "SLA_Status"


def test_tickets_dates_sla_words_and_priority_words():
    raw = pd.DataFrame({"Ticket_ID": ["T1", "T2"], "Created_Date": ["2025-02-07", "2025-02-08"],
                        "Resolved_Date": ["2025-02-09", "2025-02-08"], "Assigned_Team": ["A", "B"],
                        "Priority": ["High", "1 - Critical"], "Issue_Type": ["x", "y"], "SLA_Status": ["Missed", "Met"]})
    s = suggest_mapping(raw.columns, "incidents")
    df = standardise_tickets(apply_mapping(raw, s["mapping"], "incidents"))
    assert list(df["made_sla"]) == [False, True]
    assert list(df["opened_at"]) == ["07/02/2025 00:00", "08/02/2025 00:00"]
    assert list(df["priority"]) == ["2 - High", "1 - Critical"]


def test_orders_status_words_and_region():
    raw = pd.DataFrame({"order_id": [1, 2], "customer_id": [10, 20], "status": ["COMPLETE", "CANCELED"],
                        "order_date": ["2018-01-31 22:56", "2018-02-01 10:00"], "delivered_at": ["2018-02-03", None],
                        "promised_date": ["2018-02-04", "2018-02-05"], "region": ["South Asia", "Oceania"]})
    s = suggest_mapping(raw.columns, "sales_orders")
    assert s["ready"] and s["mapping"]["customer_state"] == "region"
    df = standardise_orders(apply_mapping(raw, s["mapping"], "sales_orders"))
    assert list(df["order_status"]) == ["delivered", "canceled"]
    assert df["order_purchase_timestamp"].iloc[0] == pd.Timestamp("2018-01-31 22:56")


def test_small_file_falls_back_to_reference_model():
    from opsinsight.engine.pipeline import own_model
    tiny = pd.DataFrame({"split": ["train"] * 10 + ["test"] * 5, "breached": [0, 1] * 7 + [0]})
    assert own_model("incidents", tiny, "tiny-upload") is None   # too small to learn from