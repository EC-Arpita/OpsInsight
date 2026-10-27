"""Domain detection and routing never guess. Run from the api folder:  python -m pytest tests -q"""
import pandas as pd

from opsinsight.models.route import choose_model
from opsinsight.prep.detect import detect_domain, detect_shape

TICKET_COLS = ["Ticket No.", "Created Date", "Last Updated", "Update Number", "SLA Met", "Team",
               "Priority", "Issue Type", "Resolved Date", "Closed Date", "Status", "Channel"]
ORDER_COLS = ["Order Number", "Client ID", "Status", "Order Date", "Paid At", "Ship Date", "Delivered At", "ETA"]


def frame(cols, rows=3, repeat_ids=1):
    """A tiny table with these column names; the first column is the ID."""
    data = {c: [f"2024-01-{i % 28 + 1:02d}" for i in range(rows)] for c in cols}   # dates, so date columns pass the value check
    data[cols[0]] = [f"id{i // repeat_ids}" for i in range(rows)]
    return pd.DataFrame(data)


def test_renamed_ticket_file_is_tickets():
    d = detect_domain(TICKET_COLS)
    assert d["domain"] == "incidents" and d["decision"] == "confident" and d["best_score"] >= 0.9


def test_renamed_order_file_is_orders():
    d = detect_domain(ORDER_COLS)
    assert d["domain"] == "sales_orders" and d["decision"] == "confident"


def test_unrelated_file_is_refused():
    d = detect_domain(["product_id", "product_weight_g", "product_length_cm", "product_photos_qty"])
    assert d["decision"] == "unknown" and d["domain"] is None


def test_event_log_is_recognised():
    df = frame(TICKET_COLS, rows=12, repeat_ids=4)          # 3 tickets, 4 rows each
    assert detect_shape(df, "Ticket No.")["shape"] == "event_log"


def test_routing_answers():
    assert choose_model(frame(TICKET_COLS))["model"] == "incidents"
    assert choose_model(frame(ORDER_COLS))["model"] == "sales"
    assert choose_model(frame(["product_id", "product_weight_g", "product_length_cm"]))["status"] == "refused"
    missing = choose_model(frame(TICKET_COLS[:6]))                 # no priority, category, resolved, closed
    assert missing["status"] in ("cannot_run", "needs_confirmation") and missing["model"] is None

def test_days_columns_count_fully_in_the_match_score():
    cols = ["Order No", "Client ID", "Order Status", "Order Datetime", "Delivery Days", "SLA Days", "Delivery Region"]
    d = detect_domain(cols)
    assert d["domain"] == "sales_orders" and d["decision"] == "confident" and d["best_score"] >= 0.95
