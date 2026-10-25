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


def test_days_instead_of_dates_and_text_never_a_date():
    raw = pd.DataFrame({"Order Id": [7, 7, 8], "Customer Id": [1, 1, 2], "Order Status": ["COMPLETE"] * 3,
                        "Order State": ["Lagos", "Lagos", "Delhi"],            # text that looks like "order date"
                        "order date (DateOrders)": ["1/31/2018 22:56", "1/31/2018 22:56", "2/1/2018 10:00"],
                        "Days for shipping (real)": [3, 3, 6], "Days for shipment (scheduled)": [4, 4, 2]})
    s = suggest_mapping(raw.columns, "sales_orders", raw)
    assert s["ready"] and s["mapping"]["order_purchase_timestamp"] == "order date (DateOrders)"
    df = standardise_orders(apply_mapping(raw, s["mapping"], "sales_orders"))
    assert len(df) == 2                                                   # order lines -> one row per order
    late = df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    assert list(late) == [False, True]                                    # 3 days vs 4 promised; 6 vs 2


def test_date_order_us_india_and_unclear():
    from opsinsight.engine.uploads import date_order, to_dates
    us = pd.Series(["05/31/2024 09:00", "06/01/2024 10:30"])
    assert date_order(us) == "month" and to_dates(us).iloc[0] == pd.Timestamp("2024-05-31 09:00")
    india = pd.Series(["31/05/2024", "01/06/2024"])
    assert date_order(india) == "day" and to_dates(india).iloc[1] == pd.Timestamp("2024-06-01")
    unclear = pd.Series(["05/03/2024", "06/04/2024"])
    assert date_order(unclear) == "unclear" and to_dates(unclear, "month").iloc[0] == pd.Timestamp("2024-05-03")
    assert date_order(pd.Series(["2024-05-31"])) == "year"


def test_us_dates_read_month_first_through_the_pipeline():
    from opsinsight.engine.prep_dates import date_check
    raw = pd.DataFrame({"Ticket_ID": ["T1", "T2"], "Created_Date": ["05/31/2024 09:00", "06/02/2024 08:00"],
                        "Resolved_Date": ["06/01/2024 10:00", "06/02/2024 12:00"], "Assigned_Team": ["A", "B"],
                        "Priority": ["High", "Low"], "Issue_Type": ["x", "y"], "SLA_Status": ["Met", "Missed"]})
    s = suggest_mapping(raw.columns, "incidents", raw)
    order, warnings = date_check(raw, s["mapping"])
    assert order == "month" and warnings == []                     # has times, not ambiguous: no warnings
    df = standardise_tickets(apply_mapping(raw, s["mapping"], "incidents"), order)
    assert list(df["opened_at"]) == ["31/05/2024 09:00", "02/06/2024 08:00"]


def test_dates_without_times_use_whole_days_for_sla():
    from opsinsight.engine.prep_dates import NO_TIME, date_check
    from opsinsight.engine.sla import apply_sla_rules
    raw = pd.DataFrame({"id": ["A", "B", "C"], "opened": ["2024-05-01", "2024-05-01", "2024-05-01"],
                        "resolved": ["2024-05-01", "2024-05-02", "2024-05-04"], "prio": ["Critical", "Critical", "Low"]})
    m = {"number": "id", "opened_at": "opened", "resolved_at": "resolved", "priority": "prio"}
    out = apply_sla_rules(raw, m, {"hours": {"1 - Critical": 4, "4 - Low": 72}, "default_hours": 24})
    assert list(out["made_sla"]) == [True, True, True]    # same day, next day (4 h -> 1 day), 3 days (72 h -> 3 days)
    assert date_check(raw, m)[1][0]["message"] == NO_TIME


def test_small_company_gets_an_honest_warning_about_the_borrowed_model():
    from opsinsight.engine.findings import _risk_card
    from opsinsight.engine.predictions import model_section
    from opsinsight.models.config import DOMAINS
    f = pd.read_parquet(DOMAINS["incidents"]["features_file"]).sample(80, random_state=0)
    f["assignment_group_cat"] = "Team Z"                          # teams the reference model has never seen
    mp = model_section("incidents", None, f)
    assert "too little history" in mp["warning"] and mp["on_your_data"]["rows"] == 80
    assert mp["on_your_data"]["known_values_share"] < 1
    assert _risk_card({"model_performance": mp}, "incidents")["confidence"] == "low"


def test_us_state_codes_are_not_turned_into_brazil_regions():
    from opsinsight.prep.sales import to_region
    us = pd.Series(["CA", "PR", "MA", "PA", "NY", "TX"])           # Puerto Rico, Massachusetts... not Brazil
    assert list(to_region(us)) == list(us)
    br = pd.Series(["SP", "RJ", "BA", "PR", "AM"])
    assert list(to_region(br)) == ["Southeast", "Southeast", "Northeast", "South", "North"]


def test_breached_flag_is_read_the_right_way_round():
    from opsinsight.engine.pipeline import standardise_tickets
    raw = pd.DataFrame({"id": ["A", "B"], "opened": ["2024-05-01 09:00"] * 2, "resolved": ["2024-05-01 10:00"] * 2,
                        "SLA Breached": ["Yes", "No"]})
    m = {"number": "id", "opened_at": "opened", "resolved_at": "resolved", "made_sla": "SLA Breached"}
    df = standardise_tickets(apply_mapping(raw, m, "incidents"), "year", flag_means_breach=True)
    assert list(df["made_sla"]) == [False, True]
    assert list(df["priority"]) == ["3 - Moderate"] * 2 and list(df["assignment_group"]) == ["All teams"] * 2


def test_superstore_style_orders_need_only_order_and_ship_dates():
    from opsinsight.engine.pipeline import apply_promise_rules
    raw = pd.DataFrame({"Order ID": [1, 2], "Order Date": ["2024-01-01", "2024-01-02"], "Ship Date": ["2024-01-03", "2024-01-09"],
                        "Ship Mode": ["Same Day", "Standard Class"], "Sales": [10.5, 20.0]})
    s = suggest_mapping(raw.columns, "sales_orders", raw)
    assert s["mapping"]["order_delivered_customer_date"] == "Ship Date"     # shipped stands in for delivered
    assert "order_status" not in s["mapping"]                               # a money column is never a status
    assert s["cannot_run_without"] == []
    out, m = apply_promise_rules(raw, s["mapping"], {"column": "Ship Mode", "days": {"Same Day": 0}, "default_days": 5})
    assert list(out["promised_days"]) == [0, 5] and m["promised_days"] == "promised_days"


def test_region_stays_with_each_order_when_a_customer_orders_to_several_regions():
    from opsinsight.engine.pipeline import run_upload
    from opsinsight.engine.uploads import save_upload
    import numpy as np
    rng = np.random.default_rng(0)
    n = 4000                                                                # enough orders per region for the breakdown
    od = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 300 * 24 * 60, n)), unit="min")
    region = rng.choice(["East", "West", "North"], n)
    actual = np.where(region == "East", 7, 3)                               # East always late, others never
    df = pd.DataFrame({"order_id": range(n), "customer_id": rng.integers(1, 100, n), "order_status": "delivered",
                       "order_date": od.strftime("%Y-%m-%d %H:%M"), "delivered_at": (od + pd.to_timedelta(actual, unit="D")).strftime("%Y-%m-%d %H:%M"),
                       "promised_date": (od + pd.to_timedelta(5, unit="D")).strftime("%Y-%m-%d %H:%M"), "region": region})
    up = save_upload({"orders.csv": df.to_csv(index=False).encode()}, "region-test")
    a = run_upload(up["upload_id"], decide=False)
    rows = {r["value"]: r["late_rate"] for b in a["breakdowns"] if b["dimension"] == "customer_state" for r in b["rows"]}
    assert rows["East"] == 1.0 and rows["West"] == 0.0
