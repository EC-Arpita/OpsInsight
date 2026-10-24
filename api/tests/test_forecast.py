"""Both forecasts beat a naive average and follow their own rules. Run from the api folder:  python -m pytest tests -q"""
import pandas as pd

from opsinsight.forecast.sales import LEAVE_OUT_LAST_DAYS, ORDERS_FILE, forecast_orders
from opsinsight.forecast.tickets import forecast_tickets


def test_ticket_forecast_beats_flat_average():
    fc = forecast_tickets()
    errors = {r["method"]: r["mean_abs_error"] for r in fc["backtest_mean_abs_error"]}
    assert errors[fc["method"]] < errors["flat 4-week average"]


def test_overload_follows_the_rule():
    """Overloaded = expected tickets above capacity x 1.1 (numbers in the file are rounded, so skip exact ties)."""
    for team in forecast_tickets()["teams"]:
        for d in team["daily"]:
            limit = d["expected_capacity"] * 1.1
            if abs(d["expected_tickets"] - limit) > 0.2:
                assert d["overload"] == (d["expected_tickets"] > limit)


def test_sales_forecast_skips_the_incomplete_tail():
    fc = forecast_orders()
    last_order_day = pd.read_parquet(ORDERS_FILE)["purchased_at"].max().normalize()
    assert pd.Timestamp(fc["history_end"]) == last_order_day - pd.Timedelta(days=LEAVE_OUT_LAST_DAYS)
    assert len(fc["regions"]) == 5 and all(len(r["daily"]) == 7 for r in fc["regions"])