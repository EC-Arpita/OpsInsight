"""Both forecasts beat a naive average and follow their own rules. Run from the api folder:  python -m pytest tests -q"""
import pandas as pd

from opsinsight.forecast.sales import ORDERS_FILE, forecast_orders, incomplete_tail, region_counts
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
            if abs(d["expected_tickets"] - limit) > 0.2 and abs(d["expected_tickets"] - d["expected_capacity"] - 1) > 0.2:
                assert d["overload"] == (d["expected_tickets"] > limit and d["expected_tickets"] - d["expected_capacity"] >= 1)


def test_capacity_follows_the_weekday():
    """A team clears far fewer tickets on Sundays than on Mondays, so capacity must differ by weekday."""
    big = forecast_tickets()["teams"][0]
    caps = {pd.Timestamp(d["date"]).dayofweek: d["expected_capacity"] for d in big["daily"]}
    assert caps[0] > 3 * caps[6]
    assert big["expected_capacity_7d"] == round(sum(d["expected_capacity"] for d in big["daily"]))


def test_sales_forecast_skips_the_incomplete_tail():
    fc = forecast_orders()
    orders = pd.read_parquet(ORDERS_FILE)
    tail = incomplete_tail(region_counts(orders))
    assert 7 <= tail <= 28                                     # Olist's export thins out in its last weeks
    assert pd.Timestamp(fc["history_end"]) == orders["purchased_at"].max().normalize() - pd.Timedelta(days=tail)
    assert len(fc["regions"]) == 5 and all(len(r["daily"]) == 7 for r in fc["regions"])
