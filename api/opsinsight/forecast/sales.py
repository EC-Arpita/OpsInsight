"""Sales orders: expected orders per customer region for the next 7 days, and which days will be unusually busy."""
import pandas as pd

from ..prep.sales import to_region
from .common import HORIZON, METHODS, ROOT, backtest, daily_counts

ORDERS_FILE = ROOT / "data/interim/sales_orders_clean.parquet"
MAX_TAIL_DAYS = 28         # never cut more than 4 weeks, even if volume really fell


def region_counts(orders: pd.DataFrame) -> pd.DataFrame:
    """Orders placed per customer region per day (every order, delivered or not)."""
    orders = orders.assign(customer_region=to_region(orders["customer_state"]).fillna("Unknown"))
    regions = list(orders["customer_region"].value_counts().index.drop("Unknown", errors="ignore"))
    return daily_counts(orders, "purchased_at", "customer_region", regions)


def tail_check(counts: pd.DataFrame) -> pd.DataFrame:
    """Each region's last 7 days compared with its usual 7 days (1.0 = normal). Shows where the file goes incomplete."""
    week = counts.rolling(7).sum()
    usual = week.shift(7).rolling(28).median()
    return (week / usual).round(2)


def incomplete_tail(counts: pd.DataFrame, threshold=0.9, min_week=70) -> int:
    """Days at the end of the file where a sizeable region gets clearly fewer orders than in a normal week:
    orders to far regions reach the export late. 0 when the end of the file looks complete."""
    usual = counts.rolling(7).sum().shift(7).rolling(28).median()
    big = [c for c in counts.columns if usual[c].iloc[-1] >= min_week] or list(counts.columns)
    worst = tail_check(counts)[big].min(axis=1)
    ok = worst[worst >= threshold]
    return 0 if ok.empty else min(MAX_TAIL_DAYS, int((counts.index[-1] - ok.index[-1]).days))


def forecast_orders(orders: pd.DataFrame = None, busy_margin=1.2, min_extra=10) -> dict:
    """The `forecast` section of the sales analysis JSON."""
    orders = pd.read_parquet(ORDERS_FILE) if orders is None else orders
    full = region_counts(orders)
    tail = incomplete_tail(full)
    placed = full.iloc[:len(full) - tail]
    regions = list(placed.columns)

    scores = backtest(placed)
    best = scores.iloc[0]["method"]
    future = METHODS[best](placed)
    last4 = placed.iloc[-28:]
    usual_by_day = last4.groupby(last4.index.dayofweek).mean()   # a normal Monday, a normal Sunday... over 4 weeks

    days = pd.date_range(placed.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    usual = usual_by_day.reindex(days.dayofweek).set_axis(days)
    out = []
    for j, region in enumerate(regions):
        daily = [{"date": f"{d:%Y-%m-%d}", "expected_orders": round(float(future[i, j]), 1),
                  "usual_orders": round(float(usual.at[d, region]), 1),
                  "busy": bool(future[i, j] > usual.at[d, region] * busy_margin and future[i, j] - usual.at[d, region] >= min_extra)}
                 for i, d in enumerate(days)]
        out.append({"region": region, "expected_orders_7d": round(float(future[:, j].sum())),
                    "usual_orders_7d": round(float(usual[region].sum())),
                    "busy_days": sum(x["busy"] for x in daily), "daily": daily})
    return {"method": best, "horizon_days": HORIZON, "backtest_mean_abs_error": scores.to_dict(orient="records"),
            "history_end": f"{placed.index[-1]:%Y-%m-%d}",
            "note": (f"The last {tail} days of the file were left out: orders to distant regions reach "
                     "the export late, so those days look falsely quiet. The forecast starts after the last complete day."
                     if tail else None),
            "busy_rule": f"expected orders > the usual for that weekday (last 4 weeks) x {busy_margin}, and at least {min_extra} orders more",
            "regions": out}
