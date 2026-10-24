"""Sales orders: expected orders per customer region for the next 7 days, and which days will be unusually busy."""
import pandas as pd

from ..prep.sales import REGION
from .common import HORIZON, METHODS, ROOT, backtest, daily_counts

ORDERS_FILE = ROOT / "data/interim/sales_orders_clean.parquet"
LEAVE_OUT_LAST_DAYS = 21  # orders to far regions reach the export late, so the newest 3 weeks are incomplete


def region_counts(orders: pd.DataFrame) -> pd.DataFrame:
    """Orders placed per customer region per day (every order, delivered or not)."""
    orders = orders.assign(customer_region=orders["customer_state"].map(REGION).fillna(orders["customer_state"]).fillna("Unknown"))
    regions = list(orders["customer_region"].value_counts().index.drop("Unknown", errors="ignore"))
    return daily_counts(orders, "purchased_at", "customer_region", regions)


def tail_check(counts: pd.DataFrame) -> pd.DataFrame:
    """Each region's last 7 days compared with its usual 7 days (1.0 = normal). Shows where the file goes incomplete."""
    week = counts.rolling(7).sum()
    usual = week.shift(7).rolling(28).median()
    return (week / usual).round(2)



def forecast_orders(orders: pd.DataFrame = None, busy_margin=1.2, min_extra=10) -> dict:
    """The `forecast` section of the sales analysis JSON."""
    orders = pd.read_parquet(ORDERS_FILE) if orders is None else orders
    full = region_counts(orders)
    placed = full.iloc[:-LEAVE_OUT_LAST_DAYS]
    regions = list(placed.columns)

    scores = backtest(placed)
    best = scores.iloc[0]["method"]
    future = METHODS[best](placed)
    usual = placed.iloc[-28:].mean()  # average orders per day over the last 4 weeks

    days = pd.date_range(placed.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    out = []
    for j, region in enumerate(regions):
        daily = [
            {
                "date": f"{d:%Y-%m-%d}",
                "expected_orders": round(float(future[i, j]), 1),
                "usual_orders": round(float(usual[region]), 1),
                "busy": bool(future[i, j] > usual[region] * busy_margin and future[i, j] - usual[region] >= min_extra),
            }
            for i, d in enumerate(days)
        ]
        out.append(
            {
                "region": region,
                "expected_orders_7d": round(float(future[:, j].sum())),
                "usual_orders_7d": round(float(usual[region] * 7)),
                "busy_days": sum(x["busy"] for x in daily),
                "daily": daily,
            }
        )
    return {
        "method": best,
        "horizon_days": HORIZON,
        "backtest_mean_abs_error": scores.to_dict(orient="records"),
        "history_end": f"{placed.index[-1]:%Y-%m-%d}",
        "note": (
            f"The last {LEAVE_OUT_LAST_DAYS} days of the file were left out: orders to distant regions reach "
            "the export late, so those days look falsely quiet. The forecast starts after the last complete day."
        ),
        "busy_rule": f"expected orders > last 4-week daily average x {busy_margin}, and at least {min_extra} orders more",
        "regions": out,
    }