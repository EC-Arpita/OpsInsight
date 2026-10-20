"""Make small demo files in the SAME format as the raw downloads, for the live demo."""
from pathlib import Path

import pandas as pd

from .sales import FILES


def make_incident_demo(raw_csv, out_csv, n_tickets=6000, last_opened_before="2016-06-01"):
    """Keep every event row of the newest n_tickets tickets (opened before the cut-off date).
    The rows are copied unchanged, so the demo file looks exactly like the original download."""
    raw = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)       # read as plain text: nothing changes
    opened = pd.to_datetime(raw["opened_at"], format="%d/%m/%Y %H:%M", errors="coerce")
    first_open = opened.groupby(raw["number"]).min()
    keep = first_open[first_open < pd.Timestamp(last_opened_before)].sort_values().tail(n_tickets).index
    demo = raw[raw["number"].isin(keep)]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(out_csv, index=False)
    return {"tickets": int(demo["number"].nunique()), "rows": len(demo),
            "from": str(first_open[keep].min()), "to": str(first_open[keep].max())}


def make_sales_demo(raw_dir, out_dir, start="2018-05-01", end="2018-09-01", points_per_zip=5):
    """Keep orders purchased in [start, end) plus only the rows of the other 8 files they need.
    Geolocation is 1 million rows, so only a few points per zip are kept (enough for the median)."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    t = {name: pd.read_csv(raw_dir / f, dtype=str, keep_default_na=False) for name, f in FILES.items()}
    when = pd.to_datetime(t["orders"]["order_purchase_timestamp"], errors="coerce")
    orders = t["orders"][(when >= start) & (when < end)]
    ids = set(orders["order_id"])
    items = t["items"][t["items"]["order_id"].isin(ids)]
    customers = t["customers"][t["customers"]["customer_id"].isin(set(orders["customer_id"]))]
    sellers = t["sellers"][t["sellers"]["seller_id"].isin(set(items["seller_id"]))]
    zips = set(customers["customer_zip_code_prefix"]) | set(sellers["seller_zip_code_prefix"])
    demo = {
        "orders": orders, "items": items,
        "payments": t["payments"][t["payments"]["order_id"].isin(ids)],
        "reviews": t["reviews"][t["reviews"]["order_id"].isin(ids)],
        "customers": customers, "sellers": sellers,
        "products": t["products"][t["products"]["product_id"].isin(set(items["product_id"]))],
        "geolocation": (t["geolocation"][t["geolocation"]["geolocation_zip_code_prefix"].isin(zips)]
                        .groupby("geolocation_zip_code_prefix").head(points_per_zip)),
        "translation": t["translation"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in demo.items():
        df.to_csv(out_dir / FILES[name], index=False)
    return {name: len(df) for name, df in demo.items()}