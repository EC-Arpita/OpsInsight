"""Make both forecasts, save them, and put them into the analysis JSON the dashboard reads.

Run from the api folder:   python -m opsinsight.forecast.run
"""
import json

from .common import MODELS_DIR, ROOT
from .sales import forecast_orders
from .tickets import forecast_tickets

CONTRACTS = {
    "incidents": [ROOT / "contracts/example_incidents.json", ROOT / "frontend/public/example_incidents.json"],
    "sales": [ROOT / "contracts/example_sales.json", ROOT / "frontend/public/example_sales.json"],
}


def put_in_contract(domain: str, forecast: dict):
    """Replace only the `forecast` section; every other section stays exactly as it was."""
    for path in CONTRACTS[domain]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["forecast"] = forecast
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for domain, make, unit in [("incidents", forecast_tickets, "teams"), ("sales", forecast_orders, "regions")]:
        fc = make()
        (MODELS_DIR / f"forecast_{domain}.json").write_text(json.dumps(fc, indent=2))
        put_in_contract(domain, fc)
        print(f"\n=== {domain}: chosen method = {fc['method']}")
        for r in fc["backtest_mean_abs_error"]:
            print(f"  {r['method']:32s} {r['mean_abs_error']}")
        if fc.get("note"):
            print("  note:", fc["note"])
        for g in fc[unit]:
            name = g.get("team") or g.get("region")
            flags = g.get("overloaded_days", g.get("busy_days"))
            total = g.get("expected_tickets_7d", g.get("expected_orders_7d"))
            print(f"  {name:13s} next 7 days: {total:5d}   flagged days: {flags}")
    print("\nSaved models/forecast_incidents.json, models/forecast_sales.json and updated the analysis JSON files")


if __name__ == "__main__":
    main()