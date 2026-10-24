"""Add Roshan's decision sections to the two example analysis files the dashboard reads.

Adds: ranked_findings, priority_queue, workload_plan, recommendations (engine version), impact_simulation.
Run from the api folder AFTER Abhishek's refresh:   python -m opsinsight.engine.refresh
"""
import json

import pandas as pd

from ..models.config import ROOT
from .decide import add_decisions

FILES = {"incidents": ["contracts/example_incidents.json", "frontend/public/example_incidents.json"],
         "sales": ["contracts/example_sales.json", "frontend/public/example_sales.json"]}
TABLES = {"incidents": {"tickets": "data/interim/incidents_clean.parquet", "features": "data/processed/incidents_features.parquet"},
          "sales": {"orders": "data/interim/sales_orders_clean.parquet", "features": "data/processed/sales_orders_features.parquet"}}


def main():
    for domain, files in FILES.items():
        data = json.loads((ROOT / files[0]).read_text(encoding="utf-8"))
        tables = {name: pd.read_parquet(ROOT / path) for name, path in TABLES[domain].items()}
        add_decisions(data, domain, tables)
        for f in files:
            if (ROOT / f).parent.exists():              # the frontend folder is only there once Divesh's work is merged
                (ROOT / f).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        recs = ", ".join(f"{r['id']} {r['title'][:40]}" for r in data["recommendations"])
        print(f"{domain}: {len(data['ranked_findings']['items'])} ranked findings | {recs}")


if __name__ == "__main__":
    main()
