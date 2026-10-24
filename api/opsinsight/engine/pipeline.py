"""Run the whole analysis on one upload: map -> (SLA rules) -> clean -> features -> analytics ->
predictions -> forecast -> decisions. Saves data/app/analysis/<upload id>.json."""
import json
from pathlib import Path

import pandas as pd

from ..forecast.sales import forecast_orders
from ..forecast.tickets import forecast_tickets
from ..models.config import ROOT
from ..prep import incidents as inc
from ..prep import sales as sal
from ..prep.analytics import incident_analysis, sales_analysis
from ..prep.common import QualityLog
from ..prep.mapping import apply_mapping
from . import store
from .predictions import model_section, predictions_section
from .sla import apply_sla_rules
from .uploads import UploadError, read_table, to_dates

ANALYSIS_DIR = ROOT / "data/app/analysis"


def auto_window(dates: pd.Series, share=0.10) -> tuple:
    """Keep the months with real volume: at least `share` of the busiest month. Near-empty months at the
    edges of a file (a few stray tickets, an export that stopped early) would distort trends and forecasts."""
    d = to_dates(dates).dropna()
    months = d.dt.to_period("M").value_counts().sort_index()
    busy = months[months >= share * months.max()].index
    return f"{busy.min().start_time:%Y-%m-%d}", f"{(busy.max() + 1).start_time:%Y-%m-%d}"


def _mapping(upload: dict, overrides: dict | None) -> dict:
    mapping = dict(upload["report"]["mapping"] or {})
    mapping.update(overrides or {})                       # the analyst's corrections win
    return {k: v for k, v in mapping.items() if v}       # an empty value means "not in my file"


def run_tickets(upload, mapping, sla_rules=None) -> tuple:
    raw = read_table(Path(upload["folder"]) / upload["main_file"])
    if "made_sla" not in mapping:
        if not sla_rules:
            raise UploadError("This file has no 'SLA met' column. Send SLA rules (hours allowed per priority).")
        raw = apply_sla_rules(raw, mapping, sla_rules)
        mapping = {**mapping, "made_sla": "made_sla"}
    df = apply_mapping(raw, mapping, "incidents")
    log = QualityLog()
    first_rows = df.drop_duplicates("number")["opened_at"]      # one opening time per ticket
    events = inc.clean_events(df, window=auto_window(first_rows), log=log)
    tickets = inc.collapse_to_tickets(events, log=log)
    features, _ = inc.build_features(tickets)
    analysis = incident_analysis(tickets, features, log.to_frame(), f"{upload['company']} support tickets")
    analysis["predictions"] = predictions_section("incidents", tickets, features)
    analysis["model_performance"] = model_section("incidents")
    analysis["forecast"] = forecast_tickets(tickets)
    return analysis, {"tickets": tickets, "features": features}


def run_sales(upload, mapping) -> tuple:
    folder = Path(upload["folder"])
    orders_raw = read_table(folder / upload["main_file"])
    work = folder / "standard"                             # Olist file names, the format the sales pipeline reads
    work.mkdir(exist_ok=True)
    apply_mapping(orders_raw, mapping, "sales_orders").to_csv(work / sal.FILES["orders"], index=False)
    for key, name in sal.FILES.items():
        if key != "orders":
            if not (folder / name).exists():
                raise UploadError(f"Sales uploads need all 9 files; {name} is missing.")
            (work / name).write_bytes((folder / name).read_bytes())
    out = sal.prepare_sales(work, window=auto_window(orders_raw[mapping["order_purchase_timestamp"]]))
    orders, features = out["orders"], out["features"]
    analysis = sales_analysis(orders, features, out["quality"], f"{upload['company']} sales orders")
    analysis["predictions"] = predictions_section("sales", orders, features)
    analysis["model_performance"] = model_section("sales")
    analysis["forecast"] = forecast_orders(orders)
    return analysis, {"orders": orders, "features": features}


def run_upload(upload_id: str, mapping_overrides: dict = None, sla_rules: dict = None, decide=True) -> dict:
    upload = store.get_upload(upload_id)
    if upload is None:
        raise UploadError(f"No upload called {upload_id}.")
    domain = upload["domain"]
    if domain is None:
        raise UploadError("This upload was not recognised, so it cannot be analysed.")
    mapping = _mapping(upload, mapping_overrides)
    if domain == "incidents":
        analysis, tables = run_tickets(upload, mapping, sla_rules)
    else:
        analysis, tables = run_sales(upload, mapping)
    analysis["meta"]["analysis_id"] = upload_id
    if decide:
        from .decide import add_decisions                   # Steps 6 to 10
        add_decisions(analysis, "incidents" if domain == "incidents" else "sales", tables)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / f"{upload_id}.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    store.update_upload(upload_id, status="analysed")
    return analysis
