"""Run the whole analysis on one upload: map -> (SLA rules) -> clean -> features -> analytics ->
predictions -> forecast -> decisions. Saves data/app/analysis/<upload id>.json."""
import json
from pathlib import Path

import pandas as pd

from ..forecast.sales import forecast_orders
from ..forecast.tickets import forecast_tickets
from ..models.config import DOMAINS, ROOT
from ..models.train import train_domain
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

YES = {"true", "yes", "y", "1", "met", "achieved", "within", "ok", "on time"}
PRIORITY_WORDS = {"critical": 1, "urgent": 1, "high": 2, "medium": 3, "moderate": 3, "normal": 3, "low": 4}


def standardise_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Another company's tickets -> the forms the cleaning code expects (dates, True/False SLA, numbered priority)."""
    df = df.copy()
    for c in inc.DATE_COLS:                                # any date style -> 13/02/2016 09:47
        if df[c].notna().any():
            df[c] = to_dates(df[c]).dt.strftime(inc.DATE_FORMAT)
    if df["made_sla"].dtype != bool:                       # "Met"/"Missed", "Yes"/"No", 1/0 -> True/False
        df["made_sla"] = df["made_sla"].astype(str).str.strip().str.lower().isin(YES)
    for c in ["priority", "impact", "urgency"]:            # "High" -> "2 - High", so a number can be read
        text = df[c].astype(str).str.strip()
        number = text.str.lower().map(PRIORITY_WORDS)
        fix = number.notna() & ~text.str.contains(r"\d")
        df.loc[fix, c] = number[fix].astype(int).astype(str) + " - " + text[fix]
    return df

SALES_DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"]
DELIVERED = {"delivered", "complete", "completed", "closed", "done", "fulfilled"}
OLIST_COLUMNS = {
    "items": ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"],
    "payments": ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"],
    "reviews": ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
                "review_creation_date", "review_answer_timestamp"],
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "products": ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght",
                 "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"],
    "translation": ["product_category_name", "product_category_name_english"],
}


def standardise_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Another company's orders -> Olist forms (year-first dates, 'delivered' as the finished status)."""
    df = df.copy()
    for c in SALES_DATES:
        if df[c].notna().any():
            df[c] = to_dates(df[c])
    status = df["order_status"].astype(str).str.strip().str.lower()
    df["order_status"] = status.where(~status.isin(DELIVERED), "delivered")
    return df


MIN_TRAIN_ROWS, MIN_PROBLEMS = 500, 30     # below this a model would learn noise: use the reference model


def own_model(domain: str, features: pd.DataFrame, name: str) -> str | None:
    """Train the risk model on this company's own history (older 80% -> tested on newest 20%).
    Returns the model name, or None (= use the reference model) when the file is too small to learn from."""
    target = DOMAINS[domain]["target"]
    train, test = features[features["split"] == "train"], features[features["split"] == "test"]
    if len(train) < MIN_TRAIN_ROWS or train[target].sum() < MIN_PROBLEMS or test[target].nunique() < 2:
        return None
    train_domain(domain, df=features, name=name)
    return name


def run_tickets(upload, mapping, sla_rules=None) -> tuple:
    raw = read_table(Path(upload["folder"]) / upload["main_file"])
    if "made_sla" not in mapping:
        if not sla_rules:
            raise UploadError("This file has no 'SLA met' column. Send SLA rules (hours allowed per priority).")
        raw = apply_sla_rules(raw, mapping, sla_rules)
        mapping = {**mapping, "made_sla": "made_sla"}
    df = standardise_tickets(apply_mapping(raw, mapping, "incidents"))
    log = QualityLog()
    first_rows = df.drop_duplicates("number")["opened_at"]      # one opening time per ticket
    events = inc.clean_events(df, window=auto_window(first_rows), log=log)
    tickets = inc.collapse_to_tickets(events, log=log)
    features, _ = inc.build_features(tickets)
    analysis = incident_analysis(tickets, features, log.to_frame(), f"{upload['company']} support tickets")
    model = own_model("incidents", features, upload["id"])
    analysis["predictions"] = predictions_section("incidents", tickets, features, model=model)
    analysis["model_performance"] = model_section("incidents", model)
    analysis["forecast"] = forecast_tickets(tickets)
    return analysis, {"tickets": tickets, "features": features, "model": model}


def run_sales(upload, mapping) -> tuple:
    folder = Path(upload["folder"])
    orders_raw = read_table(folder / upload["main_file"])
    work = folder / "standard"                             # Olist file names, the format the sales pipeline reads
    work.mkdir(exist_ok=True)
    orders = standardise_orders(apply_mapping(orders_raw, mapping, "sales_orders"))
    region = orders.pop("customer_state")                  # the region lives in the customers file in Olist's format
    orders.to_csv(work / sal.FILES["orders"], index=False)
    for key, name in sal.FILES.items():
        if key == "orders":
            continue
        if (folder / name).exists():
            (work / name).write_bytes((folder / name).read_bytes())
        elif key == "customers" and region.notna().any():  # no customers file: take the region from the orders file
            customers = pd.DataFrame({"customer_id": orders["customer_id"], "customer_state": region})
            customers.drop_duplicates("customer_id").reindex(columns=OLIST_COLUMNS["customers"]).to_csv(work / name, index=False)
        else:                                              # only the orders file is required: the rest start empty
            pd.DataFrame(columns=OLIST_COLUMNS[key]).to_csv(work / name, index=False)
    out = sal.prepare_sales(work, window=auto_window(orders_raw[mapping["order_purchase_timestamp"]]))
    orders, features = out["orders"], out["features"]
    analysis = sales_analysis(orders, features, out["quality"], f"{upload['company']} sales orders")
    model = own_model("sales", features, upload["id"])
    analysis["predictions"] = predictions_section("sales", orders, features, model=model)
    analysis["model_performance"] = model_section("sales", model)
    analysis["forecast"] = forecast_orders(orders)
    return analysis, {"orders": orders, "features": features, "model": model}


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