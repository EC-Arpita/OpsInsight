"""Run the whole analysis on one upload: map -> (SLA rules) -> clean -> features -> analytics ->
predictions -> forecast -> decisions. Saves data/app/analysis/<upload id>.json."""
import json
import re
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
from ..prep.mapping import ALTERNATIVES, apply_mapping, check_needed, normalise
from . import store
from .predictions import model_section, pickup_section, predictions_section
from .sla import apply_sla_rules
from .prep_dates import date_check
from .uploads import UploadError, read_table, to_dates

ANALYSIS_DIR = ROOT / "data/app/analysis"


def auto_window(dates: pd.Series, share=0.10, order: str = None) -> tuple:
    """Keep the months with real volume: at least `share` of the busiest month. Near-empty months at the
    edges of a file (a few stray tickets, an export that stopped early) would distort trends and forecasts."""
    d = to_dates(dates, order).dropna()
    if d.empty:                                            # the analyst matched a text column as the date
        raise UploadError(f"The column '{dates.name}' has no readable dates. On the matching screen, pick the real date column.")
    months = d.dt.to_period("M").value_counts().sort_index()
    busy = months[months >= share * months.max()].index
    return f"{busy.min().start_time:%Y-%m-%d}", f"{(busy.max() + 1).start_time:%Y-%m-%d}"


def _mapping(upload: dict, overrides: dict | None) -> dict:
    mapping = dict(upload["report"]["mapping"] or {})
    mapping.update(overrides or {})                       # the analyst's corrections win
    return {k: v for k, v in mapping.items() if v}       # an empty value means "not in my file"


YES = {"true", "yes", "y", "1", "met", "achieved", "within", "ok", "on time"}
BAD_FLAG = re.compile(r"breach|violat|miss|overdue|late|fail")     # a column saying the SLA was NOT met
PRIORITY_WORDS = {"critical": 1, "urgent": 1, "high": 2, "medium": 3, "moderate": 3, "normal": 3, "low": 4}


def standardise_tickets(df: pd.DataFrame, order: str = None, flag_means_breach: bool = False) -> pd.DataFrame:
    """Another company's tickets -> the forms the cleaning code expects (dates, True/False SLA, numbered priority)."""
    df = df.copy()
    for c in inc.DATE_COLS:                                # any date style -> 13/02/2016 09:47
        if df[c].notna().any():
            df[c] = to_dates(df[c], order).dt.strftime(inc.DATE_FORMAT)
    if df["made_sla"].dtype != bool:                       # "Met"/"Missed", "Yes"/"No", 1/0 -> True/False
        df["made_sla"] = df["made_sla"].astype(str).str.strip().str.lower().isin(YES)
    if flag_means_breach:                                  # "SLA Breached = Yes" means it was NOT met
        df["made_sla"] = ~df["made_sla"]
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


def standardise_orders(df: pd.DataFrame, order: str = None) -> pd.DataFrame:
    """Another company's orders -> Olist forms (year-first dates, 'delivered' as the finished status)."""
    df = df.copy()
    for c in SALES_DATES:
        if df[c].notna().any():
            df[c] = to_dates(df[c], order)
    df["customer_id"] = df["customer_id"].fillna(df["order_id"])   # no customer column: each order its own customer
    status = df["order_status"].astype(str).str.strip().str.lower()
    df["order_status"] = status.where(~status.isin(DELIVERED), "delivered")
    for date_col, days_col in ALTERNATIVES.items():           # "promised in 4 days" -> order date + 4 days
        if df[date_col].isna().all() and df[days_col].notna().any():
            days = pd.to_numeric(df[days_col], errors="coerce")
            df[date_col] = df["order_purchase_timestamp"] + pd.to_timedelta(days, unit="D")
    return df.drop(columns=list(ALTERNATIVES.values())).drop_duplicates("order_id")   # order lines -> one row per order


MIN_TRAIN_ROWS, MIN_PROBLEMS = 100, 10     # below this a model would learn noise; above it, own data beats a foreign model


def own_model(domain: str, features: pd.DataFrame, name: str) -> str | None:
    """Train the risk model on this company's own history (older 80% -> tested on newest 20%).
    Returns the model name, or None (= use the reference model) when the file is too small to learn from."""
    target = DOMAINS[domain]["target"]
    train, test = features[features["split"] == "train"], features[features["split"] == "test"]
    if len(train) < MIN_TRAIN_ROWS or train[target].sum() < MIN_PROBLEMS or test[target].nunique() < 2:
        return None
    train_domain(domain, df=features, name=name)
    return name


def run_tickets(upload, mapping, sla_rules=None, date_order=None) -> tuple:
    raw = read_table(Path(upload["folder"]) / upload["main_file"])
    order, warnings = date_check(raw, mapping, date_order, times_matter=True)
    if "made_sla" not in mapping:
        if not sla_rules:
            raise UploadError("This file has no 'SLA met' column. Send SLA rules (hours allowed per priority).")
        raw = apply_sla_rules(raw, mapping, sla_rules, order)
        mapping = {**mapping, "made_sla": "made_sla"}
    inverted = mapping["made_sla"] != "made_sla" and bool(BAD_FLAG.search(normalise(mapping["made_sla"])))
    if inverted:
        warnings.append({"level": "info", "message": f"'{mapping['made_sla']}' was read as 'SLA missed' (Yes = breached)."})
    df = standardise_tickets(apply_mapping(raw, mapping, "incidents"), order, inverted)
    log = QualityLog()
    first_rows = df.drop_duplicates("number")["opened_at"]      # one opening time per ticket
    events = inc.clean_events(df, window=auto_window(first_rows), log=log)
    tickets = inc.collapse_to_tickets(events, log=log)
    features, _ = inc.build_features(tickets)
    analysis = incident_analysis(tickets, features, log.to_frame(), f"{upload['company']} support tickets")
    analysis["data_quality"]["warnings"] += warnings
    model = own_model("incidents", features, upload["id"])
    analysis["predictions"] = predictions_section("incidents", tickets, features, model=model)
    analysis["model_performance"] = model_section("incidents", model, features)
    if "warning" in analysis["model_performance"]:
        analysis["data_quality"]["warnings"].append({"level": "warning", "message": analysis["model_performance"]["warning"]})
    analysis["forecast"] = forecast_tickets(tickets)
    return analysis, {"tickets": tickets, "features": features, "model": model}


def apply_promise_rules(raw: pd.DataFrame, mapping: dict, rules: dict) -> tuple:
    """No promised date in the file: days promised per value of one column (e.g. ship mode), typed by the analyst.
    rules = {"column": "Ship Mode" or None, "days": {"Standard Class": 5, ...}, "default_days": 4}"""
    raw = raw.copy()
    col = rules.get("column")
    days = raw[col].astype(str).str.strip().map(rules.get("days", {})) if col in raw else pd.Series(index=raw.index, dtype=float)
    raw["promised_days"] = days.fillna(rules.get("default_days", 4)).astype(float)
    return raw, {**mapping, "promised_days": "promised_days"}


def run_sales(upload, mapping, date_order=None, promise_rules=None) -> tuple:
    folder = Path(upload["folder"])
    orders_raw = read_table(folder / upload["main_file"])
    order, warnings = date_check(orders_raw, mapping, date_order, times_matter=False)
    if "order_estimated_delivery_date" not in mapping and "promised_days" not in mapping:
        if not promise_rules:
            raise UploadError("This file has no promised delivery date. Enter the days you promise on the upload screen.")
        orders_raw, mapping = apply_promise_rules(orders_raw, mapping, promise_rules)
        warnings.append({"level": "info", "message": "Promised dates come from the days you entered on the upload screen."})
    delivered = mapping.get("order_delivered_customer_date", "")
    if re.search(r"ship|dispatch", normalise(delivered)):
        warnings.append({"level": "warning", "message": f"No delivery date: '{delivered}' is used, so 'late' means "
                                                         "shipped after the promised date."})
    work = folder / "standard"                             # Olist file names, the format the sales pipeline reads
    work.mkdir(exist_ok=True)
    orders = standardise_orders(apply_mapping(orders_raw, mapping, "sales_orders"), order)
    region = orders.pop("customer_state")                  # the region lives in the customers file in Olist's format
    value = orders.pop("order_value")                      # ... and the order value in the payments file
    if not (folder / sal.FILES["customers"]).exists():
        # Olist keeps one customer_id per ORDER and the person in customer_unique_id; so each order keeps its own
        # region even when the same customer orders to different regions
        person = orders["customer_id"]
        orders["customer_id"] = "c-" + orders["order_id"].astype(str)
        pd.DataFrame({"customer_id": orders["customer_id"], "customer_unique_id": person, "customer_state": region}
                     ).reindex(columns=OLIST_COLUMNS["customers"]).to_csv(work / sal.FILES["customers"], index=False)
    orders.to_csv(work / sal.FILES["orders"], index=False)
    for key, name in sal.FILES.items():
        if key == "orders" or (key == "customers" and (work / name).exists() and not (folder / name).exists()):
            continue
        if (folder / name).exists():
            (work / name).write_bytes((folder / name).read_bytes())
        elif key == "payments" and pd.to_numeric(value, errors="coerce").notna().any():   # no payments file: one payment per order
            pd.DataFrame({"order_id": orders["order_id"], "payment_sequential": 1, "payment_type": "unknown",
                          "payment_installments": 1, "payment_value": pd.to_numeric(value, errors="coerce")}
                         ).to_csv(work / name, index=False)
        else:                                              # only the orders file is required: the rest start empty
            pd.DataFrame(columns=OLIST_COLUMNS[key]).to_csv(work / name, index=False)
    out = sal.prepare_sales(work, window=auto_window(orders_raw[mapping["order_purchase_timestamp"]], order=order))
    orders, features = out["orders"], out["features"]
    if orders.empty or features.empty:
        raise UploadError("No finished orders with a delivery date were left after cleaning. Check that the status "
                          "and date columns in step 2 are the right ones.")
    analysis = sales_analysis(orders, features, out["quality"], f"{upload['company']} sales orders")
    analysis["meta"]["currency"] = None                     # a company's own file: its currency is not known
    analysis["data_quality"]["warnings"] += warnings
    model = own_model("sales", features, upload["id"])
    analysis["predictions"] = predictions_section("sales", orders, features, model=model)
    analysis["model_performance"] = model_section("sales", model, features)
    if "warning" in analysis["model_performance"]:
        analysis["data_quality"]["warnings"].append({"level": "warning", "message": analysis["model_performance"]["warning"]})
    pickup = out["pickup_features"]
    pickup_model = own_model("sales_pickup", pickup, upload["id"] + "-pickup") if len(pickup) else None
    stage2 = pickup_section(orders, pickup, model=pickup_model)
    if stage2:
        analysis["predictions_at_pickup"] = stage2
    analysis["forecast"] = forecast_orders(orders)
    return analysis, {"orders": orders, "features": features, "model": model}


def run_upload(upload_id: str, mapping_overrides: dict = None, sla_rules: dict = None, decide=True, date_order: str = None,
               promise_rules: dict = None) -> dict:
    upload = store.get_upload(upload_id)
    if upload is None:
        raise UploadError(f"No upload called {upload_id}.")
    domain = upload["domain"]
    if domain is None:
        raise UploadError("This upload was not recognised, so it cannot be analysed.")
    mapping = _mapping(upload, mapping_overrides)
    try:
        check_needed(mapping, domain)
    except ValueError as e:
        raise UploadError(str(e)) from None
    if domain == "incidents":
        analysis, tables = run_tickets(upload, mapping, sla_rules, date_order)
    else:
        analysis, tables = run_sales(upload, mapping, date_order, promise_rules)
    analysis["meta"]["analysis_id"] = upload_id
    if decide:
        from .decide import add_decisions                   # Steps 6 to 10
        add_decisions(analysis, "incidents" if domain == "incidents" else "sales", tables)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / f"{upload_id}.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    store.update_upload(upload_id, status="analysed")
    store.save_mapping(upload["company"], domain, mapping)
    return analysis
