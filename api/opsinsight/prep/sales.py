"""Sales (Olist) pipeline: 9 raw CSVs -> one clean order table -> model features.

Same logic as notebooks 05 and 06, packaged as functions the app can call.

    from opsinsight.prep import prepare_sales
    out = prepare_sales("data/raw/sales", save_to="data")
    out["features"], out["orders"], out["quality"], out["feature_dictionary"]
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FeatureDictionary, QualityLog, count_in_window, past_mean, safe_merge, top_n_or_other

FILES = {
    "orders": "olist_orders_dataset.csv", "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv", "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv", "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv", "geolocation": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
DEFAULT_WINDOW = ("2017-01-01", "2018-09-01")
REGION = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "North"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Northeast"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Central-West"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Southeast"),
    **dict.fromkeys(["PR", "RS", "SC"], "South"),
}


def to_region(states: pd.Series) -> pd.Series:
    """Brazilian state codes -> 5 regions. Any other country keeps its own values: codes like PR, MA, PA
    also exist in the US (Puerto Rico, Massachusetts...), so only map when nearly all values are Brazilian."""
    known = states.dropna()
    if len(known) and known.isin(REGION.keys()).mean() >= 0.9:
        return states.map(REGION).fillna(states)
    return states


CATEGORY_PLAN = {"customer_state": 27, "seller_state": 27, "customer_region": 6, "seller_region": 6,
                 "main_category": 40, "main_payment_type": 5}


def load_tables(raw_dir) -> dict:
    raw_dir = Path(raw_dir)
    missing = [f for f in FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing files in {raw_dir}: {missing}")
    return {name: pd.read_csv(raw_dir / f) for name, f in FILES.items()}


# --------------------------------------------------------------------------- cleaning
def clean_and_join(tables: dict, window=DEFAULT_WINDOW, log: QualityLog = None) -> pd.DataFrame:
    """Clean every table, reduce to one row per order, join. Notebook 05."""
    log = log or QualityLog()

    # orders
    orders = tables["orders"].copy()
    log.add("load orders", len(orders), len(orders))
    orders = orders.rename(columns={
        "order_purchase_timestamp": "purchased_at", "order_approved_at": "approved_at",
        "order_delivered_carrier_date": "carrier_at", "order_delivered_customer_date": "delivered_at",
        "order_estimated_delivery_date": "estimated_at"})
    for c in ["purchased_at", "approved_at", "carrier_at", "delivered_at", "estimated_at"]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    if window:
        before = len(orders)
        orders = orders[(orders["purchased_at"] >= window[0]) & (orders["purchased_at"] < window[1])].copy()
        log.add(f"keep purchases {window[0]} to {window[1]}", before, len(orders))
    impossible = (orders["carrier_at"] < orders["purchased_at"]) | (orders["delivered_at"] < orders["carrier_at"])
    before = len(orders)
    orders = orders[~impossible].copy()
    log.add("drop impossible timelines", before, len(orders))
    orders["approved_was_missing"] = orders["approved_at"].isna().astype(int)
    orders["approved_at"] = orders["approved_at"].fillna(orders["purchased_at"])

    # products
    products = tables["products"].rename(columns={"product_name_lenght": "product_name_length",
                                                  "product_description_lenght": "product_description_length"})
    trans = pd.concat([tables["translation"], pd.DataFrame({
        "product_category_name": ["pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos"],
        "product_category_name_english": ["pc_gamer", "portable_kitchen_food_processors"]})], ignore_index=True)
    trans = trans.drop_duplicates("product_category_name")
    products = safe_merge(products, trans, "product_category_name", "translation")
    products["category"] = products["product_category_name_english"].fillna("unknown")
    products = products.drop(columns=["product_category_name", "product_category_name_english"])
    for c in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
              "product_photos_qty", "product_name_length", "product_description_length"]:
        products[c] = products[c].fillna(products.groupby("category")[c].transform("median"))
        products[c] = products[c].fillna(products[c].median())
    products["product_volume_cm3"] = (products["product_length_cm"] * products["product_height_cm"]
                                      * products["product_width_cm"])

    # items -> one row per order
    items = tables["items"].copy()
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
    items = safe_merge(items, products[["product_id", "category", "product_weight_g", "product_volume_cm3",
                                        "product_photos_qty", "product_description_length"]], "product_id", "products")
    items_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"), n_products=("product_id", "nunique"), n_sellers=("seller_id", "nunique"),
        n_categories=("category", "nunique"), items_price=("price", "sum"), items_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"), total_volume_cm3=("product_volume_cm3", "sum"),
        shipping_limit_at=("shipping_limit_date", "max")).reset_index()
    main_item = (items.sort_values(["order_id", "price"], ascending=[True, False]).drop_duplicates("order_id")
                 [["order_id", "seller_id", "product_id", "category", "product_photos_qty", "product_description_length"]]
                 .rename(columns={"seller_id": "main_seller_id", "product_id": "main_product_id", "category": "main_category"}))
    items_agg = items_agg.merge(main_item, on="order_id")

    # payments -> one row per order
    pay = tables["payments"].copy()
    pay["payment_type"] = pay["payment_type"].replace("not_defined", "unknown")
    pay_agg = pay.groupby("order_id").agg(payment_total=("payment_value", "sum"),
                                          n_payments=("payment_sequential", "size"),
                                          max_installments=("payment_installments", "max")).reset_index()
    main_pay = (pay.sort_values(["order_id", "payment_value"], ascending=[True, False]).drop_duplicates("order_id")
                [["order_id", "payment_type"]].rename(columns={"payment_type": "main_payment_type"}))
    pay_agg = pay_agg.merge(main_pay, on="order_id")
    pay_agg["used_voucher"] = pay_agg["order_id"].isin(pay.loc[pay["payment_type"] == "voucher", "order_id"]).astype(int)

    # reviews -> latest per order (analysis only, never a feature)
    rev = tables["reviews"].copy()
    rev["review_answer_timestamp"] = pd.to_datetime(rev["review_answer_timestamp"])
    rev_one = rev.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")[["order_id", "review_score"]]

    # geolocation -> one point per zip
    geo = tables["geolocation"]
    before = len(geo)
    geo = geo[geo["geolocation_lat"].between(-34, 5.5) & geo["geolocation_lng"].between(-74, -34)]
    log.add("geo: drop points outside Brazil", before, len(geo))
    geo_zip = (geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
               .reset_index().rename(columns={"geolocation_zip_code_prefix": "zip",
                                              "geolocation_lat": "lat", "geolocation_lng": "lng"}))
    cust = safe_merge(tables["customers"], geo_zip.rename(columns={
        "zip": "customer_zip_code_prefix", "lat": "customer_lat", "lng": "customer_lng"}),
        "customer_zip_code_prefix", "geo(customers)")
    sellers = safe_merge(tables["sellers"], geo_zip.rename(columns={
        "zip": "seller_zip_code_prefix", "lat": "seller_lat", "lng": "seller_lng"}),
        "seller_zip_code_prefix", "geo(sellers)").rename(columns={"seller_id": "main_seller_id"})

    # the big join
    df = safe_merge(orders, cust, "customer_id", "customers")
    df = safe_merge(df, items_agg, "order_id", "items")
    df = safe_merge(df, pay_agg, "order_id", "payments")
    df = safe_merge(df, rev_one, "order_id", "reviews")
    df = safe_merge(df, sellers, "main_seller_id", "sellers")
    before = len(df)
    if len(items_agg):                                     # no items file at all: keep every order
        df = df[df["n_items"].notna()].copy()
    log.add("drop orders with no items", before, len(df))

    df["is_model_row"] = ((df["order_status"] == "delivered") & df["delivered_at"].notna()).astype(int)
    df["late"] = np.where(df["is_model_row"] == 1, (df["delivered_at"] > df["estimated_at"]).astype(int), np.nan)
    df["days_late"] = (df["delivered_at"] - df["estimated_at"]).dt.total_seconds() / 86400
    log.add("model rows = delivered with a delivery date", len(df), int(df["is_model_row"].sum()), "others kept, flagged 0")
    if not df["order_id"].is_unique:
        raise ValueError("duplicate orders after join")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- features
def _haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = (np.radians(pd.to_numeric(x, errors="coerce")) for x in (lat1, lng1, lat2, lng2))   # empty file: text dtype
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(orders: pd.DataFrame, test_share=0.2, m=20):
    """Model-ready features, all known when the order is approved. Notebook 06.
    Returns (features_df, FeatureDictionary)."""
    df = orders[orders["is_model_row"] == 1].copy()
    df["late"] = df["late"].astype(int)
    df = df.sort_values("approved_at").reset_index(drop=True)
    fd = FeatureDictionary("order_approval")

    cutoff = df["purchased_at"].quantile(1 - test_share)
    df["split"] = np.where(df["purchased_at"] < cutoff, "train", "test")

    # time
    df["approve_hour"] = df["approved_at"].dt.hour
    df["approve_weekday"] = df["approved_at"].dt.dayofweek
    df["is_weekend"] = (df["approve_weekday"] >= 5).astype(int)
    df["purchase_month"] = df["purchased_at"].dt.month
    df["is_holiday_season"] = (((df["purchased_at"].dt.month == 11) & (df["purchased_at"].dt.day >= 15))
                               | (df["purchased_at"].dt.month == 12)).astype(int)
    df["promised_days"] = (df["estimated_at"] - df["approved_at"]).dt.total_seconds() / 86400
    df["approval_lag_hours"] = (df["approved_at"] - df["purchased_at"]).dt.total_seconds() / 3600
    df["days_to_ship_limit"] = (df["shipping_limit_at"] - df["approved_at"]).dt.total_seconds() / 86400
    for name, how in [("approve_hour", "hour of approval (0-23)"), ("approve_weekday", "day of week of approval (0=Mon)"),
                      ("is_weekend", "1 if approved on Sat/Sun"), ("purchase_month", "month of purchase (1-12)"),
                      ("is_holiday_season", "1 if purchased 15 Nov - 31 Dec (Black Friday + Christmas)"),
                      ("promised_days", "days between approval and the promised delivery date"),
                      ("approval_lag_hours", "hours between purchase and approval"),
                      ("days_to_ship_limit", "days the seller has (from approval) to hand the order to the carrier")]:
        fd.add(name, how, "time")

    # geography
    df["distance_km"] = _haversine_km(df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["customer_region"] = to_region(df["customer_state"]).fillna("Unknown")   # non-Brazil: keep own region
    df["seller_region"] = to_region(df["seller_state"]).fillna("Unknown")
    fd.add("distance_km", "haversine distance between seller and customer zip (km); missing if a zip has no coordinates", "geography")
    fd.add("same_state", "1 if seller and customer are in the same state", "geography")

    # contents + payment
    df["freight_ratio"] = df["items_freight"] / df["items_price"].replace(0, np.nan)
    df["is_multi_seller"] = (df["n_sellers"] > 1).astype(int)
    for c, how in {"n_items": "number of items in the order", "n_products": "distinct products",
                   "n_sellers": "distinct sellers", "n_categories": "distinct product categories",
                   "items_price": "total item price (BRL)", "items_freight": "total freight charged (BRL)",
                   "freight_ratio": "freight / item price", "total_weight_g": "total weight of all items (g)",
                   "total_volume_cm3": "total volume of all items (cm³)",
                   "product_photos_qty": "photos of the main product",
                   "product_description_length": "description length of the main product",
                   "is_multi_seller": "1 if more than one seller"}.items():
        fd.add(c, how, "order content")
    for c, how in {"payment_total": "total paid (BRL)", "n_payments": "number of payment parts",
                   "max_installments": "max number of instalments", "used_voucher": "1 if a voucher was used",
                   "approved_was_missing": "1 if approval time was missing in the raw data"}.items():
        fd.add(c, how, "payment")

    # customer
    df = df.sort_values("purchased_at").reset_index(drop=True)
    df["is_repeat_customer"] = (df.groupby("customer_unique_id").cumcount() > 0).astype(int)
    fd.add("is_repeat_customer", "1 if the same customer_unique_id ordered before", "customer")

    # history (only outcomes already known at approval)
    prior = df.loc[df["split"] == "train", "late"].mean()
    df["route"] = df["seller_state"].fillna("?").astype(str) + "→" + df["customer_state"].fillna("?").astype(str)
    for key, label in [("main_seller_id", "seller"), ("main_category", "category"), ("route", "route"),
                       ("customer_state", "customer_state")]:
        df[f"{label}_past_late_rate"], _ = past_mean(df, key, "late", "delivered_at", "approved_at", prior, m)
        fd.add(f"{label}_past_late_rate",
               f"smoothed late rate of earlier orders with the same {key}, delivered before this approval (m={m})", "history")
    df["_dispatch_days"] = ((df["carrier_at"] - df["approved_at"]).dt.total_seconds() / 86400).clip(lower=0)
    prior_dispatch = df.loc[df["split"] == "train", "_dispatch_days"].mean()
    df["seller_past_dispatch_days"], _ = past_mean(df, "main_seller_id", "_dispatch_days", "carrier_at",
                                                   "approved_at", prior_dispatch, m)
    fd.add("seller_past_dispatch_days", "smoothed average days the seller took to hand earlier orders to the carrier", "history")

    # workload
    df["seller_orders_last_7d"] = 0
    df["seller_orders_last_30d"] = 0
    for _, idx in df.groupby("main_seller_id").groups.items():
        times, q = np.sort(df.loc[idx, "purchased_at"].values), df.loc[idx, "approved_at"].values
        df.loc[idx, "seller_orders_last_7d"] = count_in_window(times, q, np.timedelta64(7, "D"))
        df.loc[idx, "seller_orders_last_30d"] = count_in_window(times, q, np.timedelta64(30, "D"))
    fd.add("seller_orders_last_7d", "orders the main seller received in the 7 days before approval", "workload")
    fd.add("seller_orders_last_30d", "orders the main seller received in the 30 days before approval", "workload")

    # categories
    train = df["split"] == "train"
    for col, n in CATEGORY_PLAN.items():
        df[f"{col}_cat"] = top_n_or_other(df, col, n, train)
        fd.add(f"{col}_cat", f"{col}, top {n} values from training period, rest = 'Other'", "category")

    features = df[["order_id", "purchased_at", "approved_at", "split"] + fd.names + ["late"]].copy()
    numeric = [c for c in fd.names if not c.endswith("_cat")]
    features[numeric] = features[numeric].apply(pd.to_numeric, errors="coerce")   # an empty optional file reads as text
    leaky = {"delivered_at", "carrier_at", "review_score", "days_late", "_dispatch_days", "order_status"} & set(features.columns)
    if leaky:
        raise ValueError(f"leaky columns in features: {leaky}")
    return features, fd


# --------------------------------------------------------------------------- stage 2: at carrier pickup
PICKUP_FEATURES = ["dispatch_days", "days_left_at_pickup", "handed_over_late", "dispatch_vs_seller_usual", "pickup_weekday"]


def pickup_features(features: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: the approval features plus what is known once the carrier has picked the order up.
    Only orders that have a pickup date; nothing that happens after the pickup (delivery, review) is used."""
    o = orders[["order_id", "carrier_at", "estimated_at", "shipping_limit_at"]]
    df = features.merge(o, on="order_id", how="inner")
    df = df[df["carrier_at"].notna()].copy()
    days = lambda s: s.dt.total_seconds() / 86400
    df["dispatch_days"] = days(df["carrier_at"] - df["approved_at"]).clip(lower=0)
    df["days_left_at_pickup"] = days(df["estimated_at"] - df["carrier_at"])
    df["handed_over_late"] = (df["carrier_at"] > df["shipping_limit_at"]).astype(int)    # seller missed its own deadline
    df["dispatch_vs_seller_usual"] = df["dispatch_days"] - df["seller_past_dispatch_days"]
    df["pickup_weekday"] = df["carrier_at"].dt.dayofweek
    return df.drop(columns=["carrier_at", "estimated_at", "shipping_limit_at"]).reset_index(drop=True)


# --------------------------------------------------------------------------- one call
def prepare_sales(raw_dir, window=DEFAULT_WINDOW, test_share=0.2, save_to=None):
    """Run the whole sales pipeline on the folder with the 9 Olist CSV files.

    raw_dir : folder containing the CSVs (e.g. data/raw/sales)
    window  : (start, end) purchase dates to keep, or None to keep everything
    save_to : project 'data' folder; if given, saves interim/ and processed/ parquet files
    Returns a dict with orders, features, quality (DataFrame), feature_dictionary (DataFrame).
    """
    log = QualityLog()
    orders = clean_and_join(load_tables(raw_dir), window=window, log=log)
    features, fd = build_features(orders, test_share=test_share)
    out = {"orders": orders, "features": features, "quality": log.to_frame(), "feature_dictionary": fd.to_frame(),
           "pickup_features": pickup_features(features, orders)}
    if save_to:
        data = Path(save_to)
        (data / "interim").mkdir(parents=True, exist_ok=True)
        (data / "processed").mkdir(parents=True, exist_ok=True)
        orders.to_parquet(data / "interim/sales_orders_clean.parquet", index=False)
        features.to_parquet(data / "processed/sales_orders_features.parquet", index=False)
        out["pickup_features"].to_parquet(data / "processed/sales_pickup_features.parquet", index=False)
    return out
