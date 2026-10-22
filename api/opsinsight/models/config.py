"""Per-domain settings shared by training and prediction."""
from pathlib import Path


def project_root() -> Path:
    """Walk up from this file until we find the folder with requirements.txt."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "requirements.txt").exists():
            return parent
    raise FileNotFoundError("could not find project root (folder with requirements.txt)")


ROOT = project_root()
MODELS_DIR = ROOT / "models"

# Cautious LightGBM settings chosen in notebook 07 (small trees, 200+ rows per leaf, penalty)
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, reg_lambda=5,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)

DOMAINS = {
    "incidents": {
        "features_file": ROOT / "data/processed/incidents_features.parquet",
        "id_col": "number", "target": "breached", "target_name": "sla_breach",
        "target_text": "missing SLA", "item_type": "ticket", "prediction_moment": "ticket opened",
        "non_feature_cols": ["number", "opened_at", "split", "breached"],
    },
    "sales": {
        "features_file": ROOT / "data/processed/sales_orders_features.parquet",
        "id_col": "order_id", "target": "late", "target_name": "late_delivery",
        "target_text": "late delivery", "item_type": "order", "prediction_moment": "order approved",
        "non_feature_cols": ["order_id", "purchased_at", "approved_at", "split", "late"],
    },
}

# Business-friendly names shown in the "Why?" panel
LABELS = {
    "incidents": {
        "recent_resolved_7d": "Tickets resolved helpdesk-wide in last 7 days", "assignment_group_cat": "Assigned team",
        "group_recent_breach_rate_7d": "Team's breach rate in last 7 days", "opened_by_cat": "Opened by (agent)",
        "u_symptom_cat": "Reported symptom", "recent_breach_rate_7d": "Helpdesk breach rate in last 7 days",
        "subcategory_cat": "Subcategory", "category_cat": "Category",
        "category_past_log_hours": "Category's usual resolution time (log hours)", "priority_n": "Priority (1 = Critical)",
        "open_weekday": "Day of week opened (0 = Mon)", "knowledge": "Knowledge article linked", "location_cat": "Location",
        "group_backlog_at_open": "Team backlog when opened", "category_past_breach_rate": "Category's past breach rate",
        "assignment_group_past_breach_rate": "Team's past breach rate", "group_opened_last_24h": "Tickets the team got in last 24h",
        "group_opened_last_7d": "Tickets the team got in last 7 days", "all_opened_last_24h": "Tickets helpdesk got in last 24h",
        "open_hour": "Hour opened", "impact_n": "Impact (1 = High)", "urgency_n": "Urgency (1 = High)",
        "caller_tickets_last_30d": "Caller's tickets in last 30 days", "location_past_breach_rate": "Location's past breach rate",
        "caller_id_past_breach_rate": "Caller's past breach rate", "contact_type_cat": "Contact channel",
        "is_business_hours": "Opened in business hours", "is_weekend": "Opened at the weekend",
        "impact_x_urgency": "Impact × urgency score",
    },
    "sales": {
        "customer_state_cat": "Customer state", "promised_days": "Days promised for delivery", "purchase_month": "Purchase month",
        "customer_state_past_late_rate": "Customer state's past late rate", "main_category_cat": "Product category",
        "distance_km": "Seller-to-customer distance (km)", "seller_past_dispatch_days": "Seller's usual dispatch time (days)",
        "is_holiday_season": "Holiday season", "route_past_late_rate": "Route's past late rate", "seller_state_cat": "Seller state",
        "items_freight": "Freight charged (BRL)", "category_past_late_rate": "Category's past late rate",
        "same_state": "Same state", "seller_past_late_rate": "Seller's past late rate",
        "days_to_ship_limit": "Days seller has to ship", "approval_lag_hours": "Hours from purchase to approval",
        "seller_orders_last_30d": "Seller's orders in last 30 days", "seller_orders_last_7d": "Seller's orders in last 7 days",
        "total_weight_g": "Total weight (g)", "items_price": "Item price (BRL)", "freight_ratio": "Freight / price",
        "max_installments": "Payment instalments", "approve_weekday": "Day of week approved (0 = Mon)",
        "customer_region_cat": "Customer region", "seller_region_cat": "Seller region",
        "main_payment_type_cat": "Payment type",
    },
}
