"""Put the final model's numbers into the analysis JSON the dashboard reads.

Notebook 08 wrote contracts/example_*.json with its own quick LightGBM model. After Steps 3-5 the real
model is the blend, so this replaces the parts that came from the model and leaves everything else alone.

Run from the api folder:   python -m opsinsight.models.refresh
"""
import json

import pandas as pd

from .config import DOMAINS, MODELS_DIR, ROOT
from .predict import load, predict

FILES = {"incidents": ["contracts/example_incidents.json", "frontend/public/example_incidents.json"],
         "sales": ["contracts/example_sales.json", "frontend/public/example_sales.json"]}
WORDS = {"incidents": ("tickets", "breach", "breaches_avoided", "breach_rate"),
         "sales": ("orders", "are late", "late_orders_avoided", "late_rate")}


def refresh(domain: str, data: dict) -> dict:
    cfg = DOMAINS[domain]
    meta = json.loads((MODELS_DIR / f"{domain}_meta.json").read_text())
    k = meta["metrics"]
    items_word, verb, avoided_key, rate_key = WORDS[domain]

    # 1. model_performance: the final model's scores
    data["model_performance"].update({"algorithm": meta["algorithm"], "test_auc": k["test_auc"], "train_auc": k["train_auc"],
                                      "base_rate_test": k["base_rate_test"], "precision_top_10pct": k["precision_top_10pct"],
                                      "lift_top_10pct": k["lift_top_10pct"], "n_features": len(meta["features"])})

    # 2. predictions: re-score the same items with the final model, then sort by the new risk
    feats = pd.read_parquet(cfg["features_file"])
    test = feats[feats["split"] == "test"]
    hi, med = load(domain)["meta"]["risk_thresholds"]["high"], load(domain)["meta"]["risk_thresholds"]["medium"]
    section = data["predictions"]
    ids = [it["id"] for it in section["items"]]
    new = {p["id"]: p for p in predict(domain, test[test[cfg["id_col"]].astype(str).isin(ids)])}
    for it in section["items"]:
        p = new[it["id"]]
        it["prediction"], it["top_reasons"], it["explanation_summary"] = p["prediction"], p["top_reasons"], p["explanation_summary"]
    section["items"].sort(key=lambda it: -it["prediction"]["risk"])
    section["risk_level_rules"] = {"high": f"risk ≥ {hi:.3f} (top 10%)", "medium": f"risk ≥ {med:.3f} (top 40%)",
                                   "low": "below medium"}

    # 3. REC3 ("fast-track the riskiest 10%") depends on the model: recompute its target, evidence and impact
    scored = pd.DataFrame(predict(domain, test))
    risk = scored["prediction"].map(lambda p: p["risk"]).to_numpy()
    top = test[risk >= hi]
    saved = float(top[cfg["target"]].sum() * 0.30)                       # same assumption as notebook 08
    base_n, base_bad = len(test), int(test[cfg["target"]].sum())
    new_rate = (base_bad - saved) / base_n
    rec = next(r for r in data["recommendations"] if r["id"] == "REC3")
    rec["target"][items_word] = f"risk ≥ {hi:.2f} (top 10%)"
    rec["target"]["count_current_period"] = len(top)
    rec["evidence"] = [f"Top-10% {items_word} {verb} {top[cfg['target']].mean():.0%} vs {base_bad / base_n:.0%} overall"]
    imp = rec["expected_impact"]
    imp["expected"], imp["change_pct_points"] = round(new_rate, 3), round((new_rate - base_bad / base_n) * 100, 1)
    per = [key for key in imp if key.endswith("_per_period")][0]
    rng = [key for key in imp if key.startswith("range_")][0]
    imp[per], imp[rng] = round(saved), [round(saved * 0.5), round(saved)]
    for s in data["impact_simulation"]["scenarios"]:
        if s["recommendation_id"] == "REC3":
            s[rate_key], s[avoided_key] = round(new_rate, 3), round(saved)
    return data


def main():
    for domain, paths in FILES.items():
        for rel in paths:
            path = ROOT / rel
            if not path.exists():
                continue
            data = refresh(domain, json.loads(path.read_text(encoding="utf-8")))
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        mp = data["model_performance"]
        top = data["predictions"]["items"][0]
        rec = next(r for r in data["recommendations"] if r["id"] == "REC3")
        print(f"{domain:9s} AUC {mp['test_auc']}  top-10% {mp['precision_top_10pct']:.1%}  lift {mp['lift_top_10pct']}x   "
              f"riskiest item {top['id'][:10]} at {top['prediction']['risk']:.0%}   REC3: {rec['evidence'][0]}")
    print("Updated contracts/ and frontend/public/")


if __name__ == "__main__":
    main()
