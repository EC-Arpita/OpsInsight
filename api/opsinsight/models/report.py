"""Write reports/model_results.md: the tables for the presentation.

Run from the api folder:   python -m opsinsight.models.report
(Runs the comparison again, so it takes about a minute.)
"""
import json

from .compare import compare
from .config import DOMAINS, MODELS_DIR, ROOT

NAMES = {"incidents": "Support tickets: SLA breach at opening", "sales": "Sales orders: late delivery at approval",
         "sales_pickup": "Sales orders: late delivery at carrier pickup (stage 2)"}


def table(rows, cols):
    """A list of dicts -> a markdown table."""
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    return head + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)


def build() -> str:
    md = ["# OpsInsight model results\n\n"]
    md.append("## Final models (test period = newest 20% of the data, never used for choosing)\n\n")
    rows = []
    for d in DOMAINS:
        m = json.loads((MODELS_DIR / f"{d}_meta.json").read_text())
        k = m["metrics"]
        rows.append({"Prediction": NAMES[d], "Model": m["algorithm"], "Test AUC": k["test_auc"],
                     "Problem rate (all)": f"{k['base_rate_test']:.1%}",
                     "Problem rate (top 10% flagged)": f"{k['precision_top_10pct']:.1%}",
                     "Lift": f"{k['lift_top_10pct']}x", "Test rows": f"{k['test_rows']:,}"})
    md.append(table(rows, list(rows[0])))
    md.append("\nLift = how many times more problems a manager finds by checking the model's top 10% instead of a random 10%.\n")

    for d in DOMAINS:
        md.append(f"\n## Model comparison: {NAMES[d]}\n\n")
        c = compare(d).to_dict(orient="records")
        md.append(table(c, ["model", "validation_auc", "test_auc", "test_top10_precision", "picked"]))
    md.append("\n`picked` = best single model on validation AUC (newest 20% of the training period). The final model averages "
              "LightGBM and logistic regression, which beat both alone on incident validation data (0.783).\n")
    return "".join(md)


def main():
    out = ROOT / "reports/model_results.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
