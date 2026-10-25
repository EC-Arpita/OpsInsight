"""Run every file in a folder through the whole app, the way a new company would, and print one line each.

    python -m opsinsight.check_outside              # files in data/outside
    python -m opsinsight.check_outside my_folder    # or any folder

Each .csv/.xlsx file is one upload; each sub-folder is one upload made of all its files (e.g. orders + customers).
Files without an "SLA met" column get the default SLA rules. Nothing here changes the reference models.
"""
import sys
import time
from pathlib import Path

from .engine.pipeline import run_upload
from .engine.uploads import ALLOWED, save_upload
from .models.config import ROOT

DEFAULT_SLA = {"hours": {"1 - Critical": 4, "2 - High": 8, "3 - Moderate": 24, "4 - Low": 72}, "default_hours": 24}
DEFAULT_PROMISE = {"column": None, "days": {}, "default_days": 4}


def uploads_in(folder: Path) -> list:
    """[(name, {file name: bytes})] for every file and sub-folder."""
    items = []
    for p in sorted(folder.iterdir()):
        files = [p] if p.is_file() else sorted(p.iterdir()) if p.is_dir() else []
        files = {f.name: f.read_bytes() for f in files if f.suffix.lower() in ALLOWED}
        if files:
            items.append((p.name, files))
    return items


def check_one(name: str, files: dict) -> dict:
    start = time.time()
    row = {"name": name, "ok": False}
    try:
        up = save_upload(files, company=f"check:{name}")
        row.update(domain=up["domain"], score=up["best_score"], status=up["status"], rows=up["rows"])
        if up["domain"] is None:
            row["problem"] = up["message"]
            return row
        needs_sla = up["domain"] == "incidents" and not (up["mapping"] or {}).get("made_sla")
        m = up["mapping"] or {}
        needs_promise = up["domain"] == "sales_orders" and not ({"order_estimated_delivery_date", "promised_days"} & set(m))
        promise = {**DEFAULT_PROMISE, "column": up["promise_by"]["column"]} if needs_promise else None
        a = run_upload(up["upload_id"], sla_rules=DEFAULT_SLA if needs_sla else None, promise_rules=promise)
        mp = a.get("model_performance", {})
        fit = mp.get("on_your_data") or {}
        kpi = {k["id"]: k["value"] for k in a.get("kpis", [])}
        row.update(ok=True, rate=kpi.get("sla_breach_rate", kpi.get("late_delivery_rate")),
                   auc=fit.get("auc", mp.get("test_auc")),
                   own_model="on_your_data" not in mp,
                   warnings=[w["message"] for w in a["data_quality"].get("warnings", [])])
    except Exception as e:                  # the point is to see what breaks, so keep going
        row["problem"] = f"{e.__class__.__name__}: {e}"
    finally:
        row["seconds"] = round(time.time() - start, 1)
    return row


def main(folder=None):
    folder = Path(folder or ROOT / "data/outside")
    if not folder.is_dir():
        sys.exit(f"No folder {folder}. Put some CSV/Excel files there first.")
    rows = [check_one(n, f) for n, f in uploads_in(folder)]
    for r in rows:
        head = "PASS" if r["ok"] else "FAIL"
        print(f"{head}  {r['name']}  domain={r.get('domain')}  score={r.get('score')}  rows={r.get('rows')}  "
              f"rate={r.get('rate')}  auc={r.get('auc')}  own_model={r.get('own_model')}  {r['seconds']}s")
        if r.get("problem"):
            print(f"      problem: {r['problem']}")
        for w in r.get("warnings", []):
            print(f"      warning: {w}")
    print(f"\n{sum(r['ok'] for r in rows)} of {len(rows)} passed.")
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
