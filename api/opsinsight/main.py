"""OpsInsight API (FastAPI).

Start it (from the api folder, environment activated):
    uvicorn opsinsight.main:app --reload --port 8000
Then open http://localhost:8000/docs to try every endpoint in the browser.

An "analysis key" names one analysis:
    incidents / sales   -> the two example files in contracts/ (the demo data)
    upload-xxxxxxxx     -> an analysis made from a company's own upload
"""
import json
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import rag, store
from .engine.outcomes import compare_outcomes, kpis
from .engine.pipeline import ANALYSIS_DIR, run_upload
from .engine.uploads import UploadError, read_database, save_upload
from .models import predict as predictor
from .models.config import ROOT

CONTRACTS = ROOT / "contracts"
DOMAINS = ("incidents", "sales")
EDITABLE = ("title", "description", "suggested_owner")      # what a manager may change when editing a recommendation

app = FastAPI(title="OpsInsight API", version="0.2.0",
              description="Uploads, analysis, risk predictions with reasons, recommendations, approvals, outcomes "
                          "and Ask your data.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- helpers
def check_domain(domain: str):
    if domain not in DOMAINS:
        raise HTTPException(404, f"unknown domain '{domain}', use one of {DOMAINS}")


def read_analysis(key: str) -> dict:
    """The analysis JSON for a key, without decisions applied."""
    if key in DOMAINS:
        path = CONTRACTS / f"example_{key}.json"
    elif key.startswith("upload-") and key[7:].isalnum():
        path = ANALYSIS_DIR / f"{key}.json"
    else:
        raise HTTPException(404, f"unknown analysis '{key}'")
    if not path.exists():
        raise HTTPException(404, f"no analysis yet for '{key}' - run it first")
    return json.loads(path.read_text(encoding="utf-8"))


def domain_of(analysis: dict) -> str:
    return "incidents" if analysis["meta"]["domain"] == "support_tickets" else "sales"


def load_analysis(key: str) -> dict:
    """The analysis JSON with the manager's newest decisions applied to each recommendation."""
    data = read_analysis(key)
    saved = store.latest_decisions(key)
    for rec in data.get("recommendations", []):
        d = saved.get(rec["id"])
        if d:
            rec["status"] = d["decision"]
            rec["approval"] = d
            for field, value in (d["edited"] or {}).items():
                rec[field] = value
    return data


# --------------------------------------------------------------------------- 1. uploads
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/uploads")
async def upload_files(files: list[UploadFile] = File(...), company: str = Form("unknown")):
    """Upload one or more CSV / Excel files. Returns what was recognised and the suggested column mapping."""
    contents = {f.filename: await f.read() for f in files}
    try:
        return save_upload(contents, company, source="file")
    except UploadError as e:
        raise HTTPException(400, str(e))


class Connect(BaseModel):
    url: str = Field(..., examples=["sqlite:///C:/data/helpdesk.db"])
    table: str = Field(..., examples=["tickets"])
    company: str = "unknown"


@app.post("/api/uploads/connect")
def upload_from_database(body: Connect):
    """Copy one table out of a company database (read only) and treat it like an uploaded file."""
    try:
        return read_database(body.url, body.table, body.company)
    except UploadError as e:
        raise HTTPException(400, str(e))


@app.get("/api/uploads/{upload_id}")
def get_upload(upload_id: str):
    row = store.get_upload(upload_id)
    if row is None:
        raise HTTPException(404, f"no upload '{upload_id}'")
    row.pop("folder", None)                    # never show where files sit on the server
    return row


class RunRequest(BaseModel):
    mapping: dict[str, str] = Field(default_factory=dict, description="Corrections: standard name -> their column")
    sla_rules: Optional[dict] = Field(None, examples=[{"hours": {"1 - Critical": 4, "2 - High": 8}, "default_hours": 24}])


@app.post("/api/uploads/{upload_id}/run")
def run(upload_id: str, body: RunRequest):
    """Clean, model, analyse, forecast and decide. Takes from seconds to about a minute."""
    try:
        analysis = run_upload(upload_id, body.mapping, body.sla_rules)
    except UploadError as e:
        raise HTTPException(400, str(e))
    except (KeyError, ValueError) as e:
        store.update_upload(upload_id, status="failed")
        raise HTTPException(422, f"The data could not be analysed: {e}")
    return {"analysis_id": upload_id, "domain": domain_of(analysis), "period": analysis["meta"]["data_period"],
            "kpis": {k["id"]: k["value"] for k in analysis["kpis"]},
            "recommendations": [r["id"] for r in analysis.get("recommendations", [])]}


# --------------------------------------------------------------------------- 2. reading the analysis
@app.get("/api/analysis/{key}")
def get_analysis(key: str):
    """Everything: meta, data_quality, kpis, trends, breakdowns, root_causes, anomalies, forecast, predictions,
    ranked_findings, priority_queue, workload_plan, recommendations, impact_simulation, model_performance."""
    return load_analysis(key)


@app.get("/api/analysis/{key}/{section}")
def get_section(key: str, section: str):
    """One section only, e.g. /api/analysis/incidents/kpis"""
    data = load_analysis(key)
    if section not in data:
        raise HTTPException(404, f"no section '{section}'. Available: {list(data)}")
    return data[section]


@app.get("/api/predictions/{key}")
def get_predictions(key: str, risk_level: Optional[Literal["high", "medium", "low"]] = None,
                    limit: int = Query(10, ge=1, le=100)):
    items = load_analysis(key)["predictions"]["items"]
    if risk_level:
        items = [i for i in items if i["prediction"]["risk_level"] == risk_level]
    return {"count": len(items[:limit]), "items": items[:limit]}


# --------------------------------------------------------------------------- 3. approvals
class Decision(BaseModel):
    decision: Literal["approved", "rejected", "edited"]
    decided_by: str = Field(..., examples=["Ops manager"])
    comment: Optional[str] = Field(None, examples=["Pilot with Group 70 first"])
    edited: Optional[dict[str, str]] = Field(None, examples=[{"suggested_owner": "Team lead, Group 70"}])


@app.post("/api/recommendations/{key}/{rec_id}/decision")
def decide(key: str, rec_id: str, body: Decision):
    """Manager approves / rejects / edits a recommendation. Saved in the SQLite database with its history."""
    data = read_analysis(key)
    if rec_id not in {r["id"] for r in data.get("recommendations", [])}:
        raise HTTPException(404, f"no recommendation '{rec_id}'")
    if body.decision == "edited" and not body.edited:
        raise HTTPException(400, "an 'edited' decision needs the changed fields in 'edited'")
    bad = set(body.edited or {}) - set(EDITABLE)
    if bad:
        raise HTTPException(400, f"only these fields can be edited: {EDITABLE}")
    entry = store.add_decision(key, rec_id, body.decision, body.decided_by, body.comment, body.edited)
    return {"recommendation_id": rec_id, "status": body.decision, "approval": entry}


@app.get("/api/decisions/{key}")
def get_decisions(key: str):
    """The newest decision for each recommendation (older ones stay in the database as history)."""
    return store.latest_decisions(key)


# --------------------------------------------------------------------------- 4. outcomes
@app.get("/api/outcomes/{key}")
def outcomes(key: str, before: Optional[str] = Query(None, description="analysis key of the earlier period")):
    """Did the approved actions work? With `before`: that earlier analysis against this one.
    Without it: this analysis's previous period against its current period."""
    after = read_analysis(key)
    earlier = read_analysis(before) if before else None
    plan = earlier or after                                   # the recommendations were approved on this one
    decisions = store.latest_decisions(before or key)
    return compare_outcomes(domain_of(after), plan.get("recommendations", []), decisions,
                            kpis(earlier, "value") if earlier else kpis(after, "previous_value"), kpis(after, "value"),
                            before_analysis=earlier, after_analysis=after)


# --------------------------------------------------------------------------- 5. Ask your data
class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Why is Group 20 missing so many deadlines?"])


@app.post("/api/ask/{key}")
def ask(key: str, body: Question):
    """Answers only from this analysis's facts, with the fact IDs it used."""
    return rag.answer(body.question, load_analysis(key))


# --------------------------------------------------------------------------- 6. the model (Abhishek's)
class PredictRequest(BaseModel):
    rows: list[dict] = Field(..., description="One dict per item, with the model's feature columns")


@app.post("/api/predict/{domain}")
def predict_rows(domain: str, body: PredictRequest):
    check_domain(domain)
    try:
        return {"items": predictor.predict(domain, pd.DataFrame(body.rows))}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/model/{domain}")
def get_model(domain: str):
    check_domain(domain)
    try:
        return predictor.model_info(domain)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
