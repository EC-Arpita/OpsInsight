"""API tests: run with   python -m pytest tests/test_api.py -v
Uses a fresh, empty database in a temporary folder, so your real decisions are never touched."""
import io
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from opsinsight import main
from opsinsight.engine import pipeline, store, uploads
from opsinsight.models.config import ROOT

DEMO = ROOT / "data/demo"


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_FILE", tmp_path / "test.db")
    monkeypatch.setattr(uploads, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(pipeline, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(main, "ANALYSIS_DIR", tmp_path / "analysis")


client = TestClient(main.app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_example_analysis_has_decision_sections():
    data = client.get("/api/analysis/incidents").json()
    for section in ("ranked_findings", "priority_queue", "workload_plan", "recommendations", "impact_simulation"):
        assert section in data, section
    assert client.get("/api/analysis/nothing").status_code == 404
    assert client.get("/api/analysis/..%2Fsecrets").status_code == 404


def test_refuses_bad_files():
    r = client.post("/api/uploads", files={"files": ("notes.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 400
    r = client.post("/api/uploads", files={"files": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_decision_saved_and_applied():
    r = client.post("/api/recommendations/incidents/REC1/decision",
                    json={"decision": "edited", "decided_by": "Ops manager", "edited": {"suggested_owner": "Group 70 lead"}})
    assert r.status_code == 200
    rec = next(x for x in client.get("/api/analysis/incidents").json()["recommendations"] if x["id"] == "REC1")
    assert rec["status"] == "edited" and rec["suggested_owner"] == "Group 70 lead"
    bad = client.post("/api/recommendations/incidents/REC1/decision",
                      json={"decision": "edited", "decided_by": "x", "edited": {"expected_impact": "huge"}})
    assert bad.status_code == 400
    assert client.post("/api/recommendations/incidents/REC9/decision",
                       json={"decision": "approved", "decided_by": "x"}).status_code == 404


def test_outcomes_only_for_approved_actions():
    assert client.get("/api/outcomes/incidents").json()["actions"] == []
    client.post("/api/recommendations/incidents/REC1/decision", json={"decision": "approved", "decided_by": "Ops"})
    rows = client.get("/api/outcomes/incidents").json()["actions"]
    assert [r["recommendation_id"] for r in rows] == ["REC1"]
    assert rows[0]["overall"]["verdict"] in ("worked", "partly worked", "did not work")


def test_ask_answers_with_sources_and_admits_gaps():
    r = client.post("/api/ask/incidents", json={"question": "Why is Group 20 missing so many deadlines?"}).json()
    assert r["cited_ids"] and r["verified"]
    r = client.post("/api/ask/incidents", json={"question": "What will our revenue be next year?"}).json()
    assert r["answer"].startswith("Not enough evidence")


def test_database_connect(tmp_path):
    db = tmp_path / "company.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (a TEXT)")
    con.close()
    r = client.post("/api/uploads/connect", json={"url": f"sqlite:///{db.as_posix()}", "table": "t; DROP TABLE t"})
    assert r.status_code == 400


@pytest.mark.skipif(not (DEMO / "sales").exists(), reason="make the demo files first (Abhishek's Step 8)")
def test_sales_upload_end_to_end():
    folder = DEMO / "sales"
    files = [("files", (p.name, p.read_bytes(), "text/csv")) for p in sorted(folder.glob("*.csv"))]
    up = client.post("/api/uploads", files=files, data={"company": "ShopNest"}).json()
    assert up["domain"] == "sales_orders" and up["status"] == "ok"
    run = client.post(f"/api/uploads/{up['upload_id']}/run", json={}).json()
    assert run["domain"] == "sales" and {"REC1", "REC3"} <= set(run["recommendations"])   # region gap is small in the sample
    assert client.get(f"/api/uploads/{up['upload_id']}").json()["status"] == "analysed"
    assert "folder" not in client.get(f"/api/uploads/{up['upload_id']}").json()
    assert up["upload_id"] in [u["id"] for u in client.get("/api/uploads").json()]
    again = client.post("/api/uploads", files=files, data={"company": "ShopNest"}).json()
    assert again.get("mapping_remembered") is True                    # same company, same export
    xl = client.get(f"/api/analysis/{up['upload_id']}/export.xlsx")
    assert xl.status_code == 200 and set(pd.read_excel(io.BytesIO(xl.content), sheet_name=None)) == {"Actions", "High risk", "KPIs"}
    ans = client.post(f"/api/ask/{up['upload_id']}", json={"question": "Why are orders late?"}).json()
    assert ans["cited_ids"]


def test_vector_search_finds_by_meaning():
    from chromadb.utils.embedding_functions import EmbeddingFunction
    from opsinsight.engine import rag

    class Topics(EmbeddingFunction):
        """Tiny stand-in for the real model: one dimension per topic, so it runs without a download."""
        def __init__(self): pass
        def __call__(self, input):
            topics = [("overload", "too much work", "busy"), ("late", "delay"), ("revenue", "sales")]
            return [[1.0 if any(w in t.lower() for w in group) else 0.0 for group in topics] + [0.01] for t in input]
        @staticmethod
        def name(): return "topics-test"

    cards = [{"id": "RC2", "text": "Group 20 is overloaded"}, {"id": "RC1", "text": "Orders to the North-East are late"}]
    found = rag.vector_search("Which team has too much work?", cards, embed=Topics())
    assert [c["id"] for c in found] == ["RC2"] and found[0]["search"] == "vector"
