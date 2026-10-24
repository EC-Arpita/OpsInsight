"""RAG for 'Ask your data' and written explanations: find the right facts first, then answer only from them.

1. fact_cards(): every result in the analysis becomes one short sentence with an ID (the knowledge base).
2. search():     find the cards closest to the question (TF-IDF word matching; works offline).
3. answer():     an LLM (Ollama, on this laptop) writes the answer from those cards and cites their IDs;
                 check_numbers() confirms every number appears in a cited card; otherwise a template answer.
"""
import json
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("OPSINSIGHT_LLM", "llama3.1:8b")


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


# --------------------------------------------------------------------------- 1. the knowledge base
def fact_cards(analysis: dict) -> list:
    cards = []
    add = lambda id, text: cards.append({"id": id, "text": text})
    for k in analysis["kpis"]:
        fmt = _pct if k["unit"] == "ratio" else (lambda v: f"{v:,}")
        add(f"KPI:{k['id']}", f"{k['label']} this period is {fmt(k['value'])} (previous period {fmt(k['previous_value'])}), "
                              f"trend {k['trend']}. {k['definition']}.")
    rate_key = "breach_rate" if analysis["meta"]["domain"] == "support_tickets" else "late_rate"
    for b in analysis.get("breakdowns", []):
        for row in b["rows"][:8]:
            count = row.get("tickets", row.get("orders"))
            add(f"BD:{b['dimension']}:{row['value']}",
                f"{b['label']} {row['value']}: {count:,} items this period, problem rate {_pct(row[rate_key])}"
                f" (flag: {row['flag'].replace('_', ' ')}).")
    for rc in analysis["root_causes"]:
        add(rc["id"], f"Root cause - {rc['title']}: {rc['finding']}")
    for a in analysis.get("anomalies", [])[:6]:
        add(f"AN:{a['week_ending']}", f"Unusual week ending {a['week_ending']}: {a['label']} {_pct(a['actual'])} "
                                       f"against a normal {_pct(a['expected'])} ({a['severity']}).")
    fc = analysis.get("forecast") or {}
    for g in fc.get("teams", []):
        days = [d["date"] for d in g["daily"] if d["overload"]]
        add(f"FC:{g['team']}", f"Forecast for {g['team']}: {g['expected_tickets_7d']} tickets expected next 7 days, "
                               f"capacity {g['expected_capacity_7d']}; "
                               + (f"overloaded on {', '.join(days)}." if days else "within capacity every day."))
    for g in fc.get("regions", []):
        days = [d["date"] for d in g["daily"] if d["busy"]]
        add(f"FC:{g['region']}", f"Forecast for {g['region']}: {g['expected_orders_7d']} orders expected next 7 days, "
                                 f"usual week {g['usual_orders_7d']}; "
                                 + (f"unusually busy on {', '.join(days)}." if days else "a normal week."))
    for r in analysis.get("recommendations", []):
        ei = r["expected_impact"]
        avoided = next(v for k, v in ei.items() if k.endswith("_avoided_per_period"))
        add(r["id"], f"Recommendation {r['id']} - {r['title']}: {r['description']} Expected: problem rate "
                     f"{_pct(ei['baseline'])} to {_pct(ei['expected'])}, about {avoided} fewer problems per period "
                     f"(estimate). Effort {r['effort']}, owner {r['suggested_owner']}.")
    mp = analysis.get("model_performance")
    if mp:
        add("MODEL", f"The risk model ({mp['algorithm']}) has test AUC {mp['test_auc']}; the riskiest 10% flagged have a "
                     f"problem rate of {_pct(mp['precision_top_10pct'])} against {_pct(mp['base_rate_test'])} overall "
                     f"({mp['lift_top_10pct']}x).")
    return cards


# --------------------------------------------------------------------------- 2. search
MIN_MATCH = 0.05          # below this, a card is not really about the question
# Words a manager uses -> words the fact cards use, so "what should we do" finds the recommendations.
SYNONYMS = {"do": "recommendation", "fix": "recommendation", "action": "recommendation", "should": "recommendation",
            "deadline": "breach sla", "deadlines": "breach sla", "late": "late breach", "tomorrow": "forecast expected", "first": "recommendation",
            "busy": "overloaded busy", "overload": "overloaded forecast", "overloaded": "forecast", "coming": "forecast expected", "accurate": "model auc", "good": "model auc",
            "why": "root cause", "cause": "root cause", "team": "group"}


def _stem(word: str) -> str:
    """Very small stemmer: 'overloaded' and 'overload' become the same word."""
    for end in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(end):
            return word[: -len(end)]
    return word


def _words(text: str) -> str:
    out = []
    for w in re.findall(r"\w+", text.lower()):
        out.append(w)
        out.extend(SYNONYMS.get(w, "").split())
    return " ".join(_stem(w) for w in out)


def search(question: str, cards: list, k: int = 5) -> list:
    texts = [_words(c["id"].replace(":", " ").replace("_", " ") + " " + c["text"]) for c in cards]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    m = vec.fit_transform(texts + [_words(question)])
    scores = linear_kernel(m[-1], m[:-1]).ravel()
    best = scores.argsort()[::-1][:k]
    return [{**cards[i], "match": round(float(scores[i]), 3)} for i in best if scores[i] >= MIN_MATCH]


# --------------------------------------------------------------------------- 3. answer and check
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITATION = re.compile(r"\[[^\]]*\]")


def _numbers(text: str) -> set:
    text = CITATION.sub(" ", text)                     # [REC2] is a citation, not the number 2
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def check_numbers(answer: str, cited: list) -> list:
    """Numbers in the answer that appear in none of the cited cards (empty list = all numbers verified)."""
    allowed = set().union(*[_numbers(c["text"]) for c in cited]) if cited else set()
    return sorted(n for n in _numbers(answer) if not any(abs(n - a) <= 0.05 for a in allowed))


def ollama(prompt: str) -> str | None:
    """Ask the local LLM. Returns None if Ollama is not running, so the caller falls back to a template."""
    try:
        import requests
        r = requests.post(OLLAMA_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "format": "json"},
                          timeout=120)
        r.raise_for_status()
        return r.json()["response"]
    except Exception:
        return None


PROMPT = """You answer questions for a business manager using ONLY the facts below.
Rules: use only numbers that appear in the facts; put the fact ID in square brackets after each claim, like [RC1];
if the facts do not answer the question, set "answer" to "Not enough evidence in this data to answer that."
Reply as JSON: {{"answer": "...", "cited_ids": ["..."]}}

Facts:
{facts}

Question: {question}"""


def template_answer(question: str, found: list) -> dict:
    if not found:
        return {"answer": "Not enough evidence in this data to answer that.", "cited_ids": [], "verified": True,
                "written_by": "template"}
    text = "Here is what the data says: " + " ".join(f"{c['text']} [{c['id']}]" for c in found[:3])
    return {"answer": text, "cited_ids": [c["id"] for c in found[:3]], "verified": True, "written_by": "template"}


def answer(question: str, analysis: dict, llm=ollama) -> dict:
    cards = fact_cards(analysis)
    found = search(question, cards)
    if not found:
        return {**template_answer(question, found), "sources": []}
    facts = "\n".join(f"[{c['id']}] {c['text']}" for c in found)
    for _ in range(2):                                  # first try, then one retry
        raw = llm(PROMPT.format(facts=facts, question=question)) if llm else None
        if raw is None:
            break                                       # LLM off: use the template
        try:
            reply = json.loads(raw)
            text, ids = reply["answer"], [i for i in reply.get("cited_ids", []) if any(c["id"] == i for c in found)]
        except (ValueError, KeyError, TypeError):
            continue
        cited = [c for c in found if c["id"] in ids]
        bad = check_numbers(text, cited)
        if not bad and (ids or text.startswith("Not enough evidence")):
            return {"answer": text, "cited_ids": ids, "verified": True, "written_by": LLM_MODEL, "sources": found}
    return {**template_answer(question, found), "sources": found,
            "note": "The AI writer was off or its answer could not be verified, so the facts are shown directly."}
