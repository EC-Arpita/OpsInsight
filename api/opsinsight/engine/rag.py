"""RAG for 'Ask your data' and written explanations: find the right facts first, then answer only from them.

1. fact_cards(): every result in the analysis becomes one short sentence with an ID (the knowledge base).
2. search():     find the cards closest to the question by MEANING: ChromaDB vector database with its built-in
                 embedding model (all-MiniLM-L6-v2, downloaded once, about 80 MB, then offline).
                 If ChromaDB or the model is not available, it falls back to TF-IDF word matching.
3. answer():     an LLM (Ollama, on this laptop) writes the answer from those cards and cites their IDs;
                 check_numbers() confirms every number appears in a cited card; otherwise a template answer.
"""
import hashlib
import json
import logging
import os
import re
from functools import lru_cache

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
                               + (f"overloaded on all {len(days)} days ({days[0]} to {days[-1]})." if days and len(days) == len(g["daily"])
                                  else f"overloaded on {', '.join(days)}." if days else "within capacity every day."))
    over = sorted(((g["team"], sum(d["overload"] for d in g["daily"])) for g in fc.get("teams", [])), key=lambda x: -x[1])
    if any(n for _, n in over):     # one card naming every overloaded team, so the answer never lists just one
        add("FC:overloaded", "Teams expected to be overloaded next week: "
            + ", ".join(f"{t} on {n} of 7 days" for t, n in over if n) + ".")
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
    pk = (analysis.get("predictions_at_pickup") or {}).get("model_performance")
    if pk:
        add("MODEL_PICKUP", f"Second check when the carrier picks the order up: that model has test AUC {pk['test_auc']}; "
                            f"the riskiest 10% of picked-up orders have a late rate of {_pct(pk['precision_top_10pct'])} "
                            f"against {_pct(pk['base_rate_test'])} overall ({pk['lift_top_10pct']}x).")
    tickets = analysis["meta"]["domain"] == "support_tickets"
    _risk_cards(add, analysis.get("predictions"), "RISK", "Highest-risk tickets right now, most likely to miss their SLA"
                if tickets else "Highest-risk orders right now, most likely to be late")
    _risk_cards(add, analysis.get("predictions_at_pickup"), "RISK_PICKUP",
                "Highest-risk orders already with the courier, most likely to be late")
    return cards


TOP_RISK = 5     # one card for the list, one per item for "why is this one risky?"


def _risk_cards(add, section, prefix, heading):
    """The riskiest items as fact cards, so Ask can answer 'which ticket is at risk, and why?'."""
    items = (section or {}).get("items", [])[:TOP_RISK]
    if not items:
        return
    where = lambda c: c.get("team") or " to ".join(x for x in (c.get("seller_state"), c.get("customer_state")) if x)
    # the % as written in the item's own explanation, so both cards show the same number
    pct = lambda it: (re.search(r"\((\d+%)\)", it["explanation_summary"]) or [None, f"{it['prediction']['risk']:.0%}"])[1]
    add(f"{prefix}:top", f"{heading} (risk = chance of the problem): " + "; ".join(
        f"{it['id']} {pct(it)}" + (f" ({where(it['context'])})" if where(it["context"]) else "")
        for it in items) + ".")
    for it in items:
        add(f"{prefix}:{it['id']}", f"{it['item_type'].capitalize()} {it['id']}: {it['explanation_summary']}")


# --------------------------------------------------------------------------- 2. search
MIN_MATCH = 0.05          # below this, a card is not really about the question
# Words a manager uses -> words the fact cards use, so "what should we do" finds the recommendations.
SYNONYMS = {"do": "recommendation", "fix": "recommendation", "action": "recommendation", "should": "recommendation",
            "deadline": "breach sla", "deadlines": "breach sla", "late": "late breach", "tomorrow": "forecast expected", "first": "recommendation",
            "busy": "overloaded busy", "overload": "overloaded forecast", "overloaded": "forecast", "coming": "forecast expected", "accurate": "model auc", "good": "model auc",
            "why": "root cause", "cause": "root cause", "team": "group",
            "risky": "risk high", "riskiest": "risk high", "danger": "risk high", "which": "highest"}


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


def word_search(question: str, cards: list, k: int = 5) -> list:
    """Backup search: TF-IDF word matching (no model needed)."""
    texts = [_words(c["id"].replace(":", " ").replace("_", " ") + " " + c["text"]) for c in cards]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    m = vec.fit_transform(texts + [_words(question)])
    scores = linear_kernel(m[-1], m[:-1]).ravel()
    best = scores.argsort()[::-1][:k]
    return [{**cards[i], "match": round(float(scores[i]), 3), "search": "words"} for i in best if scores[i] >= MIN_MATCH]


VECTOR_MIN = 0.30   # cosine similarity; shortcut: set by eye for all-MiniLM-L6-v2, re-check if the model changes


@lru_cache(maxsize=1)
def _client():
    import chromadb
    from ..models.config import ROOT
    return chromadb.PersistentClient(path=str(ROOT / "data/app/chroma"))


def _collection(cards: list, embed=None):
    """One collection per set of cards, so each company's facts stay separate; built once, then reused."""
    key = hashlib.md5(json.dumps([[c["id"], c["text"]] for c in cards]).encode()).hexdigest()[:20]
    extra = {"embedding_function": embed} if embed else {}
    col = _client().get_or_create_collection("cards_" + key, metadata={"hnsw:space": "cosine"}, **extra)
    if col.count() == 0:
        col.add(ids=[str(i) for i in range(len(cards))], documents=[c["text"] for c in cards])
    return col


def vector_search(question: str, cards: list, k: int = 5, embed=None) -> list:
    col = _collection(cards, embed)
    hits = col.query(query_texts=[question], n_results=min(k, len(cards)))
    out = []
    for i, dist in zip(hits["ids"][0], hits["distances"][0]):
        score = 1 - dist                                  # cosine distance -> similarity
        if score >= VECTOR_MIN:
            out.append({**cards[int(i)], "match": round(score, 3), "search": "vector"})
    return out


def search(question: str, cards: list, k: int = 5) -> list:
    if not cards:
        return []
    try:
        return vector_search(question, cards, k)
    except Exception as e:      # ChromaDB not installed or the model cannot download (offline): words still work
        logging.warning("Vector search unavailable (%s); using word search.", e)
        return word_search(question, cards, k)


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
answer in 2 to 4 full sentences: give the main number, name EVERY team, region, ticket or order the facts list
for this question (not just one), and say why when the facts give a reason;
if the question asks why but the facts only show what is happening (for example a team's problem rate), state what
the facts show and say that this data does not record the cause;
only if no fact is about the question, set "answer" to "Not enough evidence in this data to answer that."
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