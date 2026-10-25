"""Does Ask find the right fact card? A small labelled set of questions -> the card that answers each.

hit@5 = share of questions whose right card is in the top 5. Tested on the word search (the backup that always
runs, no model download). To score the vector search too: hit_rate("sales", rag.vector_search).
"""
import json

from opsinsight.engine.rag import fact_cards, word_search
from opsinsight.models.config import ROOT

EVAL = {
    "incidents": [
        ("What is the SLA breach rate?", "KPI:sla_breach_rate"),
        ("How long does it take to resolve a ticket?", "KPI:median_resolution_hours"),
        ("How many tickets did we receive?", "KPI:ticket_volume"),
        ("Why are tickets reassigned between teams?", "RC1"),
        ("Is Group 20 under strain?", "RC2"),
        ("Do reopened tickets breach more?", "RC3"),
        ("Will Group 65 be overloaded next week?", "FC:Group 65"),
        ("How good is the risk model?", "MODEL"),
        ("Which week was unusual in March?", "AN:"),
        ("What should we do about high-risk tickets?", "REC3"),
    ],
    "sales": [
        ("What is the late delivery rate?", "KPI:late_delivery_rate"),
        ("What is our revenue?", "KPI:revenue"),
        ("What is the average review score?", "KPI:avg_review_score"),
        ("Why are sellers dispatching slowly?", "RC1"),
        ("Which region gets the most late deliveries?", "RC2"),
        ("Do late orders hurt reviews?", "RC4"),
        ("How many orders will Southeast get next week?", "FC:Southeast"),
        ("How late is health_beauty?", "BD:main_category:health_beauty"),
    ],
}


def hit_rate(domain, search=word_search):
    cards = fact_cards(json.loads((ROOT / f"contracts/example_{domain}.json").read_text(encoding="utf-8")))
    misses = [q for q, want in EVAL[domain] if not any(c["id"].startswith(want) for c in search(q, cards, 5))]
    return 1 - len(misses) / len(EVAL[domain]), misses


def test_word_search_finds_the_right_card():
    for domain in EVAL:
        rate, misses = hit_rate(domain)
        print(f"{domain}: hit@5 {rate:.0%}  missed: {misses}")
        assert rate >= 0.8, misses
