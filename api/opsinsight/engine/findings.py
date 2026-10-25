"""Decision engine: turn every result in the analysis into a finding card, score it, rank it, explain it.

score = size x confidence x urgency
    size       share of all problems the finding affects (0 to 1)
    confidence high = 1.0, medium = 0.7, low = 0.5
    urgency    happening now or next week = 1.0; already over = 0 (kept as history, not ranked)
"""
CONFIDENCE = {"high": 1.0, "medium": 0.7, "low": 0.5}
WATCH_BELOW = 0.03          # cards scoring less than this go to the watch list
WORDS = {"incidents": {"share": "share_of_breaches_current_period", "problems": "breaches", "items": "tickets"},
         "sales": {"share": "share_of_late_current_period", "problems": "late orders", "items": "orders"}}


def _card(id, kind, title, finding, size, confidence, urgency, subject=None, evidence=None, source=None):
    score = None if size is None else round(size * CONFIDENCE[confidence] * urgency, 3)
    return {"id": id, "type": kind, "title": title, "finding": finding, "subject": subject, "size": size,
            "confidence": confidence, "urgency": urgency, "score": score, "evidence": evidence or [], "source": source}


def _root_cause_cards(analysis, domain):
    w = WORDS[domain]
    cards = []
    for rc in analysis["root_causes"]:
        if rc["category"] == "customer":       # explains why the problem matters (e.g. reviews); not a cause to fix
            continue
        share = rc.get(w["share"])
        subject = rc["evidence"][0].get("segment") if rc["category"] in ("capacity", "logistics") else None
        cards.append(_card(rc["id"], rc["category"], rc["title"], rc["finding"], share, rc["confidence"], 1.0,
                           subject, rc["evidence"], "root_causes"))
    return cards


def _risk_card(analysis, domain):
    mp = analysis["model_performance"]
    w = WORDS[domain]
    fit = mp.get("on_your_data")                     # reference model: judge it on this company's data, not its own
    lift = fit["lift_top_10pct"] if fit else mp["lift_top_10pct"]
    if lift is None:
        return _card("RISK", "high_risk", f"High-risk new {w['items']}", "The risk model could not be checked on your data.",
                     None, "low", 1.0, None, [], "predictions")
    size = round(lift * 0.10, 3)                     # share of all problems found in the riskiest 10%
    conf = "low" if fit else "high" if lift >= 2 else "medium"
    return _card("RISK", "high_risk", f"High-risk new {w['items']}",
                 f"The 10% of {w['items']} the model flags hold {size:.0%} of all {w['problems']} "
                 f"({lift}x the normal rate)." + (" Measured on your data with a borrowed model." if fit else ""), size, conf, 1.0, None,
                 [{"precision_top_10pct": mp["precision_top_10pct"], "base_rate": mp["base_rate_test"]}], "predictions")


def _forecast_cards(analysis, domain):
    fc = analysis.get("forecast") or {}
    cards = []
    if domain == "incidents":
        groups = fc.get("teams", [])
        total = sum(g["expected_tickets_7d"] for g in groups) or 1
        for g in groups:
            days = [d for d in g["daily"] if d["overload"]]
            if days:
                extra = sum(d["expected_tickets"] - d["expected_capacity"] for d in days)
                when = ", ".join(d["date"] for d in days)
                cards.append(_card(f"FC-{g['team']}", "overload", f"{g['team']} overloaded next week",
                                   f"{g['team']} is expected to get about {extra:.0f} tickets more than it can handle on {when}.",
                                   round(extra / total, 3), "medium", 1.0, g["team"], days, "forecast"))
    else:
        groups = fc.get("regions", [])
        total = sum(g["expected_orders_7d"] for g in groups) or 1
        for g in groups:
            days = [d for d in g["daily"] if d["busy"]]
            if days:
                extra = sum(d["expected_orders"] - d["usual_orders"] for d in days)
                when = ", ".join(d["date"] for d in days)
                cards.append(_card(f"FC-{g['region']}", "busy", f"{g['region']} unusually busy next week",
                                   f"{g['region']} is expected to get about {extra:.0f} orders more than usual on {when}.",
                                   round(extra / total, 3), "medium", 1.0, g["region"], days, "forecast"))
    return cards


def _anomaly_cards(analysis):
    end = analysis["meta"]["data_period"]["end"]
    cards = []
    for a in analysis.get("anomalies", []):
        recent = _days_between(a["week_ending"], end) <= 14         # in the last two weeks of the data
        cards.append(_card(f"AN-{a['week_ending']}", "anomaly", f"{a['label']} unusual in the week to {a['week_ending']}",
                           f"{a['actual']:.1%} against a normal {a['expected']:.1%} (deviation score {a['deviation_score']}).",
                           min(1.0, a["deviation_score"] / 10), "high" if a["severity"] == "critical" else "medium",
                           1.0 if recent else 0.0, None, [a], "anomalies"))
    return cards


def _days_between(a, b):
    from datetime import date
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


SIZE_WORDS = {"overload": "of next week's expected tickets are extra work above capacity",
              "busy": "of next week's expected orders are extra above the usual level",
              "high_risk": "of all problems sit in the riskiest 10% of items"}


def _why(card, rank):
    if card["type"] == "anomaly" and card["urgency"] == 0:
        return "Kept as history: this unusual week is over."
    if card["score"] is None:
        return "Not ranked by number: its size is not measured yet, so it follows the measured findings."
    what = SIZE_WORDS.get(card["type"], "of this period's problems are linked to it")
    return f"Ranked {rank}: {card['size']:.0%} {what}; confidence {card['confidence']} (score {card['score']})."


def rank_findings(analysis: dict, domain: str) -> dict:
    cards = _root_cause_cards(analysis, domain) + [_risk_card(analysis, domain)] + \
        _forecast_cards(analysis, domain) + _anomaly_cards(analysis)
    # one problem, two pieces of evidence: merge a forecast card into the root cause about the same team
    by_subject = {c["subject"]: c for c in cards if c["source"] == "root_causes" and c["type"] == "capacity"}
    merged = []
    for c in cards:
        if c["source"] == "forecast" and c["subject"] in by_subject:
            main = by_subject[c["subject"]]
            main["evidence"] = main["evidence"] + [{"forecast": c["finding"]}]
            main["related"] = c["id"]
        else:
            merged.append(c)
    history = [c for c in merged if c["urgency"] == 0]
    live = [c for c in merged if c["urgency"] > 0]
    measured = sorted([c for c in live if c["score"] is not None], key=lambda c: -c["score"])
    unmeasured = [c for c in live if c["score"] is None]
    ranked = [c for c in measured if c["score"] >= WATCH_BELOW] + unmeasured
    watch = [c for c in measured if c["score"] < WATCH_BELOW]
    for i, c in enumerate(ranked, start=1):
        c["rank"], c["why"] = i, _why(c, i)
    for c in watch + history:
        c["rank"], c["why"] = None, _why(c, None) if c in history else "On the watch list: too small to act on now."
    context = [rc for rc in analysis["root_causes"] if rc["category"] == "customer"]
    return {"rule": "score = size x confidence x urgency; confidence high 1.0, medium 0.7, low 0.5",
            "items": ranked, "watch": watch, "history": history, "context": context,
            "note": "Findings are correlations in historical data; the simulation estimates the effect of acting on them."}
