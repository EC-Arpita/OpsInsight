"""The workload plan finds the best rule-respecting plan. Run from the api folder:  python -m pytest tests -q"""
import pandas as pd

from opsinsight.engine.optimiser import workload_plan


def _forecast(day):
    """day = {team: (expected tickets, capacity)} for one day."""
    return {"teams": [{"team": t, "daily": [{"date": "2026-01-05", "expected_tickets": e, "expected_capacity": c}]}
                      for t, (e, c) in day.items()]}


def test_plan_beats_first_fit():
    # A and B are overloaded; C knows both kinds of tickets, D only A's. Taking helpers in order (first fit) would
    # send A's tickets to C and leave B uncovered. The best plan sends A to D and B to C: nobody is left over.
    tickets = pd.DataFrame({"assignment_group": ["A"] * 30 + ["B"] * 30 + ["C"] * 40 + ["D"] * 20,
                            "category": ["x"] * 30 + ["y"] * 30 + ["x"] * 20 + ["y"] * 20 + ["x"] * 20})
    plan = workload_plan(_forecast({"A": (20, 10), "B": (19, 10), "C": (0, 10), "D": (0, 10)}), tickets)
    assert plan["staffing_gaps"] == []
    assert {(m["from_team"], m["to_team"], m["tickets"]) for m in plan["moves"]} == {("A", "D", 10), ("B", "C", 9)}
    assert all(plan["check"].values())


def test_only_low_risk_tickets_move():
    # A is 10 over but only 60% of its 12 expected tickets are low risk: 7 move, 3 need cover.
    tickets = pd.DataFrame({"assignment_group": ["A"] * 30 + ["B"] * 30, "category": ["x"] * 60})
    plan = workload_plan(_forecast({"A": (12, 2), "B": (0, 20)}), tickets)
    assert sum(m["tickets"] for m in plan["moves"]) == 7
    assert plan["staffing_gaps"][0]["tickets"] == 3
