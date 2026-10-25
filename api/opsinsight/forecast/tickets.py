"""Support tickets: expected tickets per team for the next 7 days, and which days will overload a team."""
import pandas as pd

from .common import HORIZON, METHODS, ROOT, backtest, daily_counts

TICKETS_FILE = ROOT / "data/interim/incidents_clean.parquet"


def forecast_tickets(tickets: pd.DataFrame = None, n_teams=5, overload_margin=1.1) -> dict:
    """The `forecast` section of the ticket analysis JSON."""
    tickets = pd.read_parquet(TICKETS_FILE) if tickets is None else tickets
    end = tickets["opened_at"].max().normalize()
    recent = tickets[tickets["opened_at"] >= end - pd.Timedelta(days=21)]
    teams = list(recent["assignment_group"].value_counts().head(n_teams).index)
    opened = daily_counts(tickets, "opened_at", "assignment_group", teams)

    scores = backtest(opened)
    best = scores.iloc[0]["method"]
    future = METHODS[best](opened)

    # capacity = tickets each team resolved on that weekday, averaged over the last 4 weeks (a Monday is not a Sunday)
    start, stop = end + pd.Timedelta(days=1) - pd.Timedelta(days=28), end + pd.Timedelta(days=1)
    done = tickets[(tickets["resolved_at"] >= start) & (tickets["resolved_at"] < stop)]
    by_day = done.groupby([done["assignment_group"], done["resolved_at"].dt.dayofweek]).size() / 4
    cap = lambda team, day: float(by_day.get((team, day.dayofweek), 0.0))

    days = pd.date_range(end + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    out = []
    for j, team in enumerate(teams):
        daily = [{"date": f"{d:%Y-%m-%d}", "expected_tickets": round(float(future[i, j]), 1),
                  "expected_capacity": round(cap(team, d), 1),
                  "overload": bool(future[i, j] > cap(team, d) * overload_margin and future[i, j] - cap(team, d) >= 1)}
                 for i, d in enumerate(days)]
        out.append({"team": team, "expected_tickets_7d": round(float(future[:, j].sum())),
                    "expected_capacity_7d": round(sum(cap(team, d) for d in days)),
                    "overloaded_days": sum(x["overload"] for x in daily), "daily": daily})
    return {"method": best, "horizon_days": HORIZON, "backtest_mean_abs_error": scores.to_dict(orient="records"),
            "capacity_rule": "tickets the team resolved on the same weekday, averaged over the last 4 weeks",
            "overload_rule": f"expected tickets > capacity x {overload_margin} and at least 1 ticket more", "teams": out}
