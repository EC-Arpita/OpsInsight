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

    # capacity = tickets each team resolved per day over the last 28 days
    done = tickets[(tickets["resolved_at"] > end - pd.Timedelta(days=28)) & (tickets["resolved_at"] <= end + pd.Timedelta(days=1))]
    capacity = done.groupby("assignment_group").size().reindex(teams, fill_value=0) / 28

    days = pd.date_range(end + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    out = []
    for j, team in enumerate(teams):
        daily = [
            {
                "date": f"{d:%Y-%m-%d}",
                "expected_tickets": round(float(future[i, j]), 1),
                "expected_capacity": round(float(capacity[team]), 1),
                "overload": bool(future[i, j] > capacity[team] * overload_margin),
            }
            for i, d in enumerate(days)
        ]
        out.append(
            {
                "team": team,
                "expected_tickets_7d": round(float(future[:, j].sum())),
                "expected_capacity_7d": round(float(capacity[team] * 7)),
                "overloaded_days": sum(x["overload"] for x in daily),
                "daily": daily,
            }
        )
    return {
        "method": best,
        "horizon_days": HORIZON,
        "backtest_mean_abs_error": scores.to_dict(orient="records"),
        "capacity_rule": "average tickets the team resolved per day over the last 28 days",
        "overload_rule": f"expected tickets > capacity x {overload_margin}",
        "teams": out,
    }