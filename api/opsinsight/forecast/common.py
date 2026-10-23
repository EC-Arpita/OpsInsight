"""Shared forecasting tools: daily counts, four simple methods, and the backtest that picks one."""
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 7  # forecast the next 7 days


def project_root() -> Path:
    """The opsinsight folder (the one with requirements.txt), wherever this file is run from."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "requirements.txt").exists():
            return parent
    raise FileNotFoundError("could not find the project folder (the one with requirements.txt)")


ROOT = project_root()
MODELS_DIR = ROOT / "models"


def daily_counts(df: pd.DataFrame, time_col: str, group_col: str, groups) -> pd.DataFrame:
    """One row per day, one column per group, number of items that day (0 on quiet days)."""
    d = df[df[group_col].isin(groups)]
    out = d.groupby([d[time_col].dt.normalize(), group_col]).size().unstack(fill_value=0)
    days = pd.date_range(df[time_col].min().normalize(), df[time_col].max().normalize(), freq="D")
    return out.reindex(days, fill_value=0).reindex(columns=list(groups), fill_value=0)



# --- forecasting methods: history (days x groups) -> next 7 days (7 x groups) ---
def same_day_last_week(h):
    return h.iloc[-7:].to_numpy()


def flat_average_4w(h):
    return np.tile(h.iloc[-28:].mean().to_numpy(), (HORIZON, 1))


def weekday_average_4w(h):
    last = h.iloc[-28:]
    profile = last.groupby(last.index.dayofweek).mean()
    days = pd.date_range(h.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    return profile.reindex(days.dayofweek).to_numpy()


def weekday_average_x_trend(h):
    """Weekday shape from 4 weeks, scaled by how busy the last week was compared with those 4 weeks."""
    base = weekday_average_4w(h)
    level = h.iloc[-7:].mean() / h.iloc[-28:].mean().replace(0, np.nan)
    return base * level.fillna(1).clip(0.5, 1.5).to_numpy()


METHODS = {
    "same day last week": same_day_last_week,
    "flat 4-week average": flat_average_4w,
    "weekday average (4 weeks)": weekday_average_4w,
    "weekday average x recent trend": weekday_average_x_trend,
}




def backtest(counts: pd.DataFrame, origins=(14, 7)) -> pd.DataFrame:
    """Average error (items per group per day) of each method when forecasting from 14 and 7 days before the end."""
    rows = []
    for name, f in METHODS.items():
        errors = []
        for back in origins:
            hist, actual = counts.iloc[:-back], counts.iloc[-back:][:HORIZON]
            errors.append(np.abs(f(hist) - actual.to_numpy()).mean())
        rows.append({"method": name, "mean_abs_error": round(float(np.mean(errors)), 2)})
    return pd.DataFrame(rows).sort_values("mean_abs_error").reset_index(drop=True)