"""Shared forecasting tools: daily counts, four simple methods + AutoETS and MSTL, and the backtest that picks one."""
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
    res = h.iloc[-7:].to_numpy()
    if res.shape[0] < HORIZON:
        repeat_count = int(np.ceil(HORIZON / max(1, res.shape[0])))
        res = np.tile(res, (repeat_count, 1))[:HORIZON]
    return res


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


def statsforecast_method(model):
    """Wrap a statsforecast model (fitted per group on its own history) as a method: history -> next 7 days."""
    from statsforecast import StatsForecast

    def method(h):
        long = h.rename_axis("ds").reset_index().melt("ds", var_name="unique_id", value_name="y")
        fc = StatsForecast(models=[model], freq="D").forecast(df=long, h=HORIZON)
        wide = fc.pivot(index="ds", columns="unique_id", values=fc.columns[-1])
        return np.clip(wide[list(h.columns)].to_numpy(), 0, None)   # counts cannot go below 0
    return method


METHODS = {
    "same day last week": same_day_last_week,
    "flat 4-week average": flat_average_4w,
    "weekday average (4 weeks)": weekday_average_4w,
    "weekday average x recent trend": weekday_average_x_trend,
}
try:    # industry models; if statsforecast is not installed the four simple methods still run
    from statsforecast.models import MSTL, AutoETS
    METHODS["AutoETS (Holt-Winters, auto-tuned)"] = statsforecast_method(AutoETS(season_length=7))
    METHODS["MSTL (trend + weekly pattern)"] = statsforecast_method(MSTL(season_length=7))
except ImportError:
    pass


def backtest(counts: pd.DataFrame, weeks=8, min_history=28) -> pd.DataFrame:
    """Average error (items per group per day) of each method, forecasting 7 days from each of the last `weeks`
    weeks (fewer if the file is short, so every method still has 4 weeks of history)."""
    origins = [7 * k for k in range(weeks, 0, -1) if len(counts) - 7 * k >= min_history] or [7]
    rows = []
    for name, f in METHODS.items():
        errors = []
        for back in origins:
            hist, actual = counts.iloc[:-back], counts.iloc[-back:][:HORIZON]
            try:
                errors.append(np.abs(f(hist) - actual.to_numpy()).mean())
            except Exception:   # a model that cannot fit this history (too short, all zeros) just drops out
                errors = []
                break
        if errors:
            rows.append({"method": name, "mean_abs_error": round(float(np.mean(errors)), 2)})
    # shortcut: with under ~2 weeks of history no method can be backtested; callers then use FALLBACK
    return pd.DataFrame(rows, columns=["method", "mean_abs_error"]).sort_values("mean_abs_error").reset_index(drop=True)


FALLBACK = "flat 4-week average"     # used when the file is too short to compare methods
