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