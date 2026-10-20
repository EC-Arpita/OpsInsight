"""Small helpers shared by the incident and sales pipelines."""
import numpy as np
import pandas as pd


class QualityLog:
    """Collects one line per cleaning step: rows before/after and why (notebook 02's log())."""

    def __init__(self):
        self.rows = []

    def add(self, step, before, after, note=""):
        self.rows.append({"step": step, "rows_before": int(before), "rows_after": int(after),
                          "rows_removed": int(before - after), "note": note})

    def to_frame(self):
        return pd.DataFrame(self.rows)


def safe_merge(left, right, on, name):
    """Left join that raises an error if it would duplicate rows (notebook 05)."""
    before = len(left)
    out = left.merge(right, on=on, how="left")
    if len(out) != before:
        raise ValueError(f"Join with {name} changed rows {before} -> {len(out)}: key not unique")
    return out


def count_in_window(times_sorted, query_times, window):
    """For each query time q: how many of times_sorted fall in [q - window, q) (notebook 03)."""
    right = np.searchsorted(times_sorted, query_times, side="left")
    left = np.searchsorted(times_sorted, query_times - window, side="left")
    return right - left


def past_mean(df, key, value_col, known_col, query_col, prior, m=20, window=None):
    """Smoothed mean of value_col over earlier rows with the same key whose value was
    already known (known_col) before this row's query_col. No leakage (notebooks 03, 06).
    Returns (smoothed_mean, n_known)."""
    out_mean = np.full(len(df), np.nan)
    out_n = np.zeros(len(df), dtype=int)
    groups = df.groupby(key).groups if key else {"all": df.index}
    for _, idx in groups.items():
        sub = df.loc[idx]
        ok = sub[known_col].notna().values & sub[value_col].notna().values
        order = np.argsort(sub[known_col].values[ok])
        known_times = sub[known_col].values[ok][order]
        cum = np.concatenate([[0.0], np.cumsum(sub[value_col].values[ok][order])])
        q = sub[query_col].values
        hi = np.searchsorted(known_times, q, side="left")
        lo = np.zeros_like(hi) if window is None else np.searchsorted(known_times, q - window, side="left")
        n = hi - lo
        pos = df.index.get_indexer(idx)
        out_n[pos] = n
        out_mean[pos] = (cum[hi] - cum[lo] + m * prior) / (n + m)
    return out_mean, out_n


def top_n_or_other(df, col, n, train_mask):
    """Keep the n most common values (learned on training rows only); others -> 'Other'."""
    top = df.loc[train_mask, col].value_counts().head(n).index
    return df[col].where(df[col].isin(top), "Other").astype("category")


class FeatureDictionary:
    """Documents every feature as it is created (notebook 03's describe())."""

    def __init__(self, known_at):
        self.known_at = known_at
        self.rows = []

    def add(self, name, how, group):
        self.rows.append({"feature": name, "how_it_is_computed": how, "group": group,
                          f"known_at_{self.known_at}": "yes"})

    @property
    def names(self):
        return [r["feature"] for r in self.rows]

    def to_frame(self):
        return pd.DataFrame(self.rows)