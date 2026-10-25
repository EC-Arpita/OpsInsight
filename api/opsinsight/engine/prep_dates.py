"""One date order for the whole file, and plain warnings when dates are ambiguous or have no time of day."""
import pandas as pd

from ..prep.mapping import DATE_FIELDS
from .uploads import date_order, has_times, to_dates

UNCLEAR = ("Dates such as 05/03/2016 could mean 5 March or 3 May. They were read day first (5 March). "
           "If your system writes the month first, choose 'Month first' on the upload screen and run again.")
NO_TIME = ("The dates have no time of day, so durations are counted in whole days "
           "(a ticket resolved the same day shows 0 hours) and SLA hours are rounded up to whole days.")


def date_check(raw: pd.DataFrame, mapping: dict, chosen: str = None, times_matter: bool = True) -> tuple:
    """Returns (order to use, warnings). chosen = the analyst's 'day' or 'month', which always wins."""
    cols = [col for std, col in mapping.items() if std in DATE_FIELDS and col in raw]
    values = pd.concat([raw[c] for c in cols]) if cols else pd.Series(dtype=str)
    found = date_order(values)
    order = chosen or ("day" if found == "unclear" else found)
    warnings = []
    if found == "unclear" and not chosen:
        warnings.append({"level": "warning", "message": UNCLEAR})
    if times_matter and len(values.dropna()) and not has_times(to_dates(values, order)):
        warnings.append({"level": "warning", "message": NO_TIME})
    return order, warnings
