"""
Survival estimators for monthly (discrete-time) loan data, written out explicitly.

With competing risks (default vs prepayment), the probability of defaulting by month t is the
cumulative incidence function (Aalen-Johansen estimator):

    CIF_k(t) = sum_{s <= t} S(s-1) * d_k(s) / n(s)

where n(s) loans are still at risk at the start of month s, d_k(s) end in cause k during month s,
and S(s) = prod_{u <= s} (1 - d(u) / n(u)) is the probability of having had no event at all.

The common shortcut, 1 - Kaplan-Meier with prepayments treated as censored, assumes that a
prepaid loan "would have defaulted at the same rate as the others", which is false: a refinanced
loan cannot default. It overstates the default probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _counts(time: np.ndarray, event: np.ndarray, horizon: int) -> pd.DataFrame:
    months = np.arange(1, horizon + 1)
    ends = pd.crosstab(time, event).reindex(months, fill_value=0)
    for k in (0, 1, 2):
        if k not in ends.columns:
            ends[k] = 0
    # at risk at the start of month s: loans whose observed time is >= s
    at_risk = np.array([(time >= s).sum() for s in months])
    return pd.DataFrame({"month": months, "n": at_risk, "d_default": ends[1].values,
                         "d_prepay": ends[2].values, "censored": ends[0].values})


def cumulative_incidence(time, event, horizon: int | None = None) -> pd.DataFrame:
    """Aalen-Johansen cumulative incidence of default (event 1) and prepayment (event 2),
    plus 1 - Kaplan-Meier for default treating prepayment as censoring, for comparison."""
    time, event = np.asarray(time), np.asarray(event)
    horizon = horizon or int(time.max())
    t = _counts(time, event, horizon)
    n = t["n"].replace(0, np.nan)
    surv_all = (1 - (t["d_default"] + t["d_prepay"]) / n).fillna(1).cumprod()
    surv_prev = surv_all.shift(1, fill_value=1.0)
    t["cif_default"] = (surv_prev * t["d_default"] / n).fillna(0).cumsum()
    t["cif_prepay"] = (surv_prev * t["d_prepay"] / n).fillna(0).cumsum()
    t["one_minus_km_default"] = 1 - (1 - t["d_default"] / n).fillna(1).cumprod()
    t["still_active"] = surv_all
    return t


def cif_at(df: pd.DataFrame, months=(12, 36, 60, 108), by=None) -> pd.DataFrame:
    """Cumulative incidence of default at selected months, optionally by group."""
    groups = [((), df)] if by is None else df.groupby(by)
    rows = []
    for key, g in groups:
        c = cumulative_incidence(g["time"], g["event"], max(months)).set_index("month")
        row = {f"{m} months": c.loc[m, "cif_default"] for m in months}
        if by is not None:
            row.update(dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,))))
        row["loans"] = len(g)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.set_index(by) if by is not None else out
