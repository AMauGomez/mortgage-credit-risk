"""
Build one row per loan from Freddie Mac's Single-Family Loan-Level Dataset (sample files).

The raw files are NOT part of this repository (Freddie Mac's terms forbid redistributing
them). Download them yourself (free registration) and set the environment variable
FREDDIE_DIR, or pass `freddie_dir`, pointing to the folder that holds:

    sample_orig_YYYY.txt, sample_perf_YYYY.txt    (directly or inside a sample_YYYY/ folder)

All the heavy lifting is SQL run by DuckDB, an in-process database that queries the
pipe-delimited text files directly (no server, no loading everything into memory).
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

# Column names in file order (from Freddie Mac's file layout, July 2026).
ORIG_COLS = [
    "fico", "first_payment", "first_time_buyer", "maturity", "msa", "mi_pct", "units",
    "occupancy", "cltv", "dti", "upb", "ltv", "rate", "channel", "ppm", "amortization",
    "state", "property_type", "zip3", "loan_id", "purpose", "term", "n_borrowers",
    "seller", "super_conforming", "pre_harp_id", "program", "harp", "valuation",
    "interest_only", "vantage",
]
PERF_COLS = [
    "loan_id", "period", "current_upb", "dlq", "age", "remaining", "defect_date", "mod_flag",
    "zbc", "zb_date", "current_rate", "non_int_upb", "ddlpi", "mi_recoveries", "net_sales",
    "non_mi_recoveries", "expenses", "legal_costs", "maintenance_costs", "taxes_insurance",
    "misc_expenses", "actual_loss", "cum_mod_cost", "step_rate", "deferral", "eltv",
    "zb_removal_upb", "dlq_accrued_interest", "disaster", "assistance_plan",
    "period_mod_cost", "int_upb", "mi_cancel", "servicer", "cramdown",
]

# Zero Balance Codes that are credit events (user guide): third-party sale, short sale or
# charge-off, REO disposition. 01 = voluntary payoff (prepayment). 15/16/96 = loan sales and
# repurchases: neither a default nor a prepayment, so the loan is censored at that point.
CREDIT_EVENT_ZBC = ("02", "03", "09")


def _find(freddie_dir: Path, name: str) -> Path:
    hits = list(freddie_dir.glob(name)) + list(freddie_dir.glob(f"*/{name}"))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {freddie_dir}")
    return hits[0]


def _reader(path: Path, cols: list[str]) -> str:
    columns = ", ".join(f"'{c}': 'VARCHAR'" for c in cols)
    return (f"read_csv('{path.as_posix()}', delim='|', header=false, auto_detect=false, "
            f"null_padding=true, columns={{{columns}}})")


def build_loans(vintage: int, freddie_dir: str | Path | None = None) -> pd.DataFrame:
    """One row per loan: origination attributes + time to default / prepayment / censoring.

    Outcome definitions (loan age in months):
      * default_any   first month 90+ days delinquent, REO acquisition, or a credit-event ZBC
      * default_adj   same, but a 90+ delinquency only counts if the loan was NOT in a
                      forbearance plan or disaster-related delinquency that month
                      (COVID-19 payment relief is not a credit default)
      * prepay        voluntary payoff (ZBC 01)
      * otherwise     censored at the last month observed
    """
    freddie_dir = Path(freddie_dir or os.environ.get("FREDDIE_DIR", "~/Downloads/freddie")).expanduser()
    orig = _reader(_find(freddie_dir, f"sample_orig_{vintage}.txt"), ORIG_COLS)
    perf = _reader(_find(freddie_dir, f"sample_perf_{vintage}.txt"), PERF_COLS)
    events = ", ".join(f"'{z}'" for z in CREDIT_EVENT_ZBC)

    sql = f"""
    WITH p AS (
        SELECT loan_id,
               CAST(age AS INTEGER) AS age,
               CAST(period AS INTEGER) AS period,
               dlq,
               zbc,
               TRY_CAST(current_upb AS DOUBLE) AS current_upb,
               TRY_CAST(actual_loss AS DOUBLE) AS actual_loss,
               coalesce(assistance_plan, '') = 'F' OR coalesce(disaster, '') = 'Y' AS relief,
               dlq = 'RA' OR coalesce(TRY_CAST(dlq AS INTEGER), 0) >= 3 AS dlq90
        FROM {perf}
    ),
    first_any AS (
        SELECT loan_id, min(age) AS t_default_any FROM p
        WHERE dlq90 OR zbc IN ({events}) GROUP BY loan_id
    ),
    first_adj AS (
        SELECT loan_id, min(age) AS t_default_adj FROM p
        WHERE (dlq90 AND NOT relief) OR zbc IN ({events}) GROUP BY loan_id
    ),
    relief90 AS (   -- loans whose first 90+ delinquency happened under payment relief
        SELECT loan_id, min(age) AS t_relief90 FROM p WHERE dlq90 AND relief GROUP BY loan_id
    ),
    last AS (
        SELECT loan_id, max(age) AS t_last, max(period) AS last_period,
               max(zbc) FILTER (WHERE zbc IS NOT NULL AND zbc <> '') AS zbc,
               sum(actual_loss) AS actual_loss,
               max(current_upb) FILTER (WHERE age <= 1) AS upb_start
        FROM p GROUP BY loan_id
    ),
    upb_at_default AS (
        SELECT p.loan_id, max(p.current_upb) AS upb_default
        FROM p JOIN first_adj f ON p.loan_id = f.loan_id AND p.age = f.t_default_adj
        GROUP BY p.loan_id
    )
    SELECT o.loan_id,
           {vintage} AS vintage,
           TRY_CAST(nullif(o.fico, '9999') AS INTEGER) AS fico,
           TRY_CAST(nullif(o.ltv, '999') AS DOUBLE) AS ltv,
           TRY_CAST(nullif(o.cltv, '999') AS DOUBLE) AS cltv,
           TRY_CAST(nullif(o.dti, '999') AS DOUBLE) AS dti,
           TRY_CAST(o.rate AS DOUBLE) AS rate,
           TRY_CAST(o.upb AS DOUBLE) AS upb,
           TRY_CAST(o.term AS INTEGER) AS term,
           TRY_CAST(o.n_borrowers AS INTEGER) AS n_borrowers,
           o.first_time_buyer, o.occupancy, o.channel, o.purpose, o.property_type, o.state,
           CAST(substr(o.first_payment, 1, 4) AS INTEGER) AS first_payment_year,
           CAST(o.first_payment AS INTEGER) AS first_payment_period,
           l.t_last, l.last_period, l.zbc, l.actual_loss,
           a.t_default_any, j.t_default_adj, r.t_relief90, u.upb_default
    FROM {orig} o
    LEFT JOIN last l USING (loan_id)
    LEFT JOIN first_any a USING (loan_id)
    LEFT JOIN first_adj j USING (loan_id)
    LEFT JOIN relief90 r USING (loan_id)
    LEFT JOIN upb_at_default u USING (loan_id)
    """
    df = duckdb.sql(sql).df()

    # Competing-risks outcome on the adjusted default definition:
    # event = 1 default, 2 prepayment, 0 censored; time = months to that event.
    df["event"] = 0
    df["time"] = df["t_last"]
    is_default = df["t_default_adj"].notna()
    is_prepay = ~is_default & (df["zbc"] == "01")
    df.loc[is_default, "event"] = 1
    df.loc[is_default, "time"] = df.loc[is_default, "t_default_adj"]
    df.loc[is_prepay, "event"] = 2
    df["time"] = df["time"].clip(lower=1).astype(int)
    return df


def load_loans(vintages=(2006, 2017), freddie_dir=None, cache_dir=None) -> pd.DataFrame:
    """Build (or read from a local parquet cache outside the repository) several vintages."""
    freddie_dir = Path(freddie_dir or os.environ.get("FREDDIE_DIR", "~/Downloads/freddie")).expanduser()
    cache_dir = Path(cache_dir or freddie_dir / "processed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for v in vintages:
        f = cache_dir / f"loans_{v}.parquet"
        if not f.exists():
            build_loans(v, freddie_dir).to_parquet(f, index=False)
        out.append(pd.read_parquet(f))
    return pd.concat(out, ignore_index=True)


def monthly_new_dlq90(vintage: int, freddie_dir: str | Path | None = None) -> pd.DataFrame:
    """Aggregate only: number of loans whose FIRST 90+ delinquency happens in each calendar
    month, split by whether the loan was under payment relief (forbearance/disaster) that month."""
    freddie_dir = Path(freddie_dir or os.environ.get("FREDDIE_DIR", "~/Downloads/freddie")).expanduser()
    perf = _reader(_find(freddie_dir, f"sample_perf_{vintage}.txt"), PERF_COLS)
    sql = f"""
    WITH p AS (
        SELECT loan_id, CAST(period AS INTEGER) AS period,
               coalesce(assistance_plan, '') = 'F' OR coalesce(disaster, '') = 'Y' AS relief
        FROM {perf}
        WHERE dlq = 'RA' OR coalesce(TRY_CAST(dlq AS INTEGER), 0) >= 3
    ),
    firsts AS (SELECT loan_id, arg_min(period, period) AS period, arg_min(relief, period) AS relief
               FROM p GROUP BY loan_id)
    SELECT period, relief, count(*) AS new_dlq90 FROM firsts GROUP BY period, relief ORDER BY period
    """
    out = duckdb.sql(sql).df()
    out["month"] = pd.to_datetime(out["period"].astype(str), format="%Y%m")
    return out


def relief_dlq90_by_state(vintage: int, before: int, freddie_dir: str | Path | None = None) -> pd.DataFrame:
    """Aggregate only: loans whose first 90+ delinquency under payment relief happened before
    `before` (YYYYMM), counted by property state, next to each state's share of the portfolio."""
    freddie_dir = Path(freddie_dir or os.environ.get("FREDDIE_DIR", "~/Downloads/freddie")).expanduser()
    perf = _reader(_find(freddie_dir, f"sample_perf_{vintage}.txt"), PERF_COLS)
    orig = _reader(_find(freddie_dir, f"sample_orig_{vintage}.txt"), ORIG_COLS)
    sql = f"""
    WITH p AS (SELECT loan_id, CAST(period AS INTEGER) AS period FROM {perf}
               WHERE (dlq = 'RA' OR coalesce(TRY_CAST(dlq AS INTEGER), 0) >= 3)
                 AND (coalesce(assistance_plan, '') = 'F' OR coalesce(disaster, '') = 'Y')),
         f AS (SELECT loan_id, min(period) AS period FROM p GROUP BY loan_id),
         portfolio AS (SELECT state, count(*) AS loans FROM {orig} GROUP BY state),
         relief AS (SELECT o.state, count(*) AS relief_dlq90 FROM f JOIN {orig} o USING (loan_id)
                    WHERE f.period < {before} GROUP BY o.state)
    SELECT portfolio.state, coalesce(relief_dlq90, 0) AS relief_dlq90, loans
    FROM portfolio LEFT JOIN relief USING (state)
    """
    out = duckdb.sql(sql).df()
    out["share_of_relief_dlq90"] = out["relief_dlq90"] / out["relief_dlq90"].sum()
    out["share_of_portfolio"] = out["loans"] / out["loans"].sum()
    return out.sort_values("relief_dlq90", ascending=False).reset_index(drop=True)
