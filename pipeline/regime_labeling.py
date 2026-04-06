"""Label data with regime based on structural breakpoints."""

import pandas as pd

# Regime breakpoints from the paper
REGIME_BREAKS = [
    pd.Timestamp("2020-11-01"),
    pd.Timestamp("2024-01-11"),  # ETF approval date
]

REGIME_NAMES = ["early", "late", "post_etf"]


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'regime' column based on breakpoint dates.

    - early:    2016-01 to 2020-10-31 (on-chain + macro mix)
    - late:     2020-11-01 to 2024-01-10 (pure macro driven)
    - post_etf: 2024-01-11 onward (post-ETF stability)
    """
    out = df.copy()
    conditions = [
        out.index < REGIME_BREAKS[0],
        (out.index >= REGIME_BREAKS[0]) & (out.index < REGIME_BREAKS[1]),
        out.index >= REGIME_BREAKS[1],
    ]
    out["regime"] = pd.Series(
        pd.Categorical(
            values=sum(([name] * 1 for name in REGIME_NAMES), []),  # placeholder
            categories=REGIME_NAMES,
        )
    ).iloc[0]  # default

    # Use numpy select for proper assignment
    import numpy as np
    out["regime"] = np.select(conditions, REGIME_NAMES, default="unknown")
    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_NAMES, ordered=True)

    return out


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary stats per regime."""
    if "regime" not in df.columns:
        df = label_regimes(df)

    summary = df.groupby("regime", observed=True).agg(
        start_date=pd.NamedAgg(column="regime", aggfunc=lambda x: x.index.min()),
        end_date=pd.NamedAgg(column="regime", aggfunc=lambda x: x.index.max()),
        n_days=pd.NamedAgg(column="regime", aggfunc="count"),
    )
    return summary


if __name__ == "__main__":
    dates = pd.date_range("2016-01-01", "2025-03-01", freq="D")
    test_df = pd.DataFrame({"val": range(len(dates))}, index=dates)
    test_df.index.name = "date"
    result = label_regimes(test_df)
    print(regime_summary(result))
