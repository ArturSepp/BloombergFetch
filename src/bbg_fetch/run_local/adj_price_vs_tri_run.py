"""Compare adjusted Bloomberg prices with total-return indices interactively."""

from enum import Enum

import pandas as pd

from bbg_fetch import fetch_field_timeseries_per_tickers

TICKERS = ("OCSL US Equity",)
START = "2021-01-01"
END = "2026-04-14"
MAX_DAILY_RETURN_DIFF = 1e-3
MAX_ANN_RETURN_DIFF_BPS = 5.0
MIN_RETURN_CORRELATION = 0.9999


class Locals(Enum):
    """Available adjusted-price development diagnostics."""

    ADJ_PRICE_VS_TRI = 1


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to 100 at its first valid observation."""
    return 100.0 * series / series.dropna().iloc[0]


def _fetch_trio(ticker: str) -> pd.DataFrame:
    """Fetch adjusted price, raw price, and total-return index for one ticker."""
    adjusted = fetch_field_timeseries_per_tickers(
        tickers=[ticker],
        field="PX_LAST",
        CshAdjNormal=True,
        CshAdjAbnormal=True,
        start_date=START,
        end_date=END,
    )
    raw = fetch_field_timeseries_per_tickers(
        tickers=[ticker],
        field="PX_LAST",
        CshAdjNormal=False,
        CshAdjAbnormal=False,
        start_date=START,
        end_date=END,
    )
    total_return = fetch_field_timeseries_per_tickers(
        tickers=[ticker],
        field="TOT_RETURN_INDEX_GROSS_DVDS",
        start_date=START,
        end_date=END,
    )
    return pd.DataFrame(
        {
            "px_adj": _normalize(adjusted[ticker]),
            "px_raw": _normalize(raw[ticker]),
            "tri": _normalize(total_return[ticker]),
        }
    ).dropna()


def run_local(local: Locals) -> None:
    """Print the selected live adjusted-price diagnostic."""
    if local != Locals.ADJ_PRICE_VS_TRI:
        raise NotImplementedError(f"unsupported local: {local}")

    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"TESTING: {ticker}")
        print(f"{'=' * 60}")

        frame = _fetch_trio(ticker)
        returns = frame[["px_adj", "tri"]].pct_change().dropna()
        years = (frame.index[-1] - frame.index[0]).days / 365.25
        annualized_adjusted = (frame["px_adj"].iloc[-1] / 100.0) ** (1.0 / years) - 1.0
        annualized_total_return = (frame["tri"].iloc[-1] / 100.0) ** (1.0 / years) - 1.0

        print(f"\nDate range:  {frame.index[0].date()} to {frame.index[-1].date()} ({years:.1f}y)")
        print(f"Observations: {len(frame)}")
        print("\nCumulative returns:")
        print(f"  PX_LAST (adj):  {frame['px_adj'].iloc[-1] - 100:+.2f}%")
        print(f"  PX_LAST (raw):  {frame['px_raw'].iloc[-1] - 100:+.2f}%")
        print(f"  TRI:            {frame['tri'].iloc[-1] - 100:+.2f}%")
        print("\nAnnualised returns:")
        print(f"  PX_LAST (adj):  {annualized_adjusted * 100:+.2f}%")
        print(f"  TRI:            {annualized_total_return * 100:+.2f}%")
        print(f"  Diff:           {abs(annualized_adjusted - annualized_total_return) * 1e4:.2f} bps")

        difference = (returns["px_adj"] - returns["tri"]).abs()
        correlation = returns["px_adj"].corr(returns["tri"])
        print("\nDaily return comparison (adj vs TRI):")
        print(f"  Correlation:    {correlation:.8f}")
        print(f"  Mean |diff|:    {difference.mean():.8f}")
        print(f"  Max  |diff|:    {difference.max():.8f}")
        print(f"  Std  diff:      {returns['px_adj'].sub(returns['tri']).std():.8f}")

        checks = (
            ("Correlation >= 0.9999", correlation >= MIN_RETURN_CORRELATION),
            ("Max daily diff < 10 bps", difference.max() < MAX_DAILY_RETURN_DIFF),
            (
                "Ann. return diff < 5 bps",
                abs(annualized_adjusted - annualized_total_return) * 1e4
                < MAX_ANN_RETURN_DIFF_BPS,
            ),
        )
        print("\nChecks:")
        for name, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'}: {name}")


if __name__ == "__main__":
    run_local(local=Locals.ADJ_PRICE_VS_TRI)
