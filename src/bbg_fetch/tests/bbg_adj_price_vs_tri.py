"""
Test: cash-adjusted PX_LAST vs TOT_RETURN_INDEX_GROSS_DVDS for dividend-heavy equities.

Validates that fetch_field_timeseries_per_tickers with CshAdjNormal=True, CshAdjAbnormal=True
produces return series economically equivalent to the Bloomberg total return index.

Expected behavior:
  - Daily return correlation > 0.9999
  - Max absolute daily return difference < 1e-3 (10 bps, ex-date rounding)
  - Annualised return difference < 5 bps
  - Raw (unadjusted) price returns diverge significantly from TRI (confirms dividends matter)
"""

import numpy as np
import pandas as pd
import pytest

from bbg_fetch import fetch_field_timeseries_per_tickers

# high-dividend tickers for robust testing
TICKERS = ['OCSL US Equity']
START = '2021-01-01'
END = '2026-04-14'

# tolerances
MAX_DAILY_RETURN_DIFF = 1e-3       # 10 bps — accounts for ex-date rounding
MAX_ANN_RETURN_DIFF_BPS = 5.0      # 5 bps annualised
MIN_RETURN_CORRELATION = 0.9999
MIN_RAW_VS_ADJ_DIVERGENCE = 0.10   # raw price must diverge by at least 10% cumulative


def _normalize(s: pd.Series) -> pd.Series:
    """Normalize series to 100 at first valid observation."""
    return 100.0 * s / s.dropna().iloc[0]


def _fetch_trio(ticker: str) -> pd.DataFrame:
    """Fetch adjusted price, raw price, and TRI for a single ticker."""
    px_adj = fetch_field_timeseries_per_tickers(
        tickers=[ticker],
        field='PX_LAST',
        CshAdjNormal=True,
        CshAdjAbnormal=True,
        start_date=START,
        end_date=END,
    )
    px_raw = fetch_field_timeseries_per_tickers(
        tickers=[ticker],
        field='PX_LAST',
        CshAdjNormal=False,
        CshAdjAbnormal=False,
        start_date=START,
        end_date=END,
    )
    tri = fetch_field_timeseries_per_tickers(
        tickers=[ticker],
        field='TOT_RETURN_INDEX_GROSS_DVDS',
        start_date=START,
        end_date=END,
    )
    df = pd.DataFrame({
        'px_adj': _normalize(px_adj[ticker]),
        'px_raw': _normalize(px_raw[ticker]),
        'tri': _normalize(tri[ticker]),
    }).dropna()
    return df


@pytest.fixture(scope='module', params=TICKERS)
def price_data(request) -> tuple[str, pd.DataFrame]:
    """Fetch and cache price data per ticker for the test module."""
    ticker = request.param
    return ticker, _fetch_trio(ticker)


def test_adj_vs_tri_daily_return_correlation(price_data):
    """Cash-adjusted returns must correlate > 0.9999 with TRI returns."""
    ticker, df = price_data
    rets = df[['px_adj', 'tri']].pct_change().dropna()
    corr = rets['px_adj'].corr(rets['tri'])
    assert corr >= MIN_RETURN_CORRELATION, (
        f"{ticker}: adj vs TRI correlation {corr:.6f} < {MIN_RETURN_CORRELATION}"
    )


def test_adj_vs_tri_max_daily_diff(price_data):
    """No single-day return difference should exceed 10 bps."""
    ticker, df = price_data
    rets = df[['px_adj', 'tri']].pct_change().dropna()
    diff = (rets['px_adj'] - rets['tri']).abs()
    max_diff = diff.max()
    assert max_diff < MAX_DAILY_RETURN_DIFF, (
        f"{ticker}: max daily return diff {max_diff:.6f} >= {MAX_DAILY_RETURN_DIFF}"
    )


def test_adj_vs_tri_annualised_return_diff(price_data):
    """Annualised return difference must be < 5 bps."""
    ticker, df = price_data
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    ann_adj = (df['px_adj'].iloc[-1] / 100.0) ** (1.0 / n_years) - 1.0
    ann_tri = (df['tri'].iloc[-1] / 100.0) ** (1.0 / n_years) - 1.0
    diff_bps = abs(ann_adj - ann_tri) * 1e4
    assert diff_bps < MAX_ANN_RETURN_DIFF_BPS, (
        f"{ticker}: annualised return diff {diff_bps:.2f} bps >= {MAX_ANN_RETURN_DIFF_BPS}"
    )


def test_raw_price_diverges_from_tri(price_data):
    """
    Sanity check: raw (unadjusted) price must diverge materially from TRI.
    If it doesn't, the ticker doesn't pay meaningful dividends and the test
    is not exercising the cash-adjustment logic.
    """
    ticker, df = price_data
    cum_raw = df['px_raw'].iloc[-1] - 100.0
    cum_tri = df['tri'].iloc[-1] - 100.0
    divergence = abs(cum_tri - cum_raw)
    assert divergence > MIN_RAW_VS_ADJ_DIVERGENCE * 100, (
        f"{ticker}: raw vs TRI divergence {divergence:.1f}% is too small — "
        f"ticker may not pay meaningful dividends for this test"
    )


def test_adj_captures_dividends(price_data):
    """Cash-adjusted cumulative return must be close to TRI, not to raw price."""
    ticker, df = price_data
    cum_adj = df['px_adj'].iloc[-1] - 100.0
    cum_raw = df['px_raw'].iloc[-1] - 100.0
    cum_tri = df['tri'].iloc[-1] - 100.0

    # adj should be much closer to TRI than to raw
    dist_to_tri = abs(cum_adj - cum_tri)
    dist_to_raw = abs(cum_adj - cum_raw)
    assert dist_to_tri < dist_to_raw, (
        f"{ticker}: adj is closer to raw ({dist_to_raw:.2f}%) than to TRI ({dist_to_tri:.2f}%)"
    )


if __name__ == '__main__':
    # standalone run with verbose output for manual inspection
    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"TESTING: {ticker}")
        print(f"{'=' * 60}")

        df = _fetch_trio(ticker)
        rets = df[['px_adj', 'tri']].pct_change().dropna()

        n_years = (df.index[-1] - df.index[0]).days / 365.25
        ann_adj = (df['px_adj'].iloc[-1] / 100.0) ** (1.0 / n_years) - 1.0
        ann_tri = (df['tri'].iloc[-1] / 100.0) ** (1.0 / n_years) - 1.0

        print(f"\nDate range:  {df.index[0].date()} to {df.index[-1].date()} ({n_years:.1f}y)")
        print(f"Observations: {len(df)}")

        print(f"\nCumulative returns:")
        print(f"  PX_LAST (adj):  {df['px_adj'].iloc[-1] - 100:+.2f}%")
        print(f"  PX_LAST (raw):  {df['px_raw'].iloc[-1] - 100:+.2f}%")
        print(f"  TRI:            {df['tri'].iloc[-1] - 100:+.2f}%")

        print(f"\nAnnualised returns:")
        print(f"  PX_LAST (adj):  {ann_adj * 100:+.2f}%")
        print(f"  TRI:            {ann_tri * 100:+.2f}%")
        print(f"  Diff:           {abs(ann_adj - ann_tri) * 1e4:.2f} bps")

        diff = (rets['px_adj'] - rets['tri']).abs()
        corr = rets['px_adj'].corr(rets['tri'])

        print(f"\nDaily return comparison (adj vs TRI):")
        print(f"  Correlation:    {corr:.8f}")
        print(f"  Mean |diff|:    {diff.mean():.8f}")
        print(f"  Max  |diff|:    {diff.max():.8f}")
        print(f"  Std  diff:      {rets['px_adj'].sub(rets['tri']).std():.8f}")

        # verdict
        checks = [
            ('Correlation >= 0.9999', corr >= MIN_RETURN_CORRELATION),
            ('Max daily diff < 10 bps', diff.max() < MAX_DAILY_RETURN_DIFF),
            ('Ann. return diff < 5 bps', abs(ann_adj - ann_tri) * 1e4 < MAX_ANN_RETURN_DIFF_BPS),
        ]
        print(f"\nChecks:")
        for name, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'}: {name}")
