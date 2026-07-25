"""Pure-function smoke tests — no live Bloomberg connection required.

These run in CI to catch regressions in the parts of bbg_fetch that don't
depend on a Bloomberg Terminal: ticker parsing, field-name normalization,
input coercion, and the public import surface.
"""

import bbg_fetch
from bbg_fetch import contract_to_instrument, instrument_to_active_ticker
from bbg_fetch._blp_api import _as_list, _normalize_name


def test_version():
    assert bbg_fetch.__version__ == "2.3.0"


def test_contract_to_instrument():
    assert contract_to_instrument("ES1 Index") == "ES"
    assert contract_to_instrument("TY1 Comdty") == "TY"
    assert contract_to_instrument("CL10 Comdty") == "CL"


def test_instrument_to_active_ticker():
    assert instrument_to_active_ticker("ES1 Index", num=1) == "ES1 Index"
    assert instrument_to_active_ticker("ES1 Index", num=3) == "ES3 Index"
    assert instrument_to_active_ticker("TY1 Comdty", num=2) == "TY2 Comdty"


def test_normalize_name():
    assert _normalize_name("PX_LAST") == "px_last"
    assert _normalize_name("Last Price") == "last_price"
    assert _normalize_name("BS-LT-BORROW") == "bs_lt_borrow"


def test_as_list():
    # strings stay atomic — not split into characters
    assert _as_list("AAPL US Equity") == ["AAPL US Equity"]
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list(("a", "b")) == ["a", "b"]
    # non-iterable scalar wrapped in a single-element list
    assert _as_list(42) == [42]


def test_top_level_exports():
    """Things the README promises are importable from bbg_fetch."""
    expected = [
        "fetch_field_timeseries_per_tickers",
        "fetch_vol_timeseries",
        "FX_DICT",
        "IMPVOL_FIELDS_DELTA",
        "DEFAULT_START_DATE",
        "bdp", "bdh", "bds", "disconnect",
    ]
    for name in expected:
        assert hasattr(bbg_fetch, name), f"missing public export: {name}"


def test_fetch_vol_surface(monkeypatch):
    """the surface reshape: as-of row -> tenor rows x moneyness columns (bdh stubbed)."""
    import numpy as np
    import pandas as pd
    import bbg_fetch.core as core

    labels = ["30d90.0", "30d100.0", "60d90.0", "60d100.0"]
    dates = pd.date_range("2026-07-20", periods=3, freq="D")
    timeseries = pd.DataFrame(np.arange(10, 130, 10, dtype=float).reshape(3, 4),
                              index=dates, columns=labels)  # rows 10.., 50.., 90..

    def fake_fetch_vol_timeseries(ticker, vol_fields, start_date, add_underlying, scaler):
        out = timeseries.copy()
        return out if scaler is None else out * scaler

    monkeypatch.setattr(core, "fetch_vol_timeseries", fake_fetch_vol_timeseries)
    vol_fields = ({"F30_90": "30d90.0", "F30_100": "30d100.0"},
                  {"F60_90": "60d90.0", "F60_100": "60d100.0"})

    surface = bbg_fetch.fetch_vol_surface(ticker="KOSPI2 Index",
                                          value_date=pd.Timestamp("2026-07-22"),
                                          vol_fields=vol_fields, scaler=None)
    assert list(surface.index) == ["30d", "60d"]          # tenor rows, vol_fields order
    assert list(surface.columns) == [90.0, 100.0]         # moneyness, ascending
    assert surface.loc["30d", 90.0] == 90.0               # last row (2026-07-22)
    assert surface.loc["60d", 100.0] == 120.0

    # value_date picks the last row on or before it
    earlier = bbg_fetch.fetch_vol_surface(ticker="KOSPI2 Index",
                                          value_date=pd.Timestamp("2026-07-21"),
                                          vol_fields=vol_fields, scaler=None)
    assert earlier.loc["30d", 90.0] == 50.0               # the 2026-07-21 row
