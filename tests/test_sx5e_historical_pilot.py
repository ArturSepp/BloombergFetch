"""Terminal-free coverage and selection tests for the historical acquisition example."""

from io import StringIO

import pandas as pd
import pytest

from examples.fetch_sx5e_historical_pilot import nearest_expiry, read_listing, short_curve_tickers


def test_empty_cached_listing_is_a_valid_missing_observation():
    empty = StringIO(pd.DataFrame().to_csv(index=False))
    assert read_listing(empty).empty
    saved = StringIO(pd.DataFrame({"security": ["archived-contract"]}).to_csv(index=False))
    assert read_listing(saved).security.tolist() == ["archived-contract"]


def test_expiry_selection_excludes_expired_and_breaks_ties_earlier():
    expiries = pd.Series(["2026-01-04", "2026-01-02", "2024-12-01", None])
    assert nearest_expiry(expiries, "2025-01-03") == pd.Timestamp("2026-01-02")
    with pytest.raises(ValueError, match="positive expiry"):
        nearest_expiry(pd.Series([None, "2024-01-01"]), "2025-01-03")


def test_curve_metadata_never_requests_blank_or_long_tenor_tickers():
    frame = pd.DataFrame({"tenor": ["1D", "1M", "6M", "9M", "12M", "30Y"],
                          "tenor_ticker": [None, "", " ", "EUR OIS", "EUR OIS", "long"]})
    assert short_curve_tickers(frame) == ["EUR OIS"]
