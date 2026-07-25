"""Unit tests for bbg_fetch.option_chain — no live Bloomberg connection required.

The pure functions (strike parsing/selection, leg split, put-call parity recovery,
CSV round-trip) are tested directly; fetch_option_chain / run are tested with bdp/bds
monkeypatched, so nothing here needs a Bloomberg Terminal.
"""

import numpy as np
import pandas as pd
import pytest

import bbg_fetch
from bbg_fetch import (fetch_option_chain, recover_option_forward, run,
                       OptionPriceSource, OptionChainResult)
from bbg_fetch.option_chain import (_parse_option_strikes, _select_strikes_near_atm,
                                    _select_strikes_on_grid, _split_option_legs,
                                    _OPTION_STRIKE_RE)

# parity ground truth for the synthetic chains: C(K) - P(K) = D (F - K)
_T = 48.0 / 365.0
_R, _F, _S = 0.0291, 1059.20, 1055.58
_D = np.exp(-_R * _T)
_LISTED = np.arange(335.0, 1582.5 + 1e-9, 2.5)                 # KOSPI2-like ladder, 2.5-wide
_FAKE_EXPIRY_DT = pd.Timestamp('2099-12-31')                   # far future: run()'s year_fraction stays > 0


def _fake_bds(underlying, fld, **overrides):
    side = overrides['CHAIN_PUT_CALL_TYPE_OVRD']
    names = [f"KOSPI2 09/10/26 {side}{k:g}" for k in _LISTED]
    return pd.DataFrame({'security_description': names})


def _fake_bdp(tickers=None, flds=None, **kwargs):
    tickers = list(tickers)
    if tickers == ['KOSPI2 Index'] and flds == 'px_last':     # the internal _fetch_spot call
        return pd.DataFrame({'px_last': [_S]}, index=tickers)
    rows = []
    for ticker in tickers:
        strike = float(_OPTION_STRIKE_RE.search(ticker).group(1))
        is_call = ' C' in ticker
        price = (50.0 + _D * (_F - strike)) if is_call else 50.0
        rows.append(dict(opt_put_call='Call' if is_call else 'Put',
                         opt_strike_px=strike, opt_undl_px=_S, opt_expire_dt=_FAKE_EXPIRY_DT,
                         px_bid=price - 0.4, px_ask=price + 0.4, px_last=price))
    return pd.DataFrame(rows, index=tickers).reindex(columns=list(flds))


@pytest.fixture
def stub_bbg(monkeypatch):
    """replace bdp/bds inside option_chain with the synthetic Bloomberg above."""
    monkeypatch.setattr('bbg_fetch.option_chain.bds', _fake_bds)
    monkeypatch.setattr('bbg_fetch.option_chain.bdp', _fake_bdp)


def _synthetic_chain() -> pd.DataFrame:
    """parity-consistent chain, calls and puts at 41 strikes around spot."""
    rows, idx = [], []
    for k in np.arange(1005.0, 1105.0 + 1e-9, 2.5):
        call = 50.0 + _D * (_F - k)
        rows.append(dict(opt_put_call='Call', opt_strike_px=k, opt_undl_px=_S,
                         opt_expire_dt=pd.Timestamp('2026-09-10'),
                         px_bid=call - 0.4, px_ask=call + 0.4, px_last=call))
        idx.append(f"KOSPI2 09/10/26 C{k:g} Index")
        rows.append(dict(opt_put_call='Put', opt_strike_px=k, opt_undl_px=_S,
                         opt_expire_dt=pd.Timestamp('2026-09-10'),
                         px_bid=50.0 - 0.4, px_ask=50.0 + 0.4, px_last=50.0))
        idx.append(f"KOSPI2 09/10/26 P{k:g} Index")
    return pd.DataFrame(rows, index=idx)


def _strikes_of(tickers) -> list:
    return sorted({float(_OPTION_STRIKE_RE.search(t).group(1)) for t in tickers})


# ---- pure selection helpers -----------------------------------------------------------

def test_parse_option_strikes():
    parsed = _parse_option_strikes(["KOSPI2 09/10/26 C337.5 Index",
                                    "KOSPI2 09/10/26 P1055 Index"])
    assert parsed == [("KOSPI2 09/10/26 C337.5 Index", 337.5),
                      ("KOSPI2 09/10/26 P1055 Index", 1055.0)]
    with pytest.raises(ValueError):
        _parse_option_strikes(["KOSPI2 09/10/26 Index"])          # no strike token


def test_select_strikes_near_atm():
    tickers = [f"KOSPI2 09/10/26 C{k:g} Index" for k in _LISTED]
    strikes = _strikes_of(_select_strikes_near_atm(tickers, spot=_S, num_strikes_per_side=20))
    assert len(strikes) == 41                                    # 2 * 20 + 1
    assert min(strikes) == 1005.0 and max(strikes) == 1105.0     # 1055 ATM +/- 20 * 2.5
    low = _select_strikes_near_atm(tickers, spot=100.0, num_strikes_per_side=5)
    assert len(_strikes_of(low)) == 6                            # clamps at bottom of ladder
    with pytest.raises(ValueError):
        _select_strikes_near_atm(tickers, spot=_S, num_strikes_per_side=0)


def test_select_strikes_on_grid():
    tickers = ([f"KOSPI2 09/10/26 C{k:g} Index" for k in _LISTED]
               + [f"KOSPI2 09/10/26 P{k:g} Index" for k in _LISTED])
    grid = np.linspace(700, 1400, 15)                            # 50-wide, all listed
    sel = _select_strikes_on_grid(tickers, grid)
    assert _strikes_of(sel) == sorted(grid.tolist())
    assert len(sel) == 30                                        # 15 strikes x 2 legs
    snapped = _select_strikes_on_grid([f"KOSPI2 09/10/26 C{k:g} Index" for k in _LISTED],
                                      [701.3, 9999.0])
    assert _strikes_of(snapped) == [702.5, 1582.5]              # nearest-listed, top clamps
    with pytest.raises(ValueError):
        _select_strikes_on_grid(tickers, [])


# ---- parity recovery ------------------------------------------------------------------

def test_split_option_legs():
    chain = _synthetic_chain()
    calls, puts = _split_option_legs(chain, OptionPriceSource.LAST)
    assert len(calls) == 41 and len(puts) == 41
    assert calls.index.is_monotonic_increasing
    with pytest.raises(ValueError):
        _split_option_legs(chain.drop(columns=['opt_strike_px']), OptionPriceSource.LAST)


def test_recover_option_forward():
    chain = _synthetic_chain()
    for price_source in OptionPriceSource:
        out = recover_option_forward(chain, spot=_S, year_fraction=_T, price_source=price_source)
        assert set(out) == {'forward', 'rate', 'r2', 'num_strikes_used'}
        assert abs(out['forward'] - _F) < 0.05
        assert abs(out['rate'] - _R) < 1e-6                      # clean data -> exact
    with pytest.raises(ValueError):
        recover_option_forward(chain, spot=_S, year_fraction=0.0)
    with pytest.raises(ValueError):
        recover_option_forward(chain.iloc[:4], spot=_S, year_fraction=_T)   # < 3 both-leg strikes


# ---- fetch_option_chain / run (stubbed Bloomberg) -------------------------------------

def test_fetch_option_chain_grid(stub_bbg):
    df = fetch_option_chain(underlying='KOSPI2 Index', expiry='20260910',
                            strike_grid=np.linspace(700, 1400, 15))
    assert len(df.index) == 30
    both = fetch_option_chain(underlying='KOSPI2 Index', expiry='20260910',   # grid beats window
                              num_strikes_per_side=20, strike_grid=np.linspace(700, 1400, 15))
    assert len(both.index) == 30


def test_fetch_option_chain_window(stub_bbg):
    df = fetch_option_chain(underlying='KOSPI2 Index', expiry='20260910', num_strikes_per_side=20)
    assert len(df.index) == 82                                   # 41 strikes x 2 legs


@pytest.mark.parametrize('bad', ['20260013', '20261301', '2026-09-10', 'abc'])
def test_fetch_option_chain_rejects_bad_expiry(bad):
    with pytest.raises(ValueError, match='YYYYMMDD'):            # raised before any Bloomberg call
        fetch_option_chain(expiry=bad)


def test_run(stub_bbg):
    result = run(underlying='KOSPI2 Index', expiry='20260910', strike_grid=np.linspace(700, 1400, 15))
    assert isinstance(result, OptionChainResult)
    assert result.spot == _S                                     # inferred from opt_undl_px
    assert result.year_fraction > 0.0                            # inferred from opt_expire_dt
    assert abs(result.forward - _F) < 0.1                        # forward is year-fraction independent
    assert len(result.chain.index) == 30


# ---- OptionChainResult ----------------------------------------------------------------

def test_option_chain_result_fields():
    import dataclasses
    names = [f.name for f in dataclasses.fields(OptionChainResult)]
    assert names == ['chain', 'spot', 'year_fraction', 'forward', 'rate', 'r2', 'num_strikes_used']


def test_option_chain_result_csv_roundtrip(tmp_path):
    chain = _synthetic_chain()
    chain.loc[chain.index[0], 'px_bid'] = np.nan                 # a NaN to preserve
    result = OptionChainResult(chain=chain, spot=_S, year_fraction=_T, forward=_F,
                               rate=_R, r2=0.9994, num_strikes_used=20)
    path = str(tmp_path / 'snap.csv')
    result.to_csv(path)
    back = OptionChainResult.read_csv(path)
    assert back.spot == _S and back.forward == _F and back.num_strikes_used == 20
    assert list(back.chain.index) == list(chain.index)
    assert bool(pd.isna(back.chain.loc[chain.index[0], 'px_bid']))
    assert pd.api.types.is_datetime64_any_dtype(back.chain['opt_expire_dt'])   # re-parsed from text


def test_option_chain_top_level_exports():
    for name in ('fetch_option_chain', 'recover_option_forward', 'run',
                 'OptionPriceSource', 'OptionChainResult', 'OPTION_CHAIN_FIELDS'):
        assert hasattr(bbg_fetch, name), f"missing public export: {name}"
