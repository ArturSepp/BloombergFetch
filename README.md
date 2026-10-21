# bbg-fetch

**Bloomberg Desktop API request/response data in pandas DataFrames for quantitative research.**

It wraps BDP-, BDH-, and BDS-style requests and selected research workflows. Live requests require
a running Bloomberg Terminal, suitable entitlements, and Bloomberg's separately installed
`blpapi`; streaming and intraday subscriptions are out of scope.

**Install:** `pip install bbg-fetch` · **Import:** `bbg_fetch` · **Status:** Beta

[![PyPI](https://img.shields.io/pypi/v/bbg-fetch?style=flat-square)](https://pypi.org/project/bbg-fetch/)
[![Python](https://img.shields.io/pypi/pyversions/bbg-fetch?style=flat-square)](https://pypi.org/project/bbg-fetch/)
[![CI](https://github.com/ArturSepp/BloombergFetch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/BloombergFetch/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/bloombergfetch/badge/?version=latest)](https://bloombergfetch.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/BloombergFetch.svg?style=flat-square)](LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/bbg-fetch)](https://pepy.tech/project/bbg-fetch)
[![Monthly](https://static.pepy.tech/badge/bbg-fetch/month)](https://pepy.tech/project/bbg-fetch)

**Documentation:** [bloombergfetch.readthedocs.io](https://bloombergfetch.readthedocs.io/en/latest/) ·
[first success without a Terminal](https://bloombergfetch.readthedocs.io/en/latest/first_success.html) ·
[examples index](examples/README.md)

---

## Why bbg-fetch

Direct `blpapi` use requires session setup, request construction, event handling, and response
parsing. `bbg-fetch` centralises that request/response plumbing for repeated research workflows:
one high-level call replaces the session options, service and request objects, the field and
security appends, the adjustment flags and the event loop that a raw `blpapi` historical
request needs.

```python
import pandas as pd

from bbg_fetch import fetch_field_timeseries_per_tickers

prices = fetch_field_timeseries_per_tickers(
    tickers=['AAPL US Equity'],
    field='PX_LAST',
    start_date=pd.Timestamp('2020-01-01')
)
```

The wrapper handles the session/request/response path and returns the result as a DataFrame
with documented column naming, corporate-action flags, and index handling. The direct `blpapi`
session implementation is isolated in the private `_blp_api.py` module.

### Key differentiators

- **Multi-asset coverage.** Equities (historical prices with split/dividend adjustments,
  fundamentals, dividend history), futures (contract tables with carry analysis, active contract
  series, roll handling), options (implied volatility surfaces by moneyness and delta, option
  chains), fixed income (bond pricing and analytics by ISIN, yield curves, CDS spreads), FX
  (currency rates and volatility) and indices (constituent weights, ISIN-to-ticker resolution).
- **Dict-based ticker renaming.** `{'ES1 Index': 'SPX'}` → DataFrame columns named `SPX`, for
  tickers and for fields.
- **Automatic retry** on Bloomberg connection flakes.
- **Corporate action adjustments on by default** (normal, abnormal, splits).
- **Predefined field mappings** for vol surfaces (30d/60d/3m/6m/12m moneyness, delta) and the 19
  major FX pairs.
- **Carry computation built into futures contract tables.**

### When to use it — and when not

`bbg-fetch` targets research workflows: request/response pulls of reference data, price
histories, volatility surfaces, and index constituents into analysis-ready DataFrames from a
machine running the Bloomberg Terminal.

It is request/response by design: no streaming subscriptions and no intraday tick capture. Where
no terminal is available, the examples in [`qis`](https://github.com/ArturSepp/QuantInvestStrats)
run on free Yahoo data instead.

## Installation

### 1. Install blpapi

```bash
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

Bloomberg's current API Library provides bundled Python wheels for community-supported Python
versions. The live Desktop API workflow documented here targets Windows; Bloomberg's Linux and
macOS API distributions serve different Bloomberg server products rather than a local Professional
Terminal. Check the [official API Library](https://professional.bloomberg.com/support/api-library/)
for the current platform and release matrix.

**Corporate proxy?** Download the `.whl` from `https://blpapi.bloomberg.com/repository/releases/python/simple/blpapi/` via browser, then:

```bash
python -m pip install /path/to/blpapi-wheel.whl
```

### 2. Install bbg-fetch

```bash
pip install bbg-fetch
```

Or from source:

```bash
git clone https://github.com/ArturSepp/BloombergFetch.git
pip install .
```

**Requirements:** Python 3.10+, with the Bloomberg Desktop API and an entitled Bloomberg Terminal
session available on the same Windows machine for live requests. Installation, session,
entitlement, field and empty-response failures are diagnosed in the
[troubleshooting guide](https://bloombergfetch.readthedocs.io/en/latest/troubleshooting.html).

## Table of contents

1. [Why bbg-fetch](#why-bbg-fetch)
2. [Installation](#installation)
3. [Statement of need](#statement-of-need)
4. [Overview](#overview)
5. [Five-minute quickstart](#five-minute-quickstart)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Changelog](#changelog)
9. [Feedback & contributing](#feedback-contributing)
10. [Citation](#citation)
11. [Ecosystem](#ecosystem)
12. [License](#license)
13. [Disclaimer](#disclaimer)

## Statement of need

Research that runs on Bloomberg data repeats the same request shapes — a field across tickers, a
ticker across fields, a volatility surface, an option chain, a contract table, bond analytics by
ISIN, index members — and each one needs the same session, request and response handling before
a DataFrame exists. `bbg-fetch` is the data-access layer of the stack listed under
[Ecosystem](#ecosystem): it turns those request shapes into pandas objects with stated column
names, adjustment flags and index conventions, so that the market-data containers in `qis` and
the optional Bloomberg provider of `option-chain-analytics` consume one documented contract
rather than their own `blpapi` sessions. No research paper is implemented in this package.

## Overview

The installable package is `src/bbg_fetch/`: `__init__.py` exports the public API, `core.py`
holds the high-level fetch functions, `option_chain.py` holds option-chain fetching and parity
recovery, and the private `_blp_api.py` holds the direct `blpapi` shim (`bdp`, `bdh`, `bds`).
Source-only development runners live under `src/bbg_fetch/run_local/` and the live
adjusted-price pytest under `src/bbg_fetch/tests/`; terminal-free CI tests live in `tests/` and
the authoritative runnable scripts in `examples/`. The
[API reference](https://bloombergfetch.readthedocs.io/en/latest/api.html) documents every
public function and the exact pandas shapes it returns.

### Analytics at a glance

| Area | Function | Returns |
| --- | --- | --- |
| Price data | `fetch_field_timeseries_per_tickers()` | One field across multiple tickers (with optional dict-based renaming) |
| | `fetch_fields_timeseries_per_ticker()` | Multiple fields for a single ticker |
| | `fetch_last_prices()` | Snapshot of current prices |
| Fundamentals | `fetch_fundamentals()` | Company metadata and fundamentals (dict renaming for tickers and fields) |
| | `fetch_balance_data()` | Balance sheet ratios and credit metrics |
| | `fetch_dividend_history()` | Full dividend history (dates, amounts, types) |
| | `fetch_div_yields()` | Per-ticker dividend amounts and trailing 1-year yield |
| Derivatives | `fetch_vol_timeseries()` | Implied vol time series with underlying + rates (supports list-of-dicts for multi-tenor) |
| | `fetch_vol_surface()` | Implied vol surface for one date: tenor rows × moneyness columns |
| | `fetch_option_chain()` | Listed option chain for one expiry, trimmed to a strike window or explicit grid |
| | `recover_option_forward()` | Implied forward (and rate) from put-call parity |
| | `run()` | Fetch a chain and recover the forward/rate in one call → `OptionChainResult` |
| | `fetch_futures_contract_table()` | Contract specs, carry, timestamps |
| | `fetch_active_futures()` | Front + second month price series with retry logic |
| Fixed income | `fetch_bonds_info()` | Bond analytics by ISIN (with optional date override) |
| | `fetch_cds_info()` | CDS spread tickers from equity tickers |
| | `fetch_issuer_isins_from_bond_isins()` | Bond ISIN → issuer equity ISIN mapping |
| Index and resolution | `fetch_index_members_weights()` | Constituents and weights (configurable BDS field) |
| | `fetch_tickers_from_isins()` | ISIN → Bloomberg composite ticker |
| Futures utilities | `instrument_to_active_ticker()` | `'ES1 Index'` + `num=3` → `'ES3 Index'` |
| | `contract_to_instrument()` | `'ES1 Index'` → `'ES'` (strip generic number) |
| Low-level wrappers | `bdp()`, `bdh()`, `bds()` | Bloomberg Data Point, Data History and Data Set requests |
| | `disconnect()` | Explicitly stop the shared blpapi session (also runs at interpreter exit via `atexit`) |

Configuration constants: `DEFAULT_START_DATE` (`01Jan1959`, historical data) and
`VOLS_START_DATE` (`03Jan2005`, volatility data); `FX_DICT` with 19 major pairs; the
`IMPVOL_FIELDS_*` moneyness and delta field mappings; and Bloomberg's corporate-action
adjustment flags `CshAdjNormal`, `CshAdjAbnormal` and `CapChg`, all `True` by default on the
price functions. The [API reference](https://bloombergfetch.readthedocs.io/en/latest/api.html)
documents each constant and mapping.

## Five-minute quickstart

After installing `blpapi` and `bbg-fetch`, run the deterministic example from a source checkout:

```console
python examples/quickstart_no_terminal.py
```

It checks the public API on a synthetic option chain and prints
`Bloomberg connection: NOT TESTED`. It makes no live request and writes no data.
See [first success without a Terminal](https://bloombergfetch.readthedocs.io/en/latest/first_success.html)
for the complete script and expected evidence. On an entitled Bloomberg machine, the first live
request is one call:

```python
import pandas as pd

from bbg_fetch import fetch_field_timeseries_per_tickers

prices = fetch_field_timeseries_per_tickers(
    tickers={'ES1 Index': 'SPX', 'TY1 Comdty': '10yUST'},
    field='PX_LAST',
    start_date=pd.Timestamp('2020-01-01')
)
# Returns a clean DataFrame with renamed columns, sorted index, split/div adjusted
```

## Examples

### Highlighted examples

Each block needs a running, entitled Bloomberg Terminal.

**Prices across tickers (with renaming)**

```python
import pandas as pd
from bbg_fetch import fetch_field_timeseries_per_tickers

# Pass a dict to auto-rename columns: Bloomberg ticker → your label
prices = fetch_field_timeseries_per_tickers(
    tickers={'ES1 Index': 'SPX', 'TY1 Comdty': '10yUST', 'GC1 Comdty': 'Gold'},
    field='PX_LAST',
    start_date=pd.Timestamp('2015-01-01')
)
# DataFrame with columns ['SPX', '10yUST', 'Gold'], DatetimeIndex, sorted, adjusted

# Or pass a list — column names stay as Bloomberg tickers
prices = fetch_field_timeseries_per_tickers(
    tickers=['AAPL US Equity', 'MSFT US Equity'],
    field='PX_LAST',
    start_date=pd.Timestamp('2020-01-01')
)
```

**Implied volatility surface**

```python
from bbg_fetch import (fetch_vol_timeseries, fetch_vol_surface, IMPVOL_FIELDS_DELTA,
                       IMPVOL_FIELDS_MNY_30DAY, IMPVOL_FIELDS_MNY_60DAY,
                       IMPVOL_FIELDS_MNY_3MTH, IMPVOL_FIELDS_MNY_6MTH,
                       IMPVOL_FIELDS_MNY_12M)

# Delta-based vol for FX (1M and 2M, 10Δ to 50Δ puts and calls)
fx_vol = fetch_vol_timeseries(
    ticker='EURUSD Curncy',
    vol_fields=IMPVOL_FIELDS_DELTA,
    start_date=pd.Timestamp('2023-01-01')
)
# Returns: spot_price, div_yield, rf_rate + all vol columns

# Full moneyness surface across 5 tenors (30d, 60d, 3m, 6m, 12m × 9 strikes)
eq_vol = fetch_vol_timeseries(
    ticker='SPX Index',
    vol_fields=[IMPVOL_FIELDS_MNY_30DAY, IMPVOL_FIELDS_MNY_60DAY,
                IMPVOL_FIELDS_MNY_3MTH, IMPVOL_FIELDS_MNY_6MTH,
                IMPVOL_FIELDS_MNY_12M],
    start_date=pd.Timestamp('2010-01-01')
)

# Surface snapshot for one date: rows = tenor, columns = moneyness (percent)
surface = fetch_vol_surface(ticker='SPX Index', value_date=pd.Timestamp('2026-07-24'),
                            scaler=None)
# 5 tenors (30d, 60d, 3m, 6m, 12m) x 9 moneyness (80-120); last quote on/before the date
```

**Option chains**

```python
import numpy as np
from bbg_fetch import (fetch_option_chain, recover_option_forward, run,
                       OptionPriceSource, OptionChainResult)

# One expiry, trimmed to a strike window around the money (bounds the per-option bdp count)
chain = fetch_option_chain(underlying='KOSPI2 Index', expiry='20260910',
                           num_strikes_per_side=20)

# Implied forward from put-call parity: C(K) - P(K) = exp(-r T) (F - K)
params = recover_option_forward(chain, spot=1055.58, year_fraction=48 / 365,
                                price_source=OptionPriceSource.LAST)
# params: forward, rate, r2, num_strikes_used
# (the forward is well determined; the rate is only indicative at short maturity)

# One call end to end: fetch the chain, infer spot and year fraction from it, recover
# the forward and rate, and return an OptionChainResult snapshot
result = run(underlying='KOSPI2 Index', expiry='20260910',
             strike_grid=np.linspace(700, 1400, 15))
result.forward, result.rate, result.spot, result.year_fraction

# Persist the snapshot to one self-contained CSV and read it back
result.to_csv('kospi2_20260910.csv')
result = OptionChainResult.read_csv('kospi2_20260910.csv')
```

**Futures contract table with carry**

```python
from bbg_fetch import fetch_futures_contract_table

# Full contract table: prices, bid/ask, volume, OI, days to expiry, annualised carry
curve = fetch_futures_contract_table(ticker="ES1 Index")
```

**Bond analytics by ISIN**

```python
from bbg_fetch import fetch_bonds_info

bond_data = fetch_bonds_info(
    isins=['US03522AAJ97', 'US126650CZ11'],
    fields=['id_bb', 'name', 'security_des', 'crncy', 'amt_outstanding',
            'px_last', 'yas_bond_yld', 'yas_oas_sprd', 'yas_mod_dur']
)

# With historical override
bond_hist = fetch_bonds_info(
    isins=['US03522AAJ97'],
    fields=['px_last', 'yas_bond_yld'],
    END_DATE_OVERRIDE='20231231'
)
```

**Index members and weights**

```python
from bbg_fetch import fetch_index_members_weights, fetch_bonds_info

# Index with weights (INDX_MWEIGHT)
members = fetch_index_members_weights('SPCPGN Index')

# Index members only (some indices don't have weights)
members = fetch_index_members_weights('H04064US Index', field='INDX_MEMBERS')

# Chain: get index members → fetch bond analytics
members = fetch_index_members_weights('LUACTRUU Index')
bond_data = fetch_bonds_info(
    isins=members.index.to_list(),
    fields=['name', 'px_last', 'yas_bond_yld', 'yas_mod_dur', 'bb_composite']
)
```

Fundamentals, balance-sheet and credit metrics, current market prices and FX rates, active
futures series, futures ticker utilities, CDS spreads, bond-to-issuer ISIN mapping, dividend
history and yields, ISIN-to-ticker resolution and the direct `bdp`/`bdh`/`bds` calls are
documented function by function in the
[API reference](https://bloombergfetch.readthedocs.io/en/latest/api.html); the
[request reference](https://bloombergfetch.readthedocs.io/en/latest/task_request_data.html)
walks through the request shapes and the pandas shapes they return.

### Runnable examples

Authoritative runnable scripts are indexed in [`examples/README.md`](examples/README.md). The
index separates the terminal-free installation/API check from examples that require a running,
entitled Bloomberg Terminal:

- [`examples/quickstart_no_terminal.py`](examples/quickstart_no_terminal.py) verifies the
  installed API against deterministic synthetic data; it does not test a Bloomberg connection
  and is exercised against the built wheel from outside the checkout in CI.
- [`examples/diagnose_terminal.py`](examples/diagnose_terminal.py) makes one scalar request on an
  entitled machine and reports only success state, dimensions, and schema; select a different
  request with `--ticker` and `--field`. The live diagnostic never runs in CI.
- [`examples/fetch_core_data.py`](examples/fetch_core_data.py), [`examples/fetch_div_history.py`](examples/fetch_div_history.py)
  and [`examples/fetch_option_chain.py`](examples/fetch_option_chain.py) cover the high-level
  price, reference, volatility, futures, fixed-income and constituent fetchers, trailing dividend
  yields, and option-chain fetch with put-call-parity recovery.

## Testing

```bash
uv sync --locked --group test
uv run --no-sync pytest
uv run --locked --only-group lint ruff check src/bbg_fetch tests examples
```

Terminal-free automated tests live in `tests/test_*.py` and are the CI lane; no test, example
or CI job requires a Bloomberg session. The live adjusted-price pytest module lives in
`src/bbg_fetch/tests/bbg_adj_price_vs_tri_test.py`; it requires an active Bloomberg Terminal
and is invoked explicitly from a source checkout. Component development diagnostics are
source-only runners under `src/bbg_fetch/run_local/` (`Locals`, `run_local(local=...)`); the
folder uses implicit namespace-package discovery and is excluded from wheels and source
distributions.

<a id="whats-new-in-v310"></a>
<a id="whats-new-in-v300"></a>
<a id="whats-new-in-v230"></a>
<a id="whats-new-in-v201"></a>
<a id="whats-new-in-v200"></a>

## Changelog

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Feedback & contributing <a name="feedback-contributing"></a>

- **Bug:** [open the bug-report form](https://github.com/ArturSepp/BloombergFetch/issues/new?template=bug_report.yml)
  with the package/Python versions, platform, minimal request shape, expected result, and redacted
  actual result. Never attach licensed Bloomberg responses or credentials.
- **Feature:** [open the feature-request form](https://github.com/ArturSepp/BloombergFetch/issues/new?template=feature_request.yml)
  with the field, provider, or normalisation workflow that is missing, your current workaround,
  and the smallest useful API.
- **Question or methodology:** search or open an [issue](https://github.com/ArturSepp/BloombergFetch/issues)
  and name the request shape, field mapping, or convention involved.
- **Contribution:** read [`CONTRIBUTING.md`](CONTRIBUTING.md), then browse
  [`good first issue`](https://github.com/ArturSepp/BloombergFetch/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
  or [`help wanted`](https://github.com/ArturSepp/BloombergFetch/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22)
  work.

## Citation

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{bloombergfetch,
  author = {Sepp, Artur},
  title = {{BloombergFetch}: A Python Package for Bloomberg Terminal Data Access},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ArturSepp/BloombergFetch},
  version = {3.2.1}
}
```

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, backtesting, and factsheet reporting |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and rolling backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor-model estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) *(this package)* | Bloomberg data in pandas DataFrames |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain data and queries |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Numba-vectorised BSM and Bachelier pricing |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic-volatility pricing and calibration |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Closed-form trend-following analytics |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Multi-factor PME for private assets |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Goal-based allocation under regime-switching jump-diffusions |

`bbg-fetch` is a standalone data-access package. `option-chain-analytics` can use it through an
optional Bloomberg provider; research data fetched here can also flow into `qis`, but neither
workflow creates a runtime dependency for `bbg-fetch`.

## License

MIT. See [LICENSE.txt](LICENSE.txt).

## Disclaimer

`bbg-fetch` is distributed FREE & WITHOUT ANY WARRANTY under the MIT License. See
[LICENSE.txt](LICENSE.txt) in the release for details. Use the routes in
[Feedback & contributing](#feedback-contributing) for bugs, feature requests and
methodology questions.
