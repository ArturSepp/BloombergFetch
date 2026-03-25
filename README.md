# bbg-fetch

**Bloomberg data in DataFrames. No boilerplate. No bloat.**

`bbg-fetch` gives you clean, production-ready functions for the Bloomberg data you actually need — prices, fundamentals, vol surfaces, futures curves, bond analytics, index constituents — without writing event loops or managing sessions.

```python
from bbg_fetch import fetch_field_timeseries_per_tickers

prices = fetch_field_timeseries_per_tickers(
    tickers={'ES1 Index': 'SPX', 'TY1 Comdty': '10yUST'},
    field='PX_LAST',
    start_date=pd.Timestamp('2020-01-01')
)
# Returns a clean DataFrame with renamed columns, sorted index, split/div adjusted
```

![PyPI](https://img.shields.io/pypi/v/bbg-fetch?style=flat-square)
![Python](https://img.shields.io/pypi/pyversions/bbg-fetch?style=flat-square)
![License](https://img.shields.io/github/license/ArturSepp/BloombergFetch.svg?style=flat-square)
[![Downloads](https://pepy.tech/badge/bbg-fetch)](https://pepy.tech/project/bbg-fetch)
![Stars](https://img.shields.io/github/stars/ArturSepp/BloombergFetch?style=flat-square&logo=github)

---

## Why bbg-fetch?

### vs. raw blpapi

Writing Bloomberg queries with `blpapi` directly means 40–60 lines of session management, request construction, event-loop iteration, and response parsing — for every single query. You end up writing the same boilerplate wrapper in every project.

**With blpapi:**
```python
import blpapi

opts = blpapi.SessionOptions()
opts.setServerHost('localhost')
opts.setServerPort(8194)
session = blpapi.Session(opts)
session.start()
session.openService('//blp/refdata')
service = session.getService('//blp/refdata')
request = service.createRequest('HistoricalDataRequest')
request.getElement('securities').appendValue('AAPL US Equity')
request.getElement('fields').appendValue('PX_LAST')
request.set('startDate', '20200101')
request.set('endDate', '20241231')
request.set('adjustmentNormal', True)
request.set('adjustmentAbnormal', True)
request.set('adjustmentSplit', True)
session.sendRequest(request)
# ... then 30 more lines to parse the event loop into a DataFrame
```

**With bbg-fetch:**
```python
from bbg_fetch import fetch_field_timeseries_per_tickers

prices = fetch_field_timeseries_per_tickers(
    tickers=['AAPL US Equity'],
    field='PX_LAST',
    start_date=pd.Timestamp('2020-01-01')
)
```

Same result. One line.

### vs. xbbg

xbbg is a capable library, but its v1 rewrite introduced a Rust core, `narwhals`, and `pyarrow>=22` as hard dependencies — over 30MB of transitive installs for what most quant teams use: `bdp`, `bdh`, and `bds`.

| | **bbg-fetch** | **xbbg v1** |
|---|---|---|
| Dependencies | `numpy` + `pandas` | `narwhals` + `pyarrow` + Rust binary |
| Install size | ~50KB (plus blpapi) | ~30MB+ transitive |
| Python support | 3.9–3.12 | 3.10–3.14 |
| Intraday bars / streaming | No | Yes |
| Exchange-aware market hours | No | Yes |
| Session management | Automatic singleton | Configurable pool |
| Debug Bloomberg errors | Your code, 400 lines | Third-party internals |

**bbg-fetch is for teams that need historical, reference, and bulk data — reliably, with minimal dependencies.** If you need intraday bars or real-time streaming, use xbbg.

### The sweet spot

bbg-fetch sits between raw blpapi (too low-level) and xbbg (too many dependencies). It wraps the three Bloomberg services that cover 95% of quant workflows — BDP, BDH, BDS — into high-level functions that return clean DataFrames with proper column naming, corporate action adjustments, and index alignment.

---

## What you get

### Multi-asset coverage

- **Equities**: Historical prices with split/dividend adjustments, fundamentals, dividend history
- **Futures**: Contract tables with carry analysis, active contract series, roll handling
- **Options**: Implied volatility surfaces (moneyness and delta), option chains
- **Fixed Income**: Bond pricing and analytics by ISIN, yield curves, CDS spreads
- **FX**: Currency rates and volatility
- **Indices**: Constituent weights, ISIN-to-ticker resolution

### Production-ready details

- Dict-based ticker renaming: `{'ES1 Index': 'SPX'}` → DataFrame columns named `SPX`
- Automatic retry on Bloomberg connection flakes
- Corporate action adjustments on by default (normal, abnormal, splits)
- Predefined field mappings for vol surfaces (30d/60d/3m/6m/12m moneyness, delta)
- Carry computation built into futures contract tables

---

## Installation

### 1. Install blpapi

```bash
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
```

Pre-built wheels for Python 3.8–3.12 on Windows/macOS/Linux bundle the C++ SDK automatically.

**Corporate proxy?** Download the `.whl` from `https://blpapi.bloomberg.com/repository/releases/python/simple/blpapi/` via browser, then:
```bash
pip install /path/to/blpapi-3.24.6-cp312-cp312-win_amd64.whl
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

**Requirements:** Python 3.9+, Bloomberg Terminal running on the same machine.

---

## Examples

### Prices across tickers (with renaming)

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

# Unadjusted prices (for futures, rates, etc.)
raw = fetch_field_timeseries_per_tickers(
    tickers=['TY1 Comdty'], field='PX_LAST',
    CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False
)
```

### Multiple fields for a single ticker

```python
from bbg_fetch import fetch_fields_timeseries_per_ticker

# OHLC data
ohlc = fetch_fields_timeseries_per_ticker(
    ticker='AAPL US Equity',
    fields=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST'],
    start_date=pd.Timestamp('2023-01-01')
)

# Futures-specific fields
fut_data = fetch_fields_timeseries_per_ticker(
    ticker='ES1 Index',
    fields=['PX_LAST', 'FUT_DAYS_EXP'],
    CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False
)
```

### Company fundamentals

```python
from bbg_fetch import fetch_fundamentals

# Basic security info
info = fetch_fundamentals(
    tickers=['AAPL US Equity', 'GOOGL US Equity'],
    fields=['security_name', 'gics_sector_name', 'crncy', 'market_cap']
)

# Fund-level data
fund_info = fetch_fundamentals(
    tickers=['HAHYIM2 HK Equity'],
    fields=['name', 'front_load', 'back_load', 'fund_mgr_stated_fee', 'fund_min_invest']
)

# Dict-based renaming works for both tickers and fields
info = fetch_fundamentals(
    tickers={'AAPL US Equity': 'Apple', 'MSFT US Equity': 'Microsoft'},
    fields={'security_name': 'Name', 'gics_sector_name': 'Sector'}
)
```

### Balance sheet and credit metrics

```python
from bbg_fetch import fetch_balance_data

credit = fetch_balance_data(
    tickers=['ABI BB Equity', 'T US Equity', 'JPM US Equity'],
    fields=('GICS_SECTOR_NAME', 'TOT_COMMON_EQY', 'BS_LT_BORROW',
            'NET_DEBT_TO_EBITDA', 'INTEREST_COVERAGE_RATIO',
            'FREE_CASH_FLOW_MARGIN', 'EARN_YLD')
)
```

### Current market prices

```python
from bbg_fetch import fetch_last_prices
from bbg_fetch.core import FX_DICT

# FX rates (uses built-in FX_DICT by default: 19 major pairs)
fx = fetch_last_prices()

# Custom tickers
prices = fetch_last_prices(tickers=['AAPL US Equity', 'SPX Index', 'USGG10YR Index'])

# With renaming
prices = fetch_last_prices(
    tickers={'ES1 Index': 'SPX Fut', 'TY1 Comdty': '10y Fut', 'GC1 Comdty': 'Gold Fut'}
)
```

### Implied volatility surface

```python
from bbg_fetch import fetch_vol_timeseries, IMPVOL_FIELDS_DELTA
from bbg_fetch.core import (IMPVOL_FIELDS_MNY_30DAY, IMPVOL_FIELDS_MNY_60DAY,
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

# Single tenor with raw field names (no renaming)
vol_30d = fetch_vol_timeseries(
    ticker='SPX Index',
    vol_fields=['30DAY_IMPVOL_100.0%MNY_DF', '30DAY_IMPVOL_90.0%MNY_DF'],
    start_date=pd.Timestamp('2020-01-01')
)
```

### Futures contract table with carry

```python
from bbg_fetch import fetch_futures_contract_table

# Full contract table: prices, bid/ask, volume, OI, days to expiry, annualized carry
curve = fetch_futures_contract_table(ticker="ES1 Index")

# Nikkei futures
nk_curve = fetch_futures_contract_table(ticker="NK1 Index")
```

### Active futures price series

```python
from bbg_fetch import fetch_active_futures

# Front and second month continuous series
front, second = fetch_active_futures(generic_ticker='ES1 Index')

# Start from second generic (e.g., for roll analysis)
gen2, gen3 = fetch_active_futures(generic_ticker='ES1 Index', first_gen=2)
```

### Futures ticker utilities

```python
from bbg_fetch import instrument_to_active_ticker, contract_to_instrument

# ES1 Index → ES
instrument_to_active_ticker('ES1 Index', num=3)   # → 'ES3 Index'
contract_to_instrument('ES1 Index')                 # → 'ES'
contract_to_instrument('TY1 Comdty')                # → 'TY'
```

### Bond analytics by ISIN

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

### CDS spreads

```python
from bbg_fetch import fetch_cds_info

cds = fetch_cds_info(
    equity_tickers=['ABI BB Equity', 'CVS US Equity', 'JPM US Equity'],
    field='cds_spread_ticker_5y'
)
```

### Bond ISIN → issuer equity ISIN mapping

```python
from bbg_fetch import fetch_issuer_isins_from_bond_isins

# Map bond ISINs to their issuer's equity ISIN
issuer_map = fetch_issuer_isins_from_bond_isins(
    bond_isins=['XS3034073836', 'USY0616GAA14', 'XS3023923314']
)
# Returns: pd.Series with bond ISIN as index, issuer equity ISIN as values
```

### Dividend history and yields

```python
from bbg_fetch import fetch_dividend_history, fetch_div_yields

# Full dividend history
divs = fetch_dividend_history(ticker='AAPL US Equity')
# Columns: declared_date, ex_date, record_date, payable_date,
#          dividend_amount, dividend_frequency, dividend_type

# Trailing 1-year dividend yield for multiple tickers
div_amounts, div_yields_1y = fetch_div_yields(
    tickers=['AHYG SP Equity', 'TIP US Equity'],
    dividend_types=('Income', 'Distribution')
)

# With renaming
div_amounts, div_yields_1y = fetch_div_yields(
    tickers={'TIP US Equity': 'TIPS', 'SDHA LN Equity': 'Asia HY'}
)
```

### Index members and weights

```python
from bbg_fetch import fetch_index_members_weights, fetch_bonds_info

# Index with weights (INDX_MWEIGHT)
members = fetch_index_members_weights('SPCPGN Index')

# Index members only (some indices don't have weights)
members = fetch_index_members_weights('H04064US Index', field='INDX_MEMBERS')

# Historical members
members_hist = fetch_index_members_weights(
    'I31415US Index', END_DATE_OVERRIDE='20200101'
)

# Chain: get index members → fetch bond analytics
members = fetch_index_members_weights('LUACTRUU Index')
bond_data = fetch_bonds_info(
    isins=members.index.to_list(),
    fields=['name', 'px_last', 'yas_bond_yld', 'yas_mod_dur', 'bb_composite']
)
```

### ISIN to Bloomberg ticker resolution

```python
from bbg_fetch import fetch_tickers_from_isins

# Convert ISINs to Bloomberg composite tickers
tickers = fetch_tickers_from_isins(isins=['US88160R1014', 'IL0065100930'])
# Returns: ['TSLA US Equity', ...] (with primary exchange)
```

### Direct BDP / BDH / BDS

For ad-hoc queries not covered by the high-level functions:

```python
from bbg_fetch import bdp, bdh, bds

# Reference data (BDP)
ref = bdp('AAPL US Equity', ['Security_Name', 'GICS_Sector_Name', 'PX_LAST'])

# Historical data with adjustments (BDH)
hist = bdh('SPX Index', 'PX_LAST', '2024-01-01', '2024-12-31',
           CshAdjNormal=True, CshAdjAbnormal=True, CapChg=True)

# Bulk data — option chains (BDS)
chain = bds('TSLA US Equity', 'CHAIN_TICKERS',
            CHAIN_PUT_CALL_TYPE_OVRD='PUT', CHAIN_POINTS_OVRD=1000)

# Yield curve construction
yc_members = bds("YCGT0025 Index", "INDX_MEMBERS")
yc_data = bdp(yc_members.member_ticker_and_exchange_code.tolist(),
              ['YLD_YTM_ASK', 'SECURITY NAME', 'MATURITY'])

# Clean up session explicitly (optional)
from bbg_fetch import disconnect
disconnect()
```

---

## Function reference

### Price data
| Function | Description |
|----------|-------------|
| `fetch_field_timeseries_per_tickers()` | One field across multiple tickers (with optional dict-based renaming) |
| `fetch_fields_timeseries_per_ticker()` | Multiple fields for a single ticker |
| `fetch_last_prices()` | Snapshot of current prices |

### Fundamentals
| Function | Description |
|----------|-------------|
| `fetch_fundamentals()` | Company metadata and fundamentals (dict renaming for tickers and fields) |
| `fetch_balance_data()` | Balance sheet ratios and credit metrics |
| `fetch_dividend_history()` | Full dividend history (dates, amounts, types) |
| `fetch_div_yields()` | Per-ticker dividend amounts and trailing 1-year yield |

### Derivatives
| Function | Description |
|----------|-------------|
| `fetch_vol_timeseries()` | Implied vol surface with underlying + rates (supports list-of-dicts for multi-tenor) |
| `fetch_futures_contract_table()` | Contract specs, carry, timestamps |
| `fetch_active_futures()` | Front + second month price series |

### Fixed income
| Function | Description |
|----------|-------------|
| `fetch_bonds_info()` | Bond analytics by ISIN (with optional date override) |
| `fetch_cds_info()` | CDS spread tickers from equity tickers |
| `fetch_issuer_isins_from_bond_isins()` | Bond ISIN → issuer equity ISIN mapping |

### Index and resolution
| Function | Description |
|----------|-------------|
| `fetch_index_members_weights()` | Constituents and weights (configurable BDS field) |
| `fetch_tickers_from_isins()` | ISIN → Bloomberg composite ticker |

### Futures utilities
| Function | Description |
|----------|-------------|
| `instrument_to_active_ticker()` | `'ES1 Index'` + `num=3` → `'ES3 Index'` |
| `contract_to_instrument()` | `'ES1 Index'` → `'ES'` (strip generic number) |

### Low-level blpapi wrappers
| Function | Description |
|----------|-------------|
| `bdp()` | Bloomberg Data Point — reference data (BDP in Excel) |
| `bdh()` | Bloomberg Data History — historical end-of-day data |
| `bds()` | Bloomberg Data Set — bulk data (chains, members, dividends) |
| `disconnect()` | Explicitly stop the shared blpapi session |

---

## Predefined field mappings

### FX currencies

```python
from bbg_fetch.core import FX_DICT
# 19 major pairs: EUR, GBP, CHF, CAD, JPY, AUD, NZD, MXN, HKD, SEK,
#                 PLN, KRW, TRY, SGD, ZAR, CNY, INR, TWD, NOK
```

### Implied volatility fields

| Mapping | Description |
|---------|-------------|
| `IMPVOL_FIELDS_MNY_30DAY` | 30-day moneyness-based vol (80%–120%) |
| `IMPVOL_FIELDS_MNY_60DAY` | 60-day moneyness-based vol |
| `IMPVOL_FIELDS_MNY_3MTH` | 3-month moneyness-based vol |
| `IMPVOL_FIELDS_MNY_6MTH` | 6-month moneyness-based vol |
| `IMPVOL_FIELDS_MNY_12M` | 12-month moneyness-based vol |
| `IMPVOL_FIELDS_DELTA` | 1M/2M delta-based vol (10Δ–50Δ puts and calls) |

---

## Configuration

### Date defaults

```python
DEFAULT_START_DATE = pd.Timestamp('01Jan1959')  # Historical data
VOLS_START_DATE = pd.Timestamp('03Jan2005')     # Volatility data
```

### Corporate action adjustments

Most price functions support Bloomberg's adjustment flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CshAdjNormal` | `True` | Normal cash dividends |
| `CshAdjAbnormal` | `True` | Special dividends |
| `CapChg` | `True` | Stock splits and capital changes |

---

## Testing

Integration tests require an active Bloomberg Terminal connection:

```python
from bbg_fetch import run_local_test, LocalTests

run_local_test(LocalTests.FIELD_TIMESERIES_PER_TICKERS)
run_local_test(LocalTests.IMPLIED_VOL_TIME_SERIES)
run_local_test(LocalTests.CONTRACT_TABLE)
run_local_test(LocalTests.BOND_INFO)
run_local_test(LocalTests.DIVIDEND)
run_local_test(LocalTests.BOND_MEMBERS)
```

Available tests: `FIELD_TIMESERIES_PER_TICKERS`, `FIELDS_TIMESERIES_PER_TICKER`, `FUNDAMENTALS`, `ACTIVE_FUTURES`, `CONTRACT_TABLE`, `IMPLIED_VOL_TIME_SERIES`, `BOND_INFO`, `LAST_PRICES`, `CDS_INFO`, `BALANCE_DATA`, `TICKERS_FROM_ISIN`, `DIVIDEND`, `BOND_MEMBERS`, `INDEX_MEMBERS`, `OPTION_CHAIN`, `YIELD_CURVE`, `CHECK`, `MEMBERS`.

---

## Package structure

```
bbg_fetch/
    __init__.py       # Public API
    _blp_api.py       # Direct blpapi shim (bdp, bdh, bds) — 400 lines, zero dependencies
    core.py           # High-level fetch functions
    local_tests.py    # Integration tests (requires Bloomberg Terminal)
```

## Troubleshooting

### "No module named blpapi"
Install from Bloomberg's package index — see Installation above.

### "UnboundLocalError: cannot access local variable 'toPy'"
The C++ DLLs bundled with blpapi failed to load. Reinstall: `pip uninstall blpapi -y` then reinstall. If on Python 3.13+, downgrade to 3.12.

### Corporate proxy blocks Bloomberg's pip index
Download the `.whl` file manually from `https://blpapi.bloomberg.com/repository/releases/python/simple/blpapi/` via browser and install locally with `pip install /path/to/blpapi-*.whl`.

### Empty DataFrames returned
Ensure the Bloomberg Terminal is running (blpapi connects to `localhost:8194`). Verify field names using Bloomberg's `FLDS` function and instrument formatting (e.g., `"AAPL US Equity"`, `"ES1 Index"`, `"EURUSD Curncy"`). Some indices support `INDX_MWEIGHT` (with weights) while others only support `INDX_MEMBERS` — use the `field` parameter in `fetch_index_members_weights()` accordingly.

### "No module named pip" in venv
Bootstrap pip first: `python -m ensurepip --upgrade`, then install.

### PowerShell path errors
Use `.\` prefix for relative paths: `.\.venv\Scripts\python.exe`, not `.venv\Scripts\python.exe`.

---

## What's new in v2.0.0

- **`xbbg` dependency removed** — direct `blpapi` interface, no transitive Rust/pyarrow/narwhals
- **`field` parameter** added to `fetch_index_members_weights()` — supports `INDX_MWEIGHT`, `INDX_MEMBERS`, `INDX_MEMBERS3`
- **Test code extracted** to `local_tests.py` with `run_local_test()` (renamed from `run_unit_test`)
- **`bdp()`, `bdh()`, `bds()` exported** for direct low-level access
- **Robust field name handling** — Bloomberg's inconsistent casing/spacing/hyphens normalized automatically
- **Migration from v1.x:** replace `run_unit_test` → `run_local_test`, all other imports unchanged

## License

MIT. See [LICENSE.txt](LICENSE.txt).

## Citation

```bibtex
@software{bloombergfetch,
  author = {Sepp, Artur},
  title = {{BloombergFetch}: A Python Package for Bloomberg Terminal Data Access},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ArturSepp/BloombergFetch},
  version = {2.0.0}
}
```