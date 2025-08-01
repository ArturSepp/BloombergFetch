
# 🚀 **BloombergFetch: bbg-fetch**
    
A comprehensive Python package for fetching financial data from Bloomberg Terminal using the `xbbg` library. This package provides convenient wrapper functions for accessing various types of Bloomberg data including equities, futures, bonds, options, FX rates, and more.
---

| 📊 Metric | 🔢 Value |
|-----------|----------|
| PyPI Version | ![PyPI](https://img.shields.io/pypi/v/bbg-fetch?style=flat-square) |
| Python Versions | ![Python](https://img.shields.io/pypi/pyversions/bbg-fetch?style=flat-square) |
| License | ![License](https://img.shields.io/github/license/ArturSepp/BloombergFetch.svg?style=flat-square)|


### 📈 Package Statistics

| 📊 Metric | 🔢 Value |
|-----------|----------|
| Total Downloads | [![Total](https://pepy.tech/badge/bbg-fetch)](https://pepy.tech/project/bbg-fetch) |
| Monthly | ![Monthly](https://pepy.tech/badge/bbg-fetch/month) |
| Weekly | ![Weekly](https://pepy.tech/badge/bbg-fetch/week) |
| GitHub Stars | ![GitHub stars](https://img.shields.io/github/stars/ArturSepp/BloombergFetch?style=flat-square&logo=github) |
| GitHub Forks | ![GitHub forks](https://img.shields.io/github/forks/ArturSepp/BloombergFetch?style=flat-square&logo=github) |


# Bloomberg Financial Data Package

## 🚀 Features

- **Equity Data**: Historical prices, fundamentals, dividend history
- **Futures Data**: Active contracts, contract tables, roll analysis
- **Options Data**: Implied volatility surfaces, option chains
- **Fixed Income**: Bond information, yield curves, credit spreads
- **FX Data**: Currency rates and volatility
- **Index Data**: Constituents and weights
- **Fundamentals**: Balance sheet data, financial ratios

## 📦 Installation



Install using
```python 
pip install bbg_fetch
```

Upgrade using
```python 
pip install --upgrade bbg_fetch
```

Close using
```python 
git clone https://github.com/ArturSepp/BloombergFetch.git
```

### Prerequisites

1. **Bloomberg Terminal Access**: You need access to a Bloomberg Terminal or BPIPE
2. **Bloomberg API**: Install the Bloomberg API library:
   ```bash
   pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
   ```

### Package Dependencies

```bash
pip install numpy pandas xbbg
```

## 🔧 Quick Start

### Basic Usage

```python
import pandas as pd
from bbg_fetch import (
    fetch_field_timeseries_per_tickers,
    fetch_fundamentals,
    fetch_last_prices
)

# Fetch historical prices for multiple tickers
prices = fetch_field_timeseries_per_tickers(
    tickers=['AAPL US Equity', 'MSFT US Equity'],
    field='PX_LAST',
    start_date=pd.Timestamp('2020-01-01')
)

# Get fundamental data
fundamentals = fetch_fundamentals(
    tickers=['AAPL US Equity', 'GOOGL US Equity'],
    fields=['security_name', 'gics_sector_name', 'market_cap']
)

# Fetch current FX rates
fx_rates = fetch_last_prices()
print(fx_rates)
```

### Advanced Examples

#### Options Implied Volatility
```python
from bbg_fetch import fetch_vol_timeseries, IMPVOL_FIELDS_DELTA

# Fetch SPX implied volatility surface
vol_data = fetch_vol_timeseries(
    ticker='SPX Index',
    vol_fields=IMPVOL_FIELDS_DELTA,
    start_date=pd.Timestamp('2023-01-01')
)
```

#### Futures Contract Analysis
```python
from bbg_fetch import fetch_futures_contract_table, fetch_active_futures

# Get futures contract table
contracts = fetch_futures_contract_table(ticker="ES1 Index")

# Fetch active futures data
front_month, second_month = fetch_active_futures(generic_ticker='ES1 Index')
```

#### Bond Analysis
```python
from bbg_fetch import fetch_bonds_info

# Get bond information by ISIN
bond_data = fetch_bonds_info(
    isins=['US03522AAJ97', 'US126650CZ11'],
    fields=['name', 'px_last', 'yas_bond_yld', 'yas_mod_dur']
)
```

## 📊 Main Functions

### Price Data
- `fetch_field_timeseries_per_tickers()`: Historical data for multiple tickers
- `fetch_fields_timeseries_per_ticker()`: Multiple fields for single ticker
- `fetch_last_prices()`: Current market prices

### Fundamental Data
- `fetch_fundamentals()`: Company fundamentals and metadata
- `fetch_balance_data()`: Balance sheet and financial ratios
- `fetch_dividend_history()`: Historical dividend payments
- `fetch_div_yields()`: Dividend yield calculations

### Derivatives
- `fetch_vol_timeseries()`: Options implied volatility
- `fetch_futures_contract_table()`: Futures contract specifications
- `fetch_active_futures()`: Active futures price series

### Fixed Income
- `fetch_bonds_info()`: Bond specifications and pricing
- `fetch_cds_info()`: Credit default swap information

### Index & Constituents
- `fetch_index_members_weights()`: Index constituents and weights
- `fetch_tickers_from_isins()`: Convert ISINs to Bloomberg tickers

## 🔍 Predefined Field Mappings

The package includes predefined field mappings for common data types:

### FX Currencies
```python
FX_DICT = {
    'EURUSD Curncy': 'EUR',
    'GBPUSD Curncy': 'GBP',
    'JPYUSD Curncy': 'JPY',
    # ... more currencies
}
```

### Implied Volatility Fields
- `IMPVOL_FIELDS_MNY_30DAY`: 30-day moneyness-based vol fields
- `IMPVOL_FIELDS_MNY_60DAY`: 60-day moneyness-based vol fields
- `IMPVOL_FIELDS_DELTA`: Delta-based vol fields for FX options

## ⚙️ Configuration

### Date Settings
```python
DEFAULT_START_DATE = pd.Timestamp('01Jan1959')  # Default start date for historical data
VOLS_START_DATE = pd.Timestamp('03Jan2005')     # Default start for volatility data
```

### Data Adjustments
Most price functions support Bloomberg's corporate action adjustments:
- `CshAdjNormal`: Normal cash adjustments (default: True)
- `CshAdjAbnormal`: Abnormal cash adjustments (default: True)  
- `CapChg`: Capital changes (default: True)

## 🧪 Testing

The package includes comprehensive unit tests. Run specific tests:

```python
from bbg_fetch import run_unit_test, UnitTests

# Test different functionalities
run_unit_test(UnitTests.FIELD_TIMESERIES_PER_TICKERS)
run_unit_test(UnitTests.IMPLIED_VOL_TIME_SERIES)
run_unit_test(UnitTests.BOND_INFO)
```

Available test categories:
- `FIELD_TIMESERIES_PER_TICKERS`
- `FUNDAMENTALS`  
- `ACTIVE_FUTURES`
- `IMPLIED_VOL_TIME_SERIES`
- `BOND_INFO`
- `BALANCE_DATA`
- And more...

## 📝 Error Handling

The package includes robust error handling:
- Automatic retries for failed Bloomberg requests
- Warning messages for missing data
- Graceful handling of empty datasets
- Data validation and cleaning

## 🔗 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **xbbg**: Bloomberg data access library
- **blpapi**: Bloomberg API (requires special installation)

## 📚 Bloomberg Field Reference

Common Bloomberg fields used in this package:
- `PX_LAST`: Last price
- `PX_OPEN/HIGH/LOW`: OHLC prices
- `VOLUME`: Trading volume
- `MARKET_CAP`: Market capitalization
- `YAS_BOND_YLD`: Bond yield
- `GICS_SECTOR_NAME`: GICS sector classification

## ⚠️ Important Notes

1. **Bloomberg Terminal Required**: This package requires access to Bloomberg Terminal or BPIPE
2. **Rate Limits**: Bloomberg API has rate limits - the package includes retry logic
3. **Data Availability**: Not all fields are available for all instruments
4. **Time Zones**: Default timezone handling is UTC, but can be configured

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- Add unit tests for new functionality
- Update documentation for new features
- Test with multiple Bloomberg data types

## 📄 License

[Add your license information here]

## 🆘 Support

For Bloomberg API issues:
- Check Bloomberg Terminal connection
- Verify field names using Bloomberg's FLDS function
- Ensure proper instrument formatting (e.g., "AAPL US Equity")

For package-specific issues:
- Check unit tests for usage examples
- Verify data availability for requested date ranges
- Review Bloomberg's data licensing terms
