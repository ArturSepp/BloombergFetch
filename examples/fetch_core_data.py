"""
runnable examples for the core.py data fetchers.

Run on the Bloomberg machine. Pick a case and run: each calls one fetcher with
representative arguments and prints the result. One case per fetcher, gated behind the
LocalTest enum and the run_local_test dispatcher. Option chain fetching has its own
example in fetch_option_chain.py.
"""
# packages
import pandas as pd
from enum import Enum
# bbg
import bbg_fetch as bbg


class LocalTest(Enum):
    FIELD_TIMESERIES_PER_TICKERS = 1   # fetch_field_timeseries_per_tickers
    FIELDS_TIMESERIES_PER_TICKER = 2   # fetch_fields_timeseries_per_ticker
    FUNDAMENTALS = 3                   # fetch_fundamentals
    ACTIVE_FUTURES = 4                 # fetch_active_futures
    CONTRACT_TABLE = 5                 # fetch_futures_contract_table
    IMPLIED_VOL_TIME_SERIES = 6        # fetch_vol_timeseries
    LAST_PRICES = 7                    # fetch_last_prices
    BOND_INFO = 8                      # fetch_bonds_info
    CDS_INFO = 9                       # fetch_cds_info
    BALANCE_DATA = 10                  # fetch_balance_data
    TICKERS_FROM_ISIN = 11             # fetch_tickers_from_isins
    DIVIDEND_HISTORY = 12              # fetch_dividend_history
    DIV_YIELDS = 13                    # fetch_div_yields
    INDEX_MEMBERS = 14                 # fetch_index_members_weights
    ISSUER_ISINS_FROM_BONDS = 15       # fetch_issuer_isins_from_bond_isins
    TICKER_HELPERS = 16                # instrument_to_active_ticker, contract_to_instrument
    VOL_SURFACE = 17                   # fetch_vol_surface


def run_local_test(local_test: LocalTest) -> None:
    """dispatch one fetcher example and print its result."""

    pd.set_option('display.max_columns', 500)

    if local_test == LocalTest.FIELD_TIMESERIES_PER_TICKERS:
        # one field across many tickers, dict keys queried and values used as column labels
        df = bbg.fetch_field_timeseries_per_tickers(tickers={'SPX Index': 'SPX', 'NKY Index': 'NKY'},
                                                    field='PX_LAST',
                                                    start_date=pd.Timestamp('01Jan2020'))
        print(df)

    elif local_test == LocalTest.FIELDS_TIMESERIES_PER_TICKER:
        # many fields for one ticker
        df = bbg.fetch_fields_timeseries_per_ticker(ticker='ES1 Index',
                                                    fields=('PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST'))
        print(df)

    elif local_test == LocalTest.FUNDAMENTALS:
        # reference (BDP) fundamentals, dict-mapped column labels
        df = bbg.fetch_fundamentals(tickers=['AAPL US Equity', 'MSFT US Equity'],
                                    fields={'security_name': 'Name', 'gics_sector_name': 'Sector'})
        print(df)

    elif local_test == LocalTest.ACTIVE_FUTURES:
        # front and second generic contracts (two series)
        front, second = bbg.fetch_active_futures(generic_ticker='ES1 Index')
        print(front.tail())
        print(second.tail())

    elif local_test == LocalTest.CONTRACT_TABLE:
        # the futures contract chain with carry, generic number and timestamp
        df = bbg.fetch_futures_contract_table(ticker='ES1 Index')
        print(df)

    elif local_test == LocalTest.IMPLIED_VOL_TIME_SERIES:
        # delta-quoted implied vol history plus the underlying
        df = bbg.fetch_vol_timeseries(ticker='SPX Index', vol_fields=bbg.IMPVOL_FIELDS_DELTA)
        print(df.tail())

    elif local_test == LocalTest.LAST_PRICES:
        # last prices; default tickers are the FX_DICT crosses
        prices = bbg.fetch_last_prices()
        print(prices)

    elif local_test == LocalTest.BOND_INFO:
        # reference data for bonds by ISIN
        df = bbg.fetch_bonds_info(isins=['US03522AAJ97', 'US126650CZ11'])
        print(df)

    elif local_test == LocalTest.CDS_INFO:
        # 5y CDS spread tickers for equity issuers
        df = bbg.fetch_cds_info(equity_tickers=['ABI BB Equity', 'CVS US Equity'])
        print(df)

    elif local_test == LocalTest.BALANCE_DATA:
        # balance-sheet and credit fundamentals
        df = bbg.fetch_balance_data(tickers=['ABI BB Equity', 'T US Equity', 'JPM US Equity'])
        print(df)

    elif local_test == LocalTest.TICKERS_FROM_ISIN:
        # resolve ISINs to primary-exchange composite tickers
        tickers = bbg.fetch_tickers_from_isins(isins=['US88160R1014', 'IL0065100930'])
        print(tickers)

    elif local_test == LocalTest.DIVIDEND_HISTORY:
        # per-event dividend history
        df = bbg.fetch_dividend_history(ticker='TIP US Equity')
        print(df)

    elif local_test == LocalTest.DIV_YIELDS:
        # per-event amounts, trailing-12m sums, and trailing-12m yields (three frames)
        divs, divs_1y, divs_yield = bbg.fetch_div_yields(tickers={'SPY US Equity': 'SPY',
                                                                  'TIP US Equity': 'TIP'})
        print(divs_yield.tail())

    elif local_test == LocalTest.INDEX_MEMBERS:
        # index constituents; INDX_MEMBERS returns tickers without weights
        members = bbg.fetch_index_members_weights(index='SPX Index', field='INDX_MEMBERS')
        print(members)

    elif local_test == LocalTest.ISSUER_ISINS_FROM_BONDS:
        # bond ISIN -> issuer equity ISIN
        issuer = bbg.fetch_issuer_isins_from_bond_isins(bond_isins=['XS3034073836', 'USY0616GAA14'])
        print(issuer)

    elif local_test == LocalTest.TICKER_HELPERS:
        # string helpers, no Bloomberg call
        print(bbg.instrument_to_active_ticker(instrument='ES1 Index', num=2))  # -> 'ES2 Index'
        print(bbg.contract_to_instrument(future='ES1 Index'))                  # -> 'ES'

    elif local_test == LocalTest.VOL_SURFACE:
        # implied vol surface for one date: tenor rows x moneyness columns, in percent.
        # non_null_share flags whether the 80 and 120 columns actually populate.
        surface = bbg.fetch_vol_surface(ticker='SPX Index', scaler=None)  # swap to 'KOSPI2 Index' as needed
        print(surface.to_string())
        print(surface.notna().mean().rename('non_null_share').to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':

    local_test = LocalTest.VOL_SURFACE

    is_run_all_tests = False
    if is_run_all_tests:
        for local_test in LocalTest:
            run_local_test(local_test=local_test)
    else:
        run_local_test(local_test=local_test)
