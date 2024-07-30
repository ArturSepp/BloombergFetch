"""
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
GFUT
"""

# packages
import re
import warnings
import datetime
import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Optional, Tuple, Dict, Union
from xbbg import blp

DEFAULT_START_DATE = pd.Timestamp('01Jan1959')
VOLS_START_DATE = pd.Timestamp('03Jan2005')

FX_DICT = {
    'EURUSD Curncy': 'EUR',
    'GBPUSD Curncy': 'GBP',
    'CHFUSD Curncy': 'CHF',
    'CADUSD Curncy': 'CAD',
    'JPYUSD Curncy': 'JPY',
    'AUDUSD Curncy': 'AUD',
    'NZDUSD Curncy': 'NZD',
    'MXNUSD Curncy': 'MXN',
    'HKDUSD Curncy': 'HKD',
    'SEKUSD Curncy': 'SEK',
    'PLNUSD Curncy': 'PLN',
    'KRWUSD Curncy': 'KRW',
    'TRYUSD Curncy': 'TRY',
    'SGDUSD Curncy': 'SGD',
    'ZARUSD Curncy': 'ZAR',
    'CNYUSD Curncy': 'CNY',
    'INRUSD Curncy': 'INR',
    'TWDUSD Curncy': 'TWD',
    'NOKUSD Curncy': 'NOK'
}


IMPVOL_FIELDS_MNY_30DAY = {'30DAY_IMPVOL_80%MNY_DF': '30d80.0',
                           '30DAY_IMPVOL_90.0%MNY_DF': '30d90.0',
                           '30DAY_IMPVOL_95.0%MNY_DF': '30d95.0',
                           '30DAY_IMPVOL_97.5%MNY_DF': '30d97.5',
                           '30DAY_IMPVOL_100.0%MNY_DF': '30d100.0',
                           '30DAY_IMPVOL_102.5%MNY_DF': '30d102.5',
                           '30DAY_IMPVOL_105.0%MNY_DF': '30d105.0',
                           '30DAY_IMPVOL_110.0%MNY_DF': '30d110.0',
                           '30DAY_IMPVOL_120%MNY_DF': '30d120.0'}

IMPVOL_FIELDS_MNY_60DAY = {'60DAY_IMPVOL_80%MNY_DF': '60d80.0',
                           '60DAY_IMPVOL_90.0%MNY_DF': '60d90.0',
                           '60DAY_IMPVOL_95.0%MNY_DF': '60d95.0',
                           '60DAY_IMPVOL_97.5%MNY_DF': '60d97.5',
                           '60DAY_IMPVOL_100.0%MNY_DF': '60d100.0',
                           '60DAY_IMPVOL_102.5%MNY_DF': '60d102.5',
                           '60DAY_IMPVOL_105.0%MNY_DF': '60d105.0',
                           '60DAY_IMPVOL_110.0%MNY_DF': '60d110.0',
                           '60DAY_IMPVOL_120%MNY_DF': '60d120.0'}

IMPVOL_FIELDS_MNY_3MTH = {'3MTH_IMPVOL_80%MNY_DF': '3m80.0',
                          '3MTH_IMPVOL_90.0%MNY_DF': '3m90.0',
                          '3MTH_IMPVOL_95.0%MNY_DF': '3m95.0',
                          '3MTH_IMPVOL_97.5%MNY_DF': '3m97.5',
                          '3MTH_IMPVOL_100.0%MNY_DF': '3m100.0',
                          '3MTH_IMPVOL_102.5%MNY_DF': '3m102.5',
                          '3MTH_IMPVOL_105.0%MNY_DF': '3m105.0',
                          '3MTH_IMPVOL_110.0%MNY_DF': '3m110.0',
                          '3MTH_IMPVOL_120%MNY_DF': '3m120.0'}

IMPVOL_FIELDS_MNY_6MTH = {'6MTH_IMPVOL_80%MNY_DF': '6m80.0',
                          '6MTH_IMPVOL_90.0%MNY_DF': '6m90.0',
                          '6MTH_IMPVOL_95.0%MNY_DF': '6m95.0',
                          '6MTH_IMPVOL_97.5%MNY_DF': '6m97.5',
                          '6MTH_IMPVOL_100.0%MNY_DF': '6m100.0',
                          '6MTH_IMPVOL_102.5%MNY_DF': '6m102.5',
                          '6MTH_IMPVOL_105.0%MNY_DF': '6m105.0',
                          '6MTH_IMPVOL_110.0%MNY_DF': '6m110.0',
                          '6MTH_IMPVOL_120%MNY_DF': '6m120.0'}

IMPVOL_FIELDS_MNY_12M = {'12MTH_IMPVOL_80%MNY_DF': '12m80.0',
                         '12MTH_IMPVOL_90.0%MNY_DF': '12m90.0',
                         '12MTH_IMPVOL_95.0%MNY_DF': '12m95.0',
                         '12MTH_IMPVOL_97.5%MNY_DF': '12m97.5',
                         '12MTH_IMPVOL_100.0%MNY_DF': '12m100.0',
                         '12MTH_IMPVOL_102.5%MNY_DF': '12m102.5',
                         '12MTH_IMPVOL_105.0%MNY_DF': '12m105.0',
                         '12MTH_IMPVOL_110.0%MNY_DF': '12m110.0',
                         '12MTH_IMPVOL_120%MNY_DF': '12m120.0'}

IMPVOL_FIELDS_DELTA = {'1M_CALL_IMP_VOL_10DELTA_DFLT': '1MC10D.0',
                       '1M_CALL_IMP_VOL_25DELTA_DFLT': '1MC25D.0',
                       '1M_CALL_IMP_VOL_40DELTA_DFLT': '1MC40D.0',
                       '1M_CALL_IMP_VOL_50DELTA_DFLT': '1MC50D.0',
                       '1M_PUT_IMP_VOL_50DELTA_DFLT': '1MP50D.0',
                       '1M_PUT_IMP_VOL_40DELTA_DFLT': '1MP40D.0',
                       '1M_PUT_IMP_VOL_25DELTA_DFLT': '1MP25D.0',
                       '1M_PUT_IMP_VOL_10DELTA_DFLT': '1MP10D.0',
                       '2M_CALL_IMP_VOL_10DELTA_DFLT': '2MC10D.0',
                       '2M_CALL_IMP_VOL_25DELTA_DFLT': '2MC25D.0',
                       '2M_CALL_IMP_VOL_40DELTA_DFLT': '2MC40D.0',
                       '2M_CALL_IMP_VOL_50DELTA_DFLT': '2MC50D.0',
                       '2M_PUT_IMP_VOL_50DELTA_DFLT': '2MP50D.0',
                       '2M_PUT_IMP_VOL_40DELTA_DFLT': '2MP40D.0',
                       '2M_PUT_IMP_VOL_25DELTA_DFLT': '2MP25D.0',
                       '2M_PUT_IMP_VOL_10DELTA_DFLT': '2MP10D.0'
                       }



def fetch_fundamentals(tickers: List[str],
                       fields: List[str] = ('Security_Name', 'GICS_Sector_Name',)
                       ) -> pd.DataFrame:
    df = blp.bdp(tickers=tickers, flds=fields)
    df = df.reindex(index=tickers).reindex(columns=fields)
    return df


def fetch_fields_timeseries_per_ticker(ticker: str,
                                       fields: List[str] = ('PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST',),
                                       CshAdjNormal: bool = True,
                                       CshAdjAbnormal: bool = True,
                                       CapChg: bool = True,
                                       start_date: pd.Timestamp = DEFAULT_START_DATE,
                                       end_date: pd.Timestamp = pd.Timestamp.now()
                                       ) -> Optional[pd.DataFrame]:

    try:
        # get bloomberg data adjusted for splits and divs
        field_data = blp.bdh(ticker, fields, start_date, end_date,
                             CshAdjNormal=CshAdjNormal, CshAdjAbnormal=CshAdjAbnormal, CapChg=CapChg)
    except:
        warnings.warn(f"could not get field_data for ticker={ticker}")
        return None

    try:
        field_data.columns = field_data.columns.droplevel(0)  # eliminate multiindex
    except:
        warnings.warn(f"something is wrong for ticker=r={ticker}")
        print(field_data)
        return None

    if len(fields) > 1:
        field_data = field_data.reindex(columns=fields)  # rearrange columns
    else:
        pass

    field_data.index = pd.to_datetime(field_data.index)
    field_data.sort_index()
    return field_data


def fetch_field_timeseries_per_tickers(tickers: List[str],
                                       field: str = 'PX_LAST',
                                       CshAdjNormal: bool = True,
                                       CshAdjAbnormal: bool = True,
                                       CapChg: bool = True,
                                       start_date: pd.Timestamp = DEFAULT_START_DATE,
                                       end_date: pd.Timestamp = pd.Timestamp.now(),
                                       freq: str = None
                                       ) -> Optional[pd.DataFrame]:

    """
    get bloomberg data adjusted for splits and divs
    """
    field_data = blp.bdh(tickers, field, start_date, end_date, CshAdjNormal=CshAdjNormal, CshAdjAbnormal=CshAdjAbnormal, CapChg=CapChg)

    try:
        field_data.columns = field_data.columns.droplevel(1)  # eliminate multiindex
    except:
        warnings.warn(f"something is wrong for field={field}")
        return None

    # make sure all columns are returns
    field_data = field_data.reindex(columns=tickers)
    field_data.index = pd.to_datetime(field_data.index)
    field_data = field_data.sort_index()
    if freq is not None:
        field_data = field_data.asfreq(freq, method='ffill')
    return field_data


def fetch_active_futures(generic_ticker: str = 'ES1 Index',
                         first_gen: int = 1
                         ) -> Tuple[pd.Series, pd.Series]:
    """
    need to run with GFUT settings: roll = None
    bloomberg often fails to get joint data for two adjacent futures
    we need to split the index
    """
    atickers = [instrument_to_active_ticker(generic_ticker, num=first_gen),
                instrument_to_active_ticker(generic_ticker, num=first_gen + 1)]

    start_date = DEFAULT_START_DATE
    end_date = pd.Timestamp.now()
    price_datas = {}
    for aticker in atickers:
        price_data = fetch_fields_timeseries_per_ticker(ticker=aticker, fields=['PX_LAST'],
                                                        CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False,
                                                        start_date=start_date, end_date=end_date)
        if price_data is None or price_data.empty:
            warnings.warn(f"second attempt to fetch data for {aticker}")
            price_data = fetch_fields_timeseries_per_ticker(ticker=aticker, fields=['PX_LAST'],
                                                            CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False,
                                                            start_date=start_date, end_date=end_date)
            if price_data is None or price_data.empty:
                warnings.warn(f"third attempt to fetch data for {aticker}")
                price_data = fetch_fields_timeseries_per_ticker(ticker=aticker, fields=['PX_LAST'],
                                                                CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False,
                                                                start_date=start_date, end_date=end_date)

        price_datas[aticker] = price_data.iloc[:, 0]
        start_date = price_data.index[0]
        end_date = price_data.index[-1]
    price_data = pd.DataFrame.from_dict(price_datas, orient='columns')

    return price_data.iloc[:, 0], price_data.iloc[:, 1]


def instrument_to_active_ticker(instrument: str = 'ES1 Index', num: int = 1) -> str:
    """
    ES1 Index to ES{num} Index
    Z1 Index to Z 1 Index
    """
    head = contract_to_instrument(instrument)
    ticker_split = instrument.split(' ')
    mid = "" if len(ticker_split[0]) > 1 else " "
    active_ticker = f"{head}{mid}{num} {ticker_split[-1]}"
    return active_ticker


def contract_to_instrument(future: str) -> str:
    """
    ES1 Index to ES Index
    """
    ticker_split_wo_num = re.sub('\d+', '', future).split()
    return ticker_split_wo_num[0]


def fetch_futures_contract_table(ticker: str = "ESA Index",
                                 flds: List[str] = ('name',
                                                    'px_settle',
                                                    'px_last',
                                                    'px_bid', 'px_ask', 'bid_size', 'ask_size',
                                                    'volume', 'volume_avg_5d', 'open_int',
                                                    'fut_cont_size',
                                                    'contract_value',
                                                    'fut_val_pt',
                                                    'quoted_crncy',
                                                    'fut_days_expire',
                                                    'px_settle_last_dt',
                                                    'last_tradeable_dt',
                                                    'last_update_dt',
                                                    'last_update'),
                                 add_timestamp: bool = True,
                                 add_gen_number: bool = True,
                                 add_carry: bool = True
                                 ) -> pd.DataFrame:
    contracts = blp.bds(ticker, "FUT_CHAIN")
    if contracts.empty:
        contracts = blp.bds(ticker, "FUT_CHAIN")
    if not contracts.empty:
        tickers = contracts['security_description']
        df = blp.bdp(tickers=tickers, flds=flds)
        tradable_tickers = tickers[np.isin(tickers, df.index, assume_unique=True)]
        good_columns = pd.Index(flds)[np.isin(flds, df.columns, assume_unique=True)]
        df = df.loc[tradable_tickers, good_columns]

        if add_timestamp:
            timestamps = df['last_update_dt'].copy()
            # last_update can be date.time
            for idx, (x, y) in enumerate(zip(df['last_update_dt'], df['last_update'])):
                if isinstance(y, datetime.time):
                    timestamps.iloc[idx] = pd.Timestamp.combine(x, y).tz_localize(tz='CET').tz_convert('UTC')
                elif isinstance(x, datetime.date):
                    timestamps.iloc[idx] = pd.Timestamp.combine(x, datetime.time(0,0,0)).tz_localize('UTC')
            df['update'] = timestamps
            df['timestamp'] = pd.Timestamp.utcnow()
            df = df.drop(['last_update_dt', 'last_update'], axis=1)

        if add_gen_number:
            df['gen_number'] = [n+1 for n in range(len(df.index))]

        if add_carry and len(df.index) > 1:
            n = len(df.index)
            carry = np.full(n, np.nan)
            bid_ask = df[['px_bid', 'px_ask']].to_numpy()
            is_good = np.logical_and(pd.isna(bid_ask[:, 0])==False, pd.isna(bid_ask[:, 1])==False)
            mid_price = np.where(is_good, 0.5*(bid_ask[:, 0]+bid_ask[:, 1]), np.nan)
            an_days_to_mat = df['fut_days_expire'].to_numpy() / 365.0
            for idx in range(n):
                if idx > 0:
                    carry[idx] = - (mid_price[idx] - mid_price[idx-1]) / mid_price[idx-1] / (an_days_to_mat[idx]-an_days_to_mat[idx-1])
            df['an_carry'] = carry
    else:
        print(f"no data for {ticker}")
        df = pd.DataFrame()
    df['ticker'] = ticker
    return df


def fetch_vol_timeseries(ticker: str = 'SPX Index',
                         vol_fields: Union[Dict, List] = IMPVOL_FIELDS_DELTA,
                         start_date: pd.Timestamp = VOLS_START_DATE,
                         rate_index: str = 'usgg3m Index',
                         add_underlying: bool = True,
                         rename: bool = True
                         ) -> pd.DataFrame:

    if isinstance(vol_fields, list):
        dfs = []
        for fields_ in vol_fields:
            df = fetch_fields_timeseries_per_ticker(ticker=ticker,
                                                    fields=list(fields_.keys()),
                                                    start_date=start_date)
            if rename:
                df = df.rename(fields_, axis=1)
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
    else:
        df = fetch_fields_timeseries_per_ticker(ticker=ticker,
                                                fields=list(vol_fields.keys()),
                                                start_date=start_date)
        if rename:
            df = df.rename(vol_fields, axis=1)
    df = 0.01*df

    if add_underlying:
        price = fetch_fields_timeseries_per_ticker(ticker=ticker,
                                                   fields=['PX_LAST', 'EQY_DVD_YLD_12M'],
                                                   start_date=start_date)
        price['EQY_DVD_YLD_12M'] *= 0.01
        price = price.rename({'PX_LAST': 'spot_price', 'EQY_DVD_YLD_12M': 'div_yield'}, axis=1)
        rate_3m = 0.01*fetch_fields_timeseries_per_ticker(ticker=rate_index,
                                                          fields=['PX_LAST'],
                                                          start_date=start_date)
        rate_3m = rate_3m.rename({'PX_LAST': 'rf_rate'}, axis=1)
        # drop row when vols are missing
        df = pd.concat([price, rate_3m, df], axis=1)#.dropna(axis=0, subset=df.columns, how='all')
    return df


def fetch_last_prices(tickers: Union[List, Dict] = FX_DICT) -> pd.Series:
    if isinstance(tickers, Dict):
        tickers1 = list(tickers.keys())
    else:
        tickers1 = tickers
    df = blp.bdp(tickers=tickers1, flds='px_last')
    if isinstance(tickers, Dict):
        df = df.rename(tickers, axis=0)
    return df.iloc[:, 0]


def fetch_bond_info(isins: List[str] = ['US03522AAJ97', 'US126650CZ11'],
                    fields: List[str] = ['id_bb', 'name',  'security_des',
                                         'ult_parent_ticker_exchange', 'crncy', 'amt_outstanding',
                                         'px_last',
                                         'yas_bond_yld', 'yas_oas_sprd', 'yas_mod_dur']
                    ) -> pd.DataFrame:
    issue_data = blp.bdp([f"{isin} corp" for isin in isins], fields)
    # process US03522AAH32 corp to US03522AAH32
    issue_data.insert(loc=0, column='isin', value=[x.split(' ')[0] for x in issue_data.index])
    issue_data = issue_data.reset_index(names='isin_corp').set_index('isin')
    issue_data = issue_data.reindex(index=isins)
    return issue_data


def fetch_cds_info(equity_tickers: List[str] = ['ABI BB Equity', 'CVS US Equity']) -> pd.DataFrame:
    cds_rate_tickers = blp.bdp(tickers=equity_tickers, flds='cds_spread_ticker_5y')
    cds_rate_tickers = cds_rate_tickers.reindex(index=equity_tickers)
    return cds_rate_tickers


def fetch_balance_data(tickers: List[str] = ['ABI BB Equity', 'T US Equity', 'JPM US Equity'],
                       fields: List[str] = ['GICS_SECTOR_NAME', 'BB_ISSR_COMP_BSE_ON_RTGS', 'TOT_COMMON_EQY',
                                            'BS_LT_BORROW', 'BS_ST_BORROW', 'EQY_FUND_CRNCY',
                                            'EARN_YLD',
                                            'RETURN_ON_ASSETS_ADJUSTED',
                                            'NET_DEBT_TO_FFCF',
                                            'NET_DEBT_TO_CASHFLOW',
                                            'FREE_CASH_FLOW_MARGIN',
                                            'CFO_TO_SALES',
                                            'NET_DEBT_PCT_OF_TOT_CAPITAL',
                                            'INTEREST_COVERAGE_RATIO',
                                            'BS_LIQUIDITY_COVERAGE_RATIO',
                                            'NET_DEBT_TO_EBITDA',
                                            'T12_FCF_T12_EBITDA']
                       ) -> pd.DataFrame:
    issue_data = blp.bdp(tickers, fields)
    issue_data = issue_data.rename({x: x.upper() for x in issue_data.columns}, axis=1)
    issue_data = issue_data.reindex(index=tickers).reindex(columns=fields)

    return issue_data


def fetch_tickers_from_isins(isins: List[str] = ['US88160R1014', 'IL0065100930']) -> pd.DataFrame:
    tickers = {f"/ISIN/{x}": x for x in isins}
    df = blp.bdp(list(tickers.keys()), "PARSEKYABLE_DES")
    df.index = df.index.map(tickers)  # map back to isins
    df = df.reindex(index=isins)
    return df


def fetch_dividend_history(ticker: str = 'TIP US Equity') -> pd.DataFrame:
    """
    df.columns = ['declared_date', 'ex_date', 'record_date', 'payable_date',
       'dividend_amount', 'dividend_frequency', 'dividend_type']
    """
    this = blp.bds(ticker, 'dvd_hist_all')
    return this


def fetch_div_yields(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    divs = {}
    divs_1y = {}
    for ticker in tickers:
        div = fetch_dividend_history(ticker=ticker)
        if not div.empty:
            valid_div = div.loc[div['dividend_type'] == 'Income', :].set_index('ex_date')  # set ex_date index
            if not div.empty and len(valid_div.index) > 0:
                valid_div.index = pd.to_datetime(valid_div.index)
                valid_div = valid_div.sort_index()
                div_freq = valid_div['dividend_frequency'].iloc[-1]
                if div_freq == 'Monthly': # extrapolate to 1y
                    roll_period = 12
                    an_factor = 1.0
                elif div_freq == 'Quarter': # extrapolate to 1y
                    roll_period = 4
                    an_factor = 1.0
                elif div_freq == 'Annual': # extrapolate to 1y
                    roll_period = 1
                    an_factor = 1.0
                else:
                    raise NotImplementedError(f"div_freq = {div_freq}")
                divs[ticker] = valid_div['dividend_amount']
                divs_1y[ticker] = an_factor * valid_div['dividend_amount'].rolling(roll_period).sum()
    divs = pd.DataFrame.from_dict(divs, orient='columns').reindex(columns=tickers)
    divs_1y = pd.DataFrame.from_dict(divs_1y, orient='columns').reindex(columns=tickers)
    return divs, divs_1y


"""
def fetch_option_underlying_tickers_from_isins(isins: List[str] = ['DE000C77PRU9', 'YY0160552733']) -> pd.DataFrame:
    tickers = {f"/cusip/{x} Corp": x for x in isins}
    # tickers = {f"{x}@BGN Corp": x for x in isins}
    df = blp.bdp(list(tickers.keys()), "PARSEKYABLE_DES")
    print(df)
    df = blp.bdp(list(tickers.keys()), "OPT_UNDL_TICKER")
    print(df)
    df.index = df.index.map(tickers)  # map back to isins
    df = df.reindex(index=isins)
    return df
"""


class UnitTests(Enum):
    FUNDAMENTALS = 1
    FIELDS_PER_TICKER = 2
    FIELD_PER_TICKERS = 3
    ACTIVE_FUTURES = 4
    CONTRACT_TABLE = 5
    IMPLIED_VOL_TIME_SERIES = 6
    LAST_PRICES = 7
    CDS = 8
    BOND_INFO = 9
    CDS_INFO = 10
    BALANCE_DATA = 11
    TICKERS_FROM_ISIN = 12
    # OPTION_UNDERLYING_FROM_ISIN = 14
    DIVIDEND = 14


def run_unit_test(unit_test: UnitTests):

    pd.set_option('display.max_columns', 500)

    if unit_test == UnitTests.FUNDAMENTALS:
        df = fetch_fundamentals(tickers=['AAPL US Equity', 'BAC US Equity'],
                                fields=['Security_Name', 'GICS_Sector_Name', 'CRNCY'])
        print(df)

    elif unit_test == UnitTests.FIELDS_PER_TICKER:
        # df = fetch_fields_timeseries_per_ticker(ticker='USDJPYV1M BGN Curncy', fields=['PX_LAST'])
        # df = fetch_fields_timeseries_per_ticker(ticker='SPY US Equity', fields=['30DAY_IMPVOL_100.0%MNY_DF', 'PX_LAST'])
        # df = fetch_fields_timeseries_per_ticker(ticker='USDJPY Curncy', fields=['30DAY_IMPVOL_100.0%MNY_DF', 'PX_LAST'])
        df = fetch_fields_timeseries_per_ticker(ticker='ES1 Index', fields=['PX_LAST', 'FUT_DAYS_EXP'])
        print(df)

    elif unit_test == UnitTests.FIELD_PER_TICKERS:
        # df = fetch_field_timeseries_per_tickers(tickers=['AAPL US Equity', 'OCBC SP Equity', '6920 JP Equity'], field='30DAY_IMPVOL_100.0%MNY_DF')
        # df = fetch_field_timeseries_per_tickers(tickers=['ES1 Index', 'ES2 Index', 'ES3 Index'], field='PX_LAST',
        #                                         CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False)
        #df = fetch_field_timeseries_per_tickers(tickers=['CSBC1U5 PRXY Curncy'], field='PX_LAST')
        #print(df)
        # PRXY CBGN
        # df = fetch_field_timeseries_per_tickers(tickers=['CGIS1U5 CBGN Curncy'], field='PX_LAST')
        df = fetch_field_timeseries_per_tickers(tickers=['CSBC1U5 CBGN Curncy'], field='PX_LAST')
        print(df)

    elif unit_test == UnitTests.ACTIVE_FUTURES:
        # field_data = blp.active_futures('ESA Index', dt='1997-09-10')
        # print(field_data)

        field_data = fetch_active_futures(generic_ticker='ESA Equity')
        print(field_data)

    elif unit_test == UnitTests.CONTRACT_TABLE:
        df = fetch_futures_contract_table(ticker="NK1 Index")
        print(df)

    elif unit_test == UnitTests.IMPLIED_VOL_TIME_SERIES:
        df = fetch_vol_timeseries(ticker='SPX Index', vol_fields=[IMPVOL_FIELDS_MNY_30DAY, IMPVOL_FIELDS_MNY_60DAY,
                                                                  IMPVOL_FIELDS_MNY_3MTH, IMPVOL_FIELDS_MNY_6MTH,
                                                                  IMPVOL_FIELDS_MNY_12M])
        print(df)

    elif unit_test == UnitTests.LAST_PRICES:
        fx_prices = fetch_last_prices()
        print(fx_prices)

    elif unit_test == UnitTests.CDS:
        df = fetch_field_timeseries_per_tickers(
            tickers=['CGS1U5 CBGN Curncy', 'CGS1U5 DRSK Curncy', 'CGS1U5 BEST Curncy'], field='PX_LAST')
        print(df)

    elif unit_test == UnitTests.BOND_INFO:
        data = fetch_bond_info()
        print(data)

    elif unit_test == UnitTests.CDS_INFO:
        data = fetch_cds_info()
        print(data)

    elif unit_test == UnitTests.BALANCE_DATA:
        data = fetch_balance_data(tickers=['ABI BB Equity', 'T US Equity', 'JPM US Equity', 'BAC US Equity'])
        print(data)

    elif unit_test == UnitTests.TICKERS_FROM_ISIN:
        df = fetch_tickers_from_isins()
        print(df)

    elif unit_test == UnitTests.DIVIDEND:
        this = fetch_dividend_history(ticker='ERNA LN Equity')
        print(this)
    """
    elif unit_test == UnitTests.OPTION_UNDERLYING_FROM_ISIN:
        df = fetch_option_underlying_tickers_from_isins()
        print(df)
    """


if __name__ == '__main__':

    unit_test = UnitTests.DIVIDEND

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
