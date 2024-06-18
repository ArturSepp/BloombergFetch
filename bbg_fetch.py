"""
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
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
    df = df.loc[tickers, :]
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
        try:
            field_data = field_data[fields]  # rearrange columns
        except:
            # incomplete field data
            warnings.warn(f"could not get field_data for ticker={ticker}")
            return None
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
                                       end_date: pd.Timestamp = pd.Timestamp.now()
                                       ) -> Optional[pd.DataFrame]:

    #try:
        # get bloomberg data adjusted for splits and divs
    field_data = blp.bdh(tickers, field, start_date, end_date, CshAdjNormal=CshAdjNormal, CshAdjAbnormal=CshAdjAbnormal, CapChg=CapChg)
    #field_data = blp.bdh(tickers, field, start_date, end_date)
    # field_data = blp.bdp(tickers, field)
    #except:
    #   warnings.warn(f"could not get field_data for field={field}")
    #    return None

    try:
        field_data.columns = field_data.columns.droplevel(1)  # eliminate multiindex
    except:
        warnings.warn(f"something is wrong for field={field}")
        print(field_data)
        return None

    # make sure all columns are returns
    field_data = field_data.reindex(columns=tickers)
    field_data.index = pd.to_datetime(field_data.index)
    field_data = field_data.sort_index()
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
        tradable_tickers = tickers[np.in1d(tickers, df.index, assume_unique=True)]
        good_columns = pd.Index(flds)[np.in1d(flds, df.columns, assume_unique=True)]
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
        df = pd.concat([price, rate_3m, df], axis=1).dropna(axis=0, subset=df.columns, how='all')
    return df


class UnitTests(Enum):
    FUNDAMENTALS = 1
    FIELDS_PER_TICKER = 2
    FIELD_PER_TICKERS = 3
    ACTIVE_FUTURES = 4
    CONTRACT_TABLE = 5
    IMPLIED_VOL_TIME_SERIES = 6


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
        df = fetch_field_timeseries_per_tickers(tickers=['ES1 Index', 'ES2 Index', 'ES3 Index'], field='PX_LAST',
                                                CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False)
        print(df)

        # this = blp.bds("ESA Index", "FUT_CHAIN")
        # this = blp.bds("SPY US Equity", "opt_chain", single_date_override="20191010")
        # print(this)

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


if __name__ == '__main__':

    unit_test = UnitTests.IMPLIED_VOL_TIME_SERIES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
