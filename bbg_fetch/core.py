"""
Bloomberg data fetching utilities.

Provides high-level functions for retrieving equity, futures, fixed-income,
FX, options, and index data from Bloomberg via blpapi.

Install blpapi:
    pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
"""

import re
import warnings
import datetime
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union

from bbg_fetch._blp_api import bdp, bdh, bds

DEFAULT_START_DATE = pd.Timestamp('01Jan1959')
VOLS_START_DATE = pd.Timestamp('03Jan2005')

DEFAULT_TENOR_YEARS = {
    '30d': 30 / 365,
    '60d': 60 / 365,
    '3m': 0.25,
    '6m': 0.5,
    '12m': 1.0,
}

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


def fetch_field_timeseries_per_tickers(tickers: Union[List[str], Tuple[str], Dict[str, str]],
                                       field: str = 'PX_LAST',
                                       CshAdjNormal: bool = True,
                                       CshAdjAbnormal: bool = True,
                                       CapChg: bool = True,
                                       start_date: Optional[pd.Timestamp] = DEFAULT_START_DATE,
                                       end_date: Optional[pd.Timestamp] = pd.Timestamp.now(),
                                       freq: str = None
                                       ) -> Optional[pd.DataFrame]:
    """
    get bloomberg field data adjusted for splits and divs for a list of tickers
    tickers can be a dict {'ES1 Index': 'SPY', 'UXY1 Comdty': '10yUST'}, then df columns are renamed
    """
    if isinstance(tickers, list) or isinstance(tickers, Tuple):
        tickers_ = tickers
    elif isinstance(tickers, dict):
        tickers_ = list(tickers.keys())
    else:
        raise NotImplementedError(f"type={type(tickers)}")
    field_data = bdh(tickers_, field, start_date, end_date, CshAdjNormal=CshAdjNormal,
                     CshAdjAbnormal=CshAdjAbnormal, CapChg=CapChg)

    try:
        field_data.columns = field_data.columns.droplevel(1)  # eliminate multiindex
    except:
        warnings.warn(f"something is wrong for field={field}")
        return None

    # make sure all columns are returned
    field_data.index = pd.to_datetime(field_data.index)
    if freq is not None:
        field_data = field_data.asfreq(freq).ffill()

    # align columns
    field_data = field_data.reindex(columns=tickers_)
    if isinstance(tickers, dict):
        field_data = field_data.rename(tickers, axis=1)

    field_data = field_data.sort_index()

    return field_data


def fetch_fields_timeseries_per_ticker(ticker: str,
                                       fields: List[str] = ('PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST',),
                                       CshAdjNormal: bool = True,
                                       CshAdjAbnormal: bool = True,
                                       CapChg: bool = True,
                                       start_date: pd.Timestamp = DEFAULT_START_DATE,
                                       end_date: pd.Timestamp = pd.Timestamp.now()
                                       ) -> Optional[pd.DataFrame]:
    """
    get bloomberg fields data adjusted for splits and divs for given ticker
    """
    try:
        # get bloomberg data adjusted for splits and divs
        field_data = bdh(ticker, fields, start_date, end_date,
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


def fetch_fundamentals(tickers: Union[List[str], Dict[str, str]],
                       fields: Union[List[str], Dict[str, str]] = ('security_name', 'gics_sector_name',)
                       ) -> pd.DataFrame:
    if isinstance(tickers, list):
        tickers_ = tickers
    elif isinstance(tickers, dict):
        tickers_ = list(tickers.keys())
    else:
        raise NotImplementedError(f"tickers type={type(tickers)}")
    if isinstance(fields, list):
        fields_ = fields
    elif isinstance(fields, dict):
        fields_ = list(fields.keys())
    else:
        raise NotImplementedError(f"fields type={type(fields)}")
    df = bdp(tickers=tickers_, flds=fields_)
    # align with given order of tickers
    df = df.reindex(index=tickers_).reindex(columns=fields)
    if isinstance(tickers, dict):
        df = df.rename(tickers, axis=0)
    if isinstance(fields, dict):
        df = df.rename(fields, axis=1)
    return df


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
                                 add_carry: bool = True,
                                 tz: Optional[str] = 'UTC'
                                 ) -> pd.DataFrame:
    """
    fetch contract table for active futures
    """
    contracts = bds(ticker, "FUT_CHAIN")
    if contracts.empty:
        contracts = bds(ticker, "FUT_CHAIN")
    if not contracts.empty:
        tickers = contracts['security_description']
        df = bdp(tickers=tickers, flds=flds)
        tradable_tickers = tickers[np.isin(tickers, df.index, assume_unique=True)]
        good_columns = pd.Index(flds)[np.isin(flds, df.columns, assume_unique=True)]
        df = df.loc[tradable_tickers, good_columns]

        if add_timestamp:
            timestamps = df['last_update_dt'].copy()
            # last_update can be date.time
            for idx, (x, y) in enumerate(zip(df['last_update_dt'], df['last_update'])):
                if isinstance(y, datetime.time):
                    timestamps.iloc[idx] = pd.Timestamp.combine(x, y).tz_localize(tz='CET').tz_convert(tz)
                elif isinstance(x, datetime.date):
                    timestamps.iloc[idx] = pd.Timestamp.combine(x, datetime.time(0,0,0)).tz_localize(tz)
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
                         add_forwards: bool = False,
                         tenor_years: Optional[Dict[str, float]] = None,
                         rename: bool = True,
                         scaler: Optional[float] = 0.01
                         ) -> pd.DataFrame:
    """
    Fetch implied vol time series with optional underlying data and forward prices.

    Supports three vol_fields input modes:
    - Single dict: one tenor, e.g. IMPVOL_FIELDS_DELTA
    - List of dicts: multi-tenor surface, e.g. [IMPVOL_FIELDS_MNY_30DAY, ..., IMPVOL_FIELDS_MNY_12M]
      Each dict is fetched as a separate Bloomberg call (avoids field-count limits).
    - List of strings: raw Bloomberg field names, no renaming applied.

    Parameters
    ----------
    ticker : str
        Bloomberg ticker, e.g. 'SPX Index', 'EURUSD Curncy'.
    vol_fields : dict or list
        Implied vol field specification. Dict keys are Bloomberg field names,
        values are short labels for column renaming.
    start_date : pd.Timestamp
        Start date for historical data.
    rate_index : str
        Bloomberg ticker for risk-free rate, e.g. 'usgg3m Index'.
    add_underlying : bool
        If True, prepend spot_price, div_yield (trailing 12M), and rf_rate columns.
    add_forwards : bool
        If True, compute per-tenor implied forwards and discount factors.
        Adds columns fwd_{tenor} = S * exp((r - q) * T) and df_{tenor} = exp(-r * T).
        Requires add_underlying=True.
    tenor_years : dict, optional
        Map of tenor label → year fraction for forward/discount factor computation.
        Default: {'30d': 30/365, '60d': 60/365, '3m': 0.25, '6m': 0.5, '12m': 1.0}
    rename : bool
        If True, rename vol columns using dict values. Ignored for list-of-strings input.
    scaler : float, optional
        Multiply vol and rate values by this factor. Default 0.01 converts
        Bloomberg's percentage values (e.g. 20.5) to decimals (0.205).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex. Columns depend on options:
        - Vol columns: renamed labels (e.g. '30d100.0') or raw Bloomberg field names
        - If add_underlying: spot_price, div_yield, rf_rate
        - If add_forwards: fwd_30d, fwd_60d, ..., df_30d, df_60d, ...
    """
    # handle list of dicts: fetch each tenor separately and concatenate
    if isinstance(vol_fields, list) and len(vol_fields) > 0 and isinstance(vol_fields[0], dict):
        frames = []
        for vf in vol_fields:
            df_part = fetch_fields_timeseries_per_ticker(
                ticker=ticker, fields=list(vf.keys()), start_date=start_date)
            if df_part is not None and rename:
                df_part = df_part.rename(vf, axis=1)
            if df_part is not None:
                frames.append(df_part)
        df = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    elif isinstance(vol_fields, list):
        df = fetch_fields_timeseries_per_ticker(ticker=ticker,
                                                fields=vol_fields,
                                                start_date=start_date)
    else:
        df = fetch_fields_timeseries_per_ticker(ticker=ticker,
                                                fields=list(vol_fields.keys()),
                                                start_date=start_date)
        if rename:
            df = df.rename(vol_fields, axis=1)
    if scaler is not None:
        df *= scaler

    if add_underlying:
        price = fetch_fields_timeseries_per_ticker(ticker=ticker,
                                                   fields=['PX_LAST', 'EQY_DVD_YLD_12M'],
                                                   start_date=start_date)
        if scaler is not None:
            price['EQY_DVD_YLD_12M'] *= scaler
        price = price.rename({'PX_LAST': 'spot_price', 'EQY_DVD_YLD_12M': 'div_yield'}, axis=1)
        rate_3m = fetch_fields_timeseries_per_ticker(ticker=rate_index,
                                                     fields=['PX_LAST'],
                                                     start_date=start_date)
        if scaler is not None:
            rate_3m *= scaler
        rate_3m = rate_3m.rename({'PX_LAST': 'rf_rate'}, axis=1)
        df = pd.concat([price, rate_3m, df], axis=1)

        if add_forwards:
            if tenor_years is None:
                tenor_years = DEFAULT_TENOR_YEARS
            r = df['rf_rate']
            q = df['div_yield']
            s = df['spot_price']
            for label, T in tenor_years.items():
                df[f'fwd_{label}'] = s * np.exp((r - q) * T)
                df[f'df_{label}'] = np.exp(-r * T)

    return df


def fetch_last_prices(tickers: Union[List, Dict] = FX_DICT) -> pd.Series:
    """
    fetch last prices of instruments in tickers
    """
    if isinstance(tickers, Dict):
        tickers1 = list(tickers.keys())
    else:
        tickers1 = tickers
    df = bdp(tickers=tickers1, flds='px_last')
    if isinstance(tickers, Dict):
        df = df.rename(tickers, axis=0)
    return df.iloc[:, 0]


def fetch_bonds_info(isins: List[str] = ['US03522AAJ97', 'US126650CZ11'],
                     fields: List[str] = ('id_bb', 'name',  'security_des',
                                          'ult_parent_ticker_exchange', 'crncy', 'amt_outstanding',
                                          'px_last',
                                          'yas_bond_yld', 'yas_oas_sprd', 'yas_mod_dur'),
                     END_DATE_OVERRIDE: Optional[str] = None
                     ) -> pd.DataFrame:
    """
    bonds are given by isins
    fetch fileds data for bonds
    """
    if END_DATE_OVERRIDE is None:
        issue_data = bdp([f"{isin} corp" for isin in isins], fields)
    else:
        issue_data = bdp([f"{isin} corp" for isin in isins], fields, END_DATE_OVERRIDE=END_DATE_OVERRIDE)

    # process US03522AAH32 corp to US03522AAH32
    issue_data.insert(loc=0, column='isin', value=[x.split(' ')[0] for x in issue_data.index])
    issue_data = issue_data.reset_index(names='isin_corp').set_index('isin')
    issue_data = issue_data.reindex(index=isins)
    return issue_data


def fetch_cds_info(equity_tickers: List[str] = ('ABI BB Equity', 'CVS US Equity'),
                   field: str = 'cds_spread_ticker_5y'
                   ) -> pd.DataFrame:
    """
    fetch cds info
    """
    cds_rate_tickers = bdp(tickers=equity_tickers, flds=field)
    cds_rate_tickers = cds_rate_tickers.reindex(index=equity_tickers)
    return cds_rate_tickers


def fetch_balance_data(tickers: List[str] = ('ABI BB Equity', 'T US Equity', 'JPM US Equity'),
                       fields: List[str] = ('GICS_SECTOR_NAME', 'BB_ISSR_COMP_BSE_ON_RTGS', 'TOT_COMMON_EQY',
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
                                            'T12_FCF_T12_EBITDA')
                       ) -> pd.DataFrame:
    """
    fundamentals data for tickers in tickers
    """
    issue_data = bdp(tickers, fields)
    issue_data = issue_data.rename({x: x.upper() for x in issue_data.columns}, axis=1)
    issue_data = issue_data.reindex(index=tickers).reindex(columns=fields)

    return issue_data


def fetch_tickers_from_isins(isins: List[str] = ['US88160R1014', 'IL0065100930']) -> List[str]:
    """
    =BDP("US4592001014 ISIN", "PARSEKYABLE_DES") => IBM XX Equity
    where XX depends on your terminal settings, which you can check on CNDF <Go>.
    get the main exchange composite ticker, or whatever suits your need (in A3):
    =BDP(A2,"EQY_PRIM_SECURITY_COMP_EXCH") => US
    """
    tickers = {f"/ISIN/{x}": x for x in isins}
    df = bdp(list(tickers.keys()), ["parsekyable_des", "eqy_prim_security_comp_exch"])
    df.index = df.index.map(tickers)  # map back to isins  need to sort back to isins order
    df = df.reindex(index=isins)
    # replace default country with exchange
    tickers = []
    for ticker_, exchange in zip(df["parsekyable_des"].to_list(), df["eqy_prim_security_comp_exch"].to_list()):
        ticker_s = ticker_.split(' ')
        tickers.append(f"{ticker_s[0]} {exchange} {ticker_s[-1]}")
    return tickers


def fetch_dividend_history(ticker: str = 'TIP US Equity') -> pd.DataFrame:
    """
    df.columns = ['declared_date', 'ex_date', 'record_date', 'payable_date',
       'dividend_amount', 'dividend_frequency', 'dividend_type']
    """
    this = bds(ticker, 'dvd_hist_all')
    return this


def fetch_div_yields(tickers: Union[List[str], Dict[str, str]],
                     dividend_types: List[str] = ('Income', 'Distribution')
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    dividend_types can include:
    dividend_types: List[str] = ('Income', 'Distribution')
    dividend_types: List[str] = ('Income', 'Distribution', 'Return of Capital', 'Accumulation')
    """
    if isinstance(tickers, list):
        tickers_ = tickers
    elif isinstance(tickers, dict):
        tickers_ = list(tickers.keys())
    else:
        raise NotImplementedError(f"type={type(tickers)}")
    divs = {}
    divs_1y = {}
    for ticker in tickers_:
        div = fetch_dividend_history(ticker=ticker)
        if not div.empty:
            valid_div_cond = div['dividend_type'].apply(lambda x: x in dividend_types)
            valid_div = div.loc[valid_div_cond, :].set_index('ex_date')  # set ex_date index
            if np.any(valid_div.index.duplicated()): # aggregate dividend by sum of non-unique distributions
                def sum_unique(s):
                    return s.unique().sum()
                valid_div = valid_div.groupby('declared_date', sort=False, as_index=True).agg(
                    declared_date=('declared_date', 'first'),
                    record_date=('record_date', 'first'),
                    payable_date=('payable_date', 'first'),
                    dividend_amount=('dividend_amount', sum_unique),
                    dividend_frequency=('dividend_frequency', 'first'),
                    dividend_type=('dividend_type', 'first')
                )

            if not valid_div.empty and len(valid_div.index) > 0:
                valid_div.index = pd.to_datetime(valid_div.index)
                valid_div = valid_div.sort_index()
                valid_div_amount = valid_div['dividend_amount']
                divs[ticker] = valid_div_amount
                divs_1y[ticker] = valid_div_amount.rolling("365D").sum()  # assume 365 B days in year
    divs = pd.DataFrame.from_dict(divs, orient='columns').reindex(columns=tickers)
    divs_1y = pd.DataFrame.from_dict(divs_1y, orient='columns').reindex(columns=tickers)
    if isinstance(tickers, dict):
        divs = divs.rename(tickers, axis=1)
        divs_1y = divs_1y.rename(tickers, axis=1)

    return divs, divs_1y


def fetch_index_members_weights(index: str = 'SPCPGN Index',
                                field: str = 'INDX_MWEIGHT',
                                END_DATE_OVERRIDE: Optional[str] = None
                                ) -> pd.DataFrame:
    if END_DATE_OVERRIDE is None:
        members = bds(index, field)
    else:
        members = bds(index, field, END_DATE_OVERRIDE=END_DATE_OVERRIDE)
    if members is not None and not members.empty:
        members = members.set_index(members.columns[0], drop=True)
    else:
        raise ValueError(f"no data for {index} / {field}")
    return members


####################  Helper functions ####################


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
    ticker_split_wo_num = re.sub(r'\d+', '', future).split()
    return ticker_split_wo_num[0]


def fetch_issuer_isins_from_bond_isins(bond_isins: List[str] = ['XS3034073836', 'USY0616GAA14', 'XS3023923314', 'XXXX']
                                       ) -> pd.Series:
    """
    for given bond isins find their issuer isins
    for bond_isins = ['XS3034073836', 'USY0616GAA14', 'XS3023923314', 'XXXX'], output is
    pd.Series(['XS3034073836', 'USY0616GAA14', 'XS3023923314', np.nan], index=['XS3034073836', 'USY0616GAA14', 'XS3023923314', 'XXXX'])
    """
    # add corp to bond isin
    bonds_isins_map = {f"{x} corp": x for x in bond_isins}
    bonds_issuer_tickers = bdp(bonds_isins_map.keys(), ["ult_parent_ticker_exchange"])
    # rename index to bond isin
    bonds_issuer_tickers = bonds_issuer_tickers.rename(bonds_isins_map, axis=0)
    # add equity to ticker
    equity_tickers_map = {f"{x} Equity": x for x in bonds_issuer_tickers['ult_parent_ticker_exchange'].to_list()}
    issuer_isins = bdp(equity_tickers_map.keys(), ["id_isin"])
    # drop equity from ticker
    issuer_isins = issuer_isins.rename(equity_tickers_map, axis=0).iloc[:, 0].to_dict()
    # map equity isin to equity ticker
    bonds_issuer_tickers = bonds_issuer_tickers["ult_parent_ticker_exchange"].map(issuer_isins)
    # align with input index
    bonds_issuer_tickers = bonds_issuer_tickers.reindex(index=bond_isins).rename('issuer isin')
    return bonds_issuer_tickers