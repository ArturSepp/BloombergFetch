"""
Local integration tests for bbg_fetch.

These are development/debugging tests that download real data from a Bloomberg
terminal. Run from a machine with an active Bloomberg session.

Usage:
    from bbg_fetch.local_tests import run_local_test, LocalTests
    run_local_test(LocalTests.FIELD_TIMESERIES_PER_TICKERS)
"""

from datetime import date
from enum import Enum

import numpy as np
import pandas as pd

from bbg_fetch._blp_api import bdp, bds
from bbg_fetch.core import (
    fetch_field_timeseries_per_tickers,
    fetch_fields_timeseries_per_ticker,
    fetch_fundamentals,
    fetch_active_futures,
    fetch_futures_contract_table,
    fetch_vol_timeseries,
    fetch_last_prices,
    fetch_bonds_info,
    fetch_cds_info,
    fetch_balance_data,
    fetch_tickers_from_isins,
    fetch_dividend_history,
    fetch_div_yields,
    fetch_index_members_weights,
    fetch_issuer_isins_from_bond_isins,
    IMPVOL_FIELDS_MNY_30DAY,
    IMPVOL_FIELDS_MNY_60DAY,
    IMPVOL_FIELDS_MNY_3MTH,
    IMPVOL_FIELDS_MNY_6MTH,
    IMPVOL_FIELDS_MNY_12M,
)


class LocalTests(Enum):
    FIELD_TIMESERIES_PER_TICKERS = 1
    FIELDS_TIMESERIES_PER_TICKER = 2
    FUNDAMENTALS = 3
    ACTIVE_FUTURES = 4
    CONTRACT_TABLE = 5
    IMPLIED_VOL_TIME_SERIES = 6
    BOND_INFO = 7
    LAST_PRICES = 8
    CDS_INFO = 9
    BALANCE_DATA = 10
    TICKERS_FROM_ISIN = 11
    DIVIDEND = 12
    BOND_MEMBERS = 14
    INDEX_MEMBERS = 15
    OPTION_CHAIN = 16
    YIELD_CURVE = 17
    CHECK = 18
    MEMBERS = 19


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    if local_test == LocalTests.FIELD_TIMESERIES_PER_TICKERS:
        #df = fetch_field_timeseries_per_tickers(tickers=['ES1 Index', 'ES2 Index', 'ES3 Index'], field='PX_LAST',
        #                                        CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False)
        # df = fetch_field_timeseries_per_tickers(tickers=['CGS1U5 CBGN Curncy', 'CGS1U5 DRSK Curncy', 'CGS1U5 BEST Curncy'], field='PX_LAST')
        # df = fetch_field_timeseries_per_tickers(tickers=['EUR003M Index'], field='PX_LAST')
        # df = fetch_field_timeseries_per_tickers(tickers=['TY1 Comdty'], field='FUT_EQV_DUR_NOTL')
        # df = fetch_field_timeseries_per_tickers(tickers=['TY1 Comdty', 'UXY1 Comdty'], start_date=pd.Timestamp('01Jan2015'), field='FUT_EQV_DUR_NOTL')
        # df = fetch_field_timeseries_per_tickers(tickers=['ZS877681 corp'], field='PX_LAST')
        df = fetch_field_timeseries_per_tickers(tickers=['GTDEM3Y Govt'], field='PX_LAST')

        print(df)

    elif local_test == LocalTests.FIELDS_TIMESERIES_PER_TICKER:
        df = fetch_fields_timeseries_per_ticker(ticker='ES1 Index', fields=['PX_LAST', 'FUT_DAYS_EXP'])
        print(df)

    elif local_test == LocalTests.FUNDAMENTALS:
        # df = fetch_fundamentals(tickers=['AAPL US Equity', 'BAC US Equity'],
        #                        fields=['Security_Name', 'GICS_Sector_Name', 'CRNCY'])
        df = fetch_fundamentals(tickers=['HAHYIM2 HK Equity'],
                                fields=['name', 'front_load', 'back_load', 'fund_mgr_stated_fee',
                                        'fund_min_invest'])
        print(df)

    elif local_test == LocalTests.ACTIVE_FUTURES:
        field_data = fetch_active_futures(generic_ticker='ES1 Index')
        print(field_data)

    elif local_test == LocalTests.CONTRACT_TABLE:
        df = fetch_futures_contract_table(ticker="NK1 Index")
        print(df)

    elif local_test == LocalTests.IMPLIED_VOL_TIME_SERIES:
        df = fetch_vol_timeseries(ticker='SPX Index', vol_fields=[IMPVOL_FIELDS_MNY_30DAY, IMPVOL_FIELDS_MNY_60DAY,
                                                                  IMPVOL_FIELDS_MNY_3MTH, IMPVOL_FIELDS_MNY_6MTH,
                                                                  IMPVOL_FIELDS_MNY_12M])
        # df = fetch_vol_timeseries(ticker='EURUSD Curncy', vol_fields=['1M_CALL_IMP_VOL_10DELTA_DFLT', '1M_PUT_IMP_VOL_10DELTA_DFLT'])
        print(df)

    elif local_test == LocalTests.LAST_PRICES:
        fx_prices = fetch_last_prices()
        print(fx_prices)

    elif local_test == LocalTests.BOND_INFO:
        # data = fetch_bonds_info()
        # print(data)

        data = fetch_bonds_info(isins=['EI198784'],
                                            fields=['id_bb', 'name', 'security_des',
                                                                 'ult_parent_ticker_exchange', 'crncy',
                                                                 'amt_outstanding',
                                                                 'px_last',
                                                                 'yas_bond_yld', 'yas_oas_sprd', 'yas_mod_dur'])
        print(data)

    elif local_test == LocalTests.CDS_INFO:
        data = fetch_cds_info()
        print(data)

    elif local_test == LocalTests.BALANCE_DATA:
        data = fetch_balance_data(tickers=['ABI BB Equity', 'T US Equity', 'JPM US Equity', 'BAC US Equity'])
        print(data)

    elif local_test == LocalTests.TICKERS_FROM_ISIN:
        df = fetch_tickers_from_isins()
        print(df)

    elif local_test == LocalTests.DIVIDEND:
        this = fetch_dividend_history(ticker='TIP US Equity')
        print(this)
        divs, divs_1y = fetch_div_yields(tickers=['AHYG SP Equity'])
        print(divs_1y)

    elif local_test == LocalTests.BOND_MEMBERS:
        # members = fetch_index_members_weights(index='SPCPGN Index')
        # members = fetch_index_members_weights('I31415US Index', END_DATE_OVERRIDE='20200101')
        # members = fetch_index_members_weights(index='I00182US Index')
        # members = fetch_index_members_weights('LUACTRUU Index')
        # members = fetch_index_members_weights('BEUCTRUU Index')
        members = fetch_index_members_weights('H04064US Index')

        print(members)

        fields = ['id_bb', 'name', 'security_des',
                  'ult_parent_ticker_exchange', 'crncy',
                  'px_last',
                  'yas_bond_yld', 'yas_mod_dur', 'bb_composite']

        df = fetch_bonds_info(isins=members.index.to_list()[:10],
                              fields=fields)

        print(df)
        df.to_clipboard()

    elif local_test == LocalTests.INDEX_MEMBERS:
        # members = fetch_index_members_weights(index='URTH US Equity')
        # members = fetch_index_members_weights(index='URTH US Equity')
        members = fetch_index_members_weights(index='LG30TRUH Index')
        print(members)

    elif local_test == LocalTests.OPTION_CHAIN:
        df = bds('TSLA US Equity',
                 'CHAIN_TICKERS',
                 # CHAIN_EXP_DT_OVRD='20210917',
                 CHAIN_PUT_CALL_TYPE_OVRD='PUT',  # 'Call'
                 CHAIN_POINTS_OVRD=1000
                 )

        print(df)

    elif local_test == LocalTests.YIELD_CURVE:
        YC_US = bds("YCGT0025 Index", "INDX_MEMBERS")
        print(YC_US)
        YC_US_VAL = bdp(YC_US['member_ticker_and_exchange_code'].tolist(),
                        ['YLD_YTM_ASK', 'SECURITY NAME', 'MATURITY'])
        YC_US_VAL.maturity = pd.to_datetime(YC_US_VAL.maturity)
        YC_US_VAL["Yr"] = (YC_US_VAL.maturity - pd.to_datetime(date.today())) / np.timedelta64(365, 'D')
        YC_US_VAL = YC_US_VAL.sort_values(by=["Yr"])

        print(YC_US_VAL)

    elif local_test == LocalTests.CHECK:
        #this = bds("LUACTRUU Index", "INDX_MEMBERS3")
        #members = bds("IBOXIG Index", 'INDX_MWEIGHT')
        #print(this)
        #print(members)
        # this = bds("AAPL US Equity", "BCHAIN")
        # print(this)
        df = fetch_issuer_isins_from_bond_isins()
        print(df)

    elif local_test == LocalTests.MEMBERS:
        index = 'H04064US Index'
        members = bds(index, 'INDX_MEMBERS3')  # , overrides=[('DISPLAY_ID_BB_GLOBAL_OVERRIDE', True)]
        print(members)
        fields = ['id_bb', 'name', 'security_des',
                  'ult_parent_ticker_exchange', 'crncy',
                  'px_last',
                  'yas_bond_yld', 'yas_mod_dur', 'bb_composite']

        df = fetch_bonds_info(isins=members.iloc[:, 0].to_list(),
                              fields=fields)
        print(members)
        print(df)


if __name__ == '__main__':
    #for local_test in LocalTests:
    #    print(local_test)
    #    run_local_test(local_test=local_test)
    run_local_test(local_test=LocalTests.FIELD_TIMESERIES_PER_TICKERS)
