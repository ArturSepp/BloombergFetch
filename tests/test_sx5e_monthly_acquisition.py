"""Terminal-free monthly sampling contract."""
import pandas as pd
from examples.fetch_sx5e_monthly import contract_metadata, exchange_day, monthly_dates


def test_month_end_excludes_exchange_closures_and_missing_observations():
    dates=pd.to_datetime(["2018-12-27","2018-12-28","2018-12-31","2019-01-30","2019-01-31"])
    prices=pd.Series([1.,2.,3.,4.,float("nan")],index=dates)
    assert monthly_dates(prices)==["2018-12-28","2019-01-30"]


def test_easter_and_weekends_are_not_regular_exchange_days():
    assert not exchange_day("2024-03-29")
    assert not exchange_day("2024-04-01")
    assert not exchange_day("2024-03-30")
    assert exchange_day("2024-03-28")


def test_missing_product_code_is_flagged_but_a_wrong_code_is_rejected():
    frame=pd.DataFrame({"opt_expire_dt":["2013-06-21"],"crncy":["EUR"],"opt_undl_ticker":["SX5E"],
                        "opt_exer_typ":["European"],"exch_code":["EUX"],"id_exch_symbol":[None]})
    expiry=pd.Timestamp("2013-06-21")
    assert contract_metadata(frame,expiry,1)==(True,False)
    frame.loc[0,"id_exch_symbol"]="OEXP"
    assert contract_metadata(frame,expiry,1)==(False,False)
    frame.loc[0,"id_exch_symbol"]="OESX"
    assert contract_metadata(frame,expiry,1)==(True,True)
    assert not contract_metadata(frame.iloc[:0],expiry,0)[0]
