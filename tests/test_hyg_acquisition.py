"""HYG snapshots require observed same-date quotes, not merely a response file."""
import pandas as pd
import pytest
from examples.fetch_hyg_inputs import validate_quotes


def test_empty_snapshot_is_not_completed_acquisition():
    cols=pd.MultiIndex.from_product([['option'],['PX_BID','PX_ASK']])
    with pytest.raises(ValueError,match='acquisition is unverified'):
        validate_quotes(pd.DataFrame(columns=cols),'2023-07-31')


def test_all_missing_prices_are_not_completed_acquisition():
    cols=pd.MultiIndex.from_product([['option'],['PX_BID','PX_ASK','PX_LAST']])
    frame=pd.DataFrame([[float('nan'),float('nan'),1.]],columns=cols,index=['2023-07-31'])
    with pytest.raises(ValueError,match='No bid/ask'):
        validate_quotes(frame,'2023-07-31')


def test_quote_dates_must_match_requested_date():
    cols=pd.MultiIndex.from_product([['option'],['PX_BID','PX_ASK']])
    frame=pd.DataFrame([[.1,.12]],columns=cols,index=['2023-07-28'])
    with pytest.raises(ValueError,match='same-day'):
        validate_quotes(frame,'2023-07-31')


def test_partial_market_coverage_preserved_for_separate_quality_checks():
    cols=pd.MultiIndex.from_product([['option1','option2'],['PX_BID','PX_ASK']])
    frame=pd.DataFrame([[.1,.12,float('nan'),.02]],columns=cols,index=['2023-07-31'])
    validate_quotes(frame,'2023-07-31')
