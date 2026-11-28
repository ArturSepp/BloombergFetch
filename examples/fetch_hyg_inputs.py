"""Freeze historical monthly HYG chains and dated economic inputs (private example)."""
import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path

import pandas as pd


class Locals(Enum):
    """Available acquisition operation."""
    FETCH = 'fetch'


def acquire(selection, output, target_days=30, min_days=14, max_days=50, option_periods="A"):
    """Select the nearest positive listed expiry before looking at its quotes."""
    import bbg_fetch as bbg
    output.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(pd.read_csv(selection).date).dropna().sort_values().unique()
    config = {'ticker': 'HYG US Equity', 'dates': [str(pd.Timestamp(d).date()) for d in dates],
              'expiry_target_days': target_days, 'expiry_window_days': [min_days, max_days],
              'quote_clock_verified': False, 'bid_ask_sizes_available': False,
              'option_periods': option_periods,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = output/'acquisition_config.json'
    if path.exists() and json.loads(path.read_text())['config'] != config:
        raise ValueError('Acquisition design changed; use a new output directory')
    if not path.exists():
        path.write_text(json.dumps({'recorded_utc': datetime.now(timezone.utc).isoformat(),
                                    'config': config}, indent=2)+'\n')
    observations, records = [], []
    cache = output/'expiry_metadata.csv'
    expiry = pd.read_csv(cache, index_col=0) if cache.exists() else pd.DataFrame()
    try:
        for label, ticker, field in [('spot', 'HYG US Equity', 'PX_LAST'),
                ('total_return', 'HYG US Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'),
                ('cash_rate', 'FEDL01 Index', 'PX_LAST')]:
            if not (output/f'{label}.csv').exists():
                bbg.fetch_field_timeseries_per_tickers([ticker], field=field,
                    start_date='2007-01-01', end_date='2026-11-18',
                    CshAdjNormal=False, CshAdjAbnormal=False).to_csv(output/f'{label}.csv')
        if not (output/'distributions.csv').exists():
            bbg.fetch_dividend_history('HYG US Equity').to_csv(output/'distributions.csv', index=False)
        if not (output/'fund_history.csv').exists():
            bbg.fetch_fields_timeseries_per_ticker('HYG US Equity',
                ['YAS_BOND_YLD', 'WEIGHTED_AVERAGE_MATURITY_YEARS', 'FUND_NET_ASSET_VAL'],
                start_date='2023-01-01', end_date='2026-11-18',
                CshAdjNormal=False, CshAdjAbnormal=False).to_csv(output/'fund_history.csv')
        if not (output/'curve_history.csv').exists():
            bbg.fetch_field_timeseries_per_tickers(['USGG3YR Index','USGG5YR Index'],
                start_date='2023-01-01', end_date='2026-11-18',
                CshAdjNormal=False, CshAdjAbnormal=False).to_csv(output/'curve_history.csv')
        trading = pd.read_csv(output/'spot.csv', index_col=0, parse_dates=True).dropna().index
        for value in dates:
            date = pd.Timestamp(value)
            label = str(date.date())
            record = {'date': label, 'status': 'not_acquired'}
            try:
                path = output/f'listing_{label}.csv'
                if not path.exists():
                    bbg.bds('HYG US Equity', 'OPT_CHAIN',
                            SINGLE_DATE_OVERRIDE=date.strftime('%Y%m%d'),
                            OPTION_CHAIN_OVERRIDE=option_periods).to_csv(path, index=False)
                listing = pd.read_csv(path)
                record['listing_rows'] = len(listing)
                if listing.empty or len(listing) >= 8000:
                    raise ValueError('Empty or possibly truncated historical universe')
                tickers = listing.iloc[:, 0].dropna().unique().tolist()
                missing = [t for t in tickers if t not in expiry.index]
                for i in range(0, len(missing), 250):
                    expiry = pd.concat([expiry, bbg.bdp(missing[i:i+250], 'OPT_EXPIRE_DT')])
                    expiry.to_csv(cache)
                values = pd.to_datetime(expiry.reindex(tickers).opt_expire_dt, errors='coerce')
                record['missing_expiry_metadata'] = int(values.isna().sum())
                if values.isna().any():
                    raise ValueError('Unknown expiry prevents auditable nearest selection')
                future = values[values.gt(date)]
                chosen = min(future.unique(), key=lambda x: (abs((pd.Timestamp(x)-date).days-target_days), x))
                chosen = pd.Timestamp(chosen)
                contracts = values[values.eq(chosen)].index.tolist()
                record.update(expiry=str(chosen.date()), days=(chosen-date).days,
                              selected_contracts=len(contracts))
                if not min_days <= (chosen-date).days <= max_days:
                    raise ValueError('Nearest expiry outside configured maturity window')
                path = output/f'metadata_{label}.csv'
                if not path.exists():
                    fields = ['OPT_EXPIRE_DT','OPT_PUT_CALL','OPT_STRIKE_PX','CRNCY',
                              'OPT_UNDL_TICKER','OPT_EXER_TYP','SECURITY_DES','OPT_CONT_SIZE']
                    pd.concat([bbg.bdp(contracts[i:i+100], fields)
                               for i in range(0,len(contracts),100)]).to_csv(path)
                path = output/f'quotes_{label}.csv'
                if not path.exists():
                    quotes = pd.concat([bbg.bdh(contracts[i:i+100], ['PX_BID','PX_ASK','PX_LAST'],
                        start_date=label,end_date=label) for i in range(0,len(contracts),100)],
                        axis=1)
                    validate_quotes(quotes, date)
                    quotes.to_csv(path)
                else:
                    validate_quotes(pd.read_csv(path, header=[0, 1], index_col=0), date)
                record['status'] = 'acquired'
            except (ValueError, KeyError, TimeoutError) as exc:
                record.update(status='not_admitted', reason=str(exc))
            prior = trading[trading < date][-1]
            path = output/f'duration_{label}.csv'
            if not path.exists():
                bbg.bdp('HYG US Equity', ['HB_OAS_EFF_DURTN',
                    'HB_OAS_EFF_DURTN_CVRG_RATIO','HB_ANALYTICS_AS_OF_DT'],
                    HB_ANALYTICS_AS_OF_DT=prior.strftime('%Y%m%d')).to_csv(path)
            row = pd.read_csv(path, index_col=0).loc['HYG US Equity']
            observations.append({'date':label,'requested_asof':str(prior.date()),
                'returned_asof':row.hb_analytics_as_of_dt,'duration':row.hb_oas_eff_durtn,
                'coverage':row.hb_oas_eff_durtn_cvrg_ratio})
            records.append(record)
            pd.DataFrame(records).to_csv(output/'availability.csv', index=False)
            pd.DataFrame(observations).to_csv(output/'historical_duration.csv', index=False)
            print(json.dumps(record), flush=True)
        manifest = {'recorded_utc':datetime.now(timezone.utc).isoformat(),
                    'config':config, 'hashes': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(output.glob('*.csv'))}}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    except RuntimeError as exc:
        # Capacity/authentication failure is an acquisition failure, not no data.
        failure = {'date': locals().get('label'), 'error': str(exc),
                   'error_type': type(exc).__name__, 'resume_requires_provider_recovery': True}
        (output/'acquisition_error.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise
    finally:
        bbg.disconnect()


def validate_quotes(quotes, date):
    """Reject empty/wrong-date/all-missing snapshots before marking acquisition complete."""
    if len(quotes) != 1 or pd.Timestamp(quotes.index[0]) != pd.Timestamp(date):
        raise ValueError('No same-day quote row: acquisition is unverified, not proof of no market data')
    if not isinstance(quotes.columns, pd.MultiIndex):
        raise ValueError('Expected security/field columns')
    prices = quotes.loc[:, quotes.columns.get_level_values(1).isin(['PX_BID', 'PX_ASK'])]
    if prices.empty or not prices.notna().any().any():
        raise ValueError('No bid/ask observations returned')


def run_local(local=Locals.FETCH):
    """Acquire private data using the external BloombergFetch environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-days', type=int, default=30)
    parser.add_argument('--min-days', type=int, default=14)
    parser.add_argument('--max-days', type=int, default=50)
    parser.add_argument('--option-periods', choices=['A', 'M', 'W', 'Q'], default='A')
    args = parser.parse_args()
    if local == Locals.FETCH:
        acquire(args.selection, args.output, args.target_days, args.min_days, args.max_days, args.option_periods)


if __name__ == '__main__':
    run_local(local=Locals.FETCH)
