"""Freeze historical holdings duration and Treasury curves for the TLT CMA study."""
import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path

import pandas as pd


class Locals(Enum):
    """Available provider operation."""
    FETCH = 'fetch'


def acquire(selection, spot, output):
    """Use prior observed trading dates and documented holdings-analytics overrides."""
    import bbg_fetch as bbg

    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Frozen inputs already exist; choose a new directory')
    dates = pd.to_datetime(pd.read_csv(selection).date).dropna().sort_values().unique()
    trading = pd.read_csv(spot, index_col=0, parse_dates=True).dropna().index.sort_values()
    requested = [(pd.Timestamp(date), trading[trading < date][-1]) for date in dates]
    fields = ['HB_OAS_EFF_DURTN', 'HB_OAS_EFF_DURTN_CVRG_RATIO', 'HB_ANALYTICS_AS_OF_DT']
    observations = []
    try:
        bbg.bdp(['TLT US Equity', 'USGG7YR Index', 'USGG10YR Index', 'USGG20YR Index',
                 'USGG30YR Index', 'S0042FS 5Y5Y BLC Curncy'],
                ['NAME', 'CRNCY', 'SECURITY_TYP']).to_csv(output/'metadata.csv')
        for date, prior in requested:
            frame = bbg.bdp('TLT US Equity', fields,
                            HB_ANALYTICS_AS_OF_DT=prior.strftime('%Y%m%d'))
            frame.to_csv(output/f'duration_{date:%Y%m%d}.csv')
            row = frame.loc['TLT US Equity']
            actual = pd.Timestamp(row.hb_analytics_as_of_dt)
            duration = float(row.hb_oas_eff_durtn)
            coverage = float(row.hb_oas_eff_durtn_cvrg_ratio)
            if actual != prior or not 5 < duration < 30 or coverage < .98:
                raise ValueError(f'Historical duration metadata/coverage failed at {date}')
            observations.append({'date': date, 'requested_asof': prior, 'returned_asof': actual,
                                 'duration': duration, 'coverage': coverage})
        pd.DataFrame(observations).to_csv(output/'historical_duration.csv', index=False)
        start = (min(dates)-pd.Timedelta(days=130)).strftime('%Y-%m-%d')
        end = max(dates).strftime('%Y-%m-%d')
        fields = ['WEIGHTED_AVERAGE_MATURITY_YEARS', 'FUND_EXPENSE_RATIO']
        bbg.fetch_fields_timeseries_per_ticker('TLT US Equity', fields,
            start_date=start, end_date=end, CshAdjNormal=False, CshAdjAbnormal=False).to_csv(
                output/'fund_history.csv')
        tickers = ['USGG3YR Index', 'USGG7YR Index', 'USGG10YR Index',
                   'USGG20YR Index', 'USGG30YR Index', 'S0042FS 5Y5Y BLC Curncy']
        bbg.fetch_field_timeseries_per_tickers(tickers, start_date=start, end_date=end,
            CshAdjNormal=False, CshAdjAbnormal=False).to_csv(output/'curve_history.csv')
        record = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
                  'duration': 'Holdings Based OAS Effective Duration; years; prior observed trading date',
                  'coverage': 'fraction, minimum 0.98',
                  'duration_date': 'HB_ANALYTICS_AS_OF_DT backdates to latest available portfolio holdings; not an archived publication vintage',
                  'maturity': 'WEIGHTED_AVERAGE_MATURITY_YEARS; years',
                  'fee': 'FUND_EXPENSE_RATIO; percent per annum',
                  'curves': 'PX_LAST; percent yields, current vendor vintage',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'input_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [selection, spot]},
                  'hashes': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(output.glob('*.csv'))}}
        (output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print('Saved', len(observations), 'historical durations; range',
              min(row['duration'] for row in observations), max(row['duration'] for row in observations))
    finally:
        bbg.disconnect()


def run_local(local=Locals.FETCH):
    """Acquire a new private snapshot through the existing Bloomberg adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--spot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if local == Locals.FETCH:
        acquire(args.selection, args.spot, args.output)


if __name__ == '__main__':
    run_local(local=Locals.FETCH)
