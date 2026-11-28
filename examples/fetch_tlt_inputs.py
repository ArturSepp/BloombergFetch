"""Acquire private TLT price, total-return, cash-rate and distribution inputs."""
import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path


class Locals(Enum):
    """Available provider operation."""
    FETCH = "fetch"


def run_local(local=Locals.FETCH):
    """Save unadjusted prices separately from total returns and dated dividends."""
    import bbg_fetch as bbg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', default='2005-01-01')
    parser.add_argument('--end', default='2026-11-18')
    args = parser.parse_args()
    if local != Locals.FETCH:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise ValueError('Inputs already frozen; choose a new directory')
    try:
        metadata = bbg.bdp(['TLT US Equity', 'FEDL01 Index'],
                           ['NAME', 'CRNCY', 'SECURITY_TYP', 'FUND_TYP'])
        metadata.to_csv(args.output/'metadata.csv')
        assert metadata.loc['TLT US Equity', 'crncy'] == 'USD'
        for label, ticker, field in [('spot', 'TLT US Equity', 'PX_LAST'),
                                     ('total_return', 'TLT US Equity',
                                      'TOT_RETURN_INDEX_GROSS_DVDS'),
                                     ('cash_rate', 'FEDL01 Index', 'PX_LAST')]:
            data = bbg.fetch_field_timeseries_per_tickers(
                tickers=[ticker], field=field, start_date=args.start, end_date=args.end,
                CshAdjNormal=False, CshAdjAbnormal=False)
            data.to_csv(args.output/f'{label}.csv')
            print(label, len(data), flush=True)
        dividends = bbg.fetch_dividend_history('TLT US Equity')
        dividends.to_csv(args.output/'distributions.csv', index=False)
        manifest = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
                    'start': args.start, 'end': args.end,
                    'discount': 'lagged effective Fed funds held flat: proxy, not OIS zero curve',
                    'distributions': 'current-vintage declared/ex/pay dates; no assumed vintage proof',
                    'source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(args.output.glob('*.csv'))}}
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',
                                                encoding='utf-8')
        print('dividend rows', len(dividends), flush=True)
    finally:
        bbg.disconnect()


if __name__ == '__main__':
    run_local(local=Locals.FETCH)
