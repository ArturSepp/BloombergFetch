"""Point-in-time SX5E near-year monthly chain acquisition, with raw carry inputs.

Research example: daily closing bid/ask timestamps are not asserted by this API.
Historical universes use SINGLE_DATE_OVERRIDE, never today's listing universe.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import shutil

import bbg_fetch as bbg
import pandas as pd
from dateutil.easter import easter

if __package__:
    from .fetch_sx5e_historical_pilot import nearest_expiry, read_listing
else:
    from fetch_sx5e_historical_pilot import nearest_expiry, read_listing


class Locals(Enum):
    """Finite acquisition operation."""

    MONTHLY = "monthly"


def exchange_day(date):
    """Regular OESX exchange day for the 2012-onward monthly study calendar.

    Source: Eurex trading calendars; 24 and 31 December are excluded even when
    an underlying index observation is available. Unexpected closures remain
    missing acquisitions rather than being backfilled from another month.
    """
    date = pd.Timestamp(date)
    sunday = pd.Timestamp(easter(date.year))
    holidays = {(1, 1), (5, 1), (12, 24), (12, 25), (12, 26), (12, 31)}
    return (date.weekday() < 5 and (date.month, date.day) not in holidays
            and date not in (sunday-pd.Timedelta(days=2), sunday+pd.Timedelta(days=1)))


def monthly_dates(series):
    """Select the last observed regular exchange day before inspecting any options."""
    available = series.dropna().sort_index()
    available = available[[exchange_day(x) for x in available.index]]
    return [str(pd.Timestamp(x).date()) for x in available.groupby(available.index.to_period("M")).tail(1).index]


def contract_metadata(metadata, selected_expiry, count):
    """Separate verified basic metadata from absent archived exchange-product codes."""
    symbols=metadata.id_exch_symbol.dropna().astype(str).str.strip()
    valid=(len(metadata)==count and count>0
           and pd.to_datetime(metadata.opt_expire_dt).eq(selected_expiry).all()
           and metadata.crncy.eq("EUR").all() and metadata.opt_undl_ticker.eq("SX5E").all()
           and metadata.opt_exer_typ.eq("European").all() and metadata.exch_code.eq("EUX").all()
           and symbols.eq("OESX").all())
    return bool(valid),bool(len(symbols)==count and symbols.eq("OESX").all())


def acquire(output, start, end, seed_cache=None, reuse_raw=None):
    """Save all requested months and rejection reasons, with a frozen source identity."""
    output.mkdir(parents=True, exist_ok=True)
    config = {"start": start, "end": end, "universe": "SX5E Index",
              "selection": "last observed regular exchange day monthly; closest positive expiry to 365 days before quotes",
              "historical_universe_override": "SINGLE_DATE_OVERRIDE",
              "quote_clock_verified": False, "exact_zero_curve_verified": False,
              "absent_exchange_product_code": "retain as unverified; reject any observed non-OESX code",
              "reused_csv_hashes": {} if reuse_raw is None else {
                  p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(reuse_raw.glob("*.csv"))},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest = output/"acquisition_config.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Changed acquisition source/design; choose a new output directory")
    else:
        manifest.write_text(json.dumps({"recorded_utc": datetime.now(timezone.utc).isoformat(), "config": config}, indent=2)+"\n", encoding="utf-8")
    if reuse_raw is not None:
        for path in reuse_raw.glob("*.csv"):
            if (path.name=="index_history.csv" or path.name.startswith(("listing_","metadata_","quotes_","ois_curve_"))) and not (output/path.name).exists():
                shutil.copyfile(path,output/path.name)
    cache_path = output/"expiry_metadata.csv"
    expiry = pd.read_csv(cache_path, index_col=0) if cache_path.exists() else (
        pd.read_csv(seed_cache, index_col=0) if seed_cache else pd.DataFrame())
    records = []
    history_path = output/"index_history.csv"
    try:
        if not history_path.exists():
            bbg.bdh(["SX5E Index", "SX5GT Index"], "PX_LAST", start_date=start, end_date=end).to_csv(history_path)
        history = pd.read_csv(history_path, header=[0, 1], index_col=0).dropna(how="all")
        history.index = pd.to_datetime(history.index)
        dates = monthly_dates(history[("SX5E Index", "PX_LAST")])
        pd.DataFrame({"date": dates}).to_csv(output/"selected_dates.csv", index=False)
        for date in dates:
            record = {"date": date, "status": "not_acquired"}
            try:
                path = output/f"listing_{date}.csv"
                if path.exists():
                    chain = read_listing(path)
                else:
                    chain = bbg.bds("SX5E Index", "OPT_CHAIN", SINGLE_DATE_OVERRIDE=date.replace("-", ""))
                    chain.to_csv(path, index=False)
                record["listing_rows"] = len(chain)
                if chain.empty or len(chain) >= 8000:
                    raise ValueError("Empty or possibly truncated historical listing")
                tickers = chain.iloc[:, 0].dropna().unique().tolist()
                missing = [t for t in tickers if t not in expiry.index]
                for i in range(0, len(missing), 250):
                    expiry = pd.concat([expiry, bbg.bdp(missing[i:i+250], "OPT_EXPIRE_DT")])
                    expiry.to_csv(cache_path)
                values = pd.to_datetime(expiry.reindex(tickers).opt_expire_dt, errors="coerce")
                record["missing_expiry_metadata"] = int(values.isna().sum())
                if values.isna().any():
                    raise ValueError("Unknown expiry prevents auditable nearest-expiry selection")
                selected_expiry = nearest_expiry(values, date)
                selected = values[values.eq(selected_expiry)].index.tolist()
                record.update(expiry=str(selected_expiry.date()), days=(selected_expiry-pd.Timestamp(date)).days,
                              selected_contracts=len(selected))
                path = output/f"metadata_{date}.csv"
                if path.exists():
                    metadata = pd.read_csv(path, index_col=0)
                else:
                    fields = ["OPT_EXPIRE_DT", "OPT_PUT_CALL", "OPT_STRIKE_PX", "CRNCY", "OPT_UNDL_TICKER",
                              "OPT_EXER_TYP", "SECURITY_DES", "EXCH_CODE", "ID_EXCH_SYMBOL"]
                    metadata = pd.concat([bbg.bdp(selected[i:i+100], fields) for i in range(0, len(selected), 100)])
                    metadata.to_csv(path)
                verified,symbol_observed = contract_metadata(metadata,selected_expiry,len(selected))
                record["metadata_verified"] = bool(verified)
                record["oesx_symbol_observed"] = symbol_observed
                if not verified:
                    raise ValueError("Historical slice does not match the OESX contract")
                path = output/f"quotes_{date}.csv"
                if not path.exists():
                    quotes = pd.concat([bbg.bdh(selected[i:i+100], ["PX_BID", "PX_ASK", "PX_LAST", "PX_SETTLE"],
                        start_date=date, end_date=date) for i in range(0, len(selected), 100)], axis=1)
                    quotes.to_csv(path)
                path = output/f"ois_curve_{date}.csv"
                if not path.exists():
                    bbg.bds("YCSW0133 Index", "CURVE_TENOR_RATES", CURVE_DATE=date.replace("-", "")).to_csv(path, index=False)
                record["status"] = "acquired"
            except (ValueError, KeyError, TimeoutError) as exc:
                record.update(status="not_admitted", reason=str(exc))
            records.append(record)
            pd.DataFrame(records).to_csv(output/"availability.csv", index=False)
            print(json.dumps(record), flush=True)
    finally:
        bbg.disconnect()


def run_local(local=Locals.MONTHLY):
    """Run an explicit calendar interval using the external provider environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--seed-expiry-cache", type=Path)
    parser.add_argument("--reuse-raw", type=Path)
    args = parser.parse_args()
    if local == Locals.MONTHLY:
        acquire(args.output, args.start, args.end, args.seed_expiry_cache,args.reuse_raw)


if __name__ == "__main__":
    run_local(local=Locals.MONTHLY)
