"""Calendar-selected historical SX5E listings and near-year bid/ask availability.

SINGLE_DATE_OVERRIDE follows the MathWorks Datafeed Toolbox support example.
Expiry-only requests are not used to reconstruct a historical universe.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path

import bbg_fetch as bbg
import pandas as pd


class Locals(Enum):
    """Available data-acquisition operations."""

    PILOT = "pilot"
    CURVES = "curves"


def read_listing(path):
    """An empty cached response remains an empty universe on subsequent runs."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def short_curve_tickers(frame):
    """Return only named short-tenor instruments; some curve nodes have no ticker."""
    selected = frame[frame.tenor.isin(["1D", "1M", "6M", "9M", "11M", "12M", "1Y"])]
    tickers = selected.tenor_ticker.dropna().astype(str).str.strip()
    return tickers[tickers.ne("")].drop_duplicates().tolist()


def acquire_curves(output, dates):
    """Archive historical EUR OIS par nodes and metadata, without claiming zero rates.

    Generic MATURITY can remain current-dated despite REFERENCE_DATE. Consumers
    must not use it to build historical schedules without separate verification.
    """
    output.mkdir(parents=True, exist_ok=True)
    records = []
    fields = ["NAME", "CRNCY", "SECURITY_TYP", "DAY_CNT_DES", "SW_PAY_FREQ", "SW_RCV_FREQ",
              "SW_PAY_DAY_CNT", "SW_RCV_DAY_CNT", "DAYS_TO_SETTLE", "SETTLE_DT", "MATURITY"]
    try:
        for date in dates:
            frame = bbg.bds("YCSW0133 Index", "CURVE_TENOR_RATES", CURVE_DATE=date.replace("-", ""))
            frame.to_csv(output / f"ois_curve_{date}.csv", index=False)
            tickers = short_curve_tickers(frame)
            metadata = (bbg.bdp(tickers, fields, REFERENCE_DATE=date.replace("-", ""))
                        if tickers else pd.DataFrame())
            metadata.to_csv(output / f"ois_contracts_{date}.csv")
            record = {"date": date, "rows": len(frame), "requested_tickers": tickers,
                      "last_updates": frame.last_update.astype(str).unique().tolist(),
                      "exact_zero_curve_verified": False}
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        bbg.disconnect()
    manifest = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "curve": "YCSW0133 Index", "field": "CURVE_TENOR_RATES",
                "override": "CURVE_DATE", "records": records}
    (output / "curve_acquisition.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


def nearest_expiry(expiries, date, target=365):
    """Select actual expiry before quote filtering; earlier expiry breaks a tie."""
    values = pd.to_datetime(expiries, errors="coerce").dropna().unique()
    eligible = [pd.Timestamp(x) for x in values if pd.Timestamp(x) > pd.Timestamp(date)]
    if not eligible:
        raise ValueError("No verified positive expiry in historical listing")
    return min(eligible, key=lambda x: (abs((x-pd.Timestamp(date)).days-target), x))


def acquire(output, dates):
    """Cache immutable expiry metadata and save every missing/invalid pilot observation."""
    output.mkdir(parents=True, exist_ok=True)
    config = {"dates": dates, "underlying": "SX5E Index", "field": "OPT_CHAIN",
              "override": "SINGLE_DATE_OVERRIDE", "target_days": 365,
              "selection": "closest positive metadata expiry; earlier tie; before bid/ask filtering",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = output / "historical_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Changed pilot design; choose a new output directory")
    else:
        path.write_text(json.dumps({"recorded_utc": datetime.now(timezone.utc).isoformat(), "config": config}, indent=2)+"\n", encoding="utf-8")
    records = []
    expiry_cache = pd.DataFrame()
    cache_path = output / "expiry_metadata.csv"
    if cache_path.exists():
        expiry_cache = pd.read_csv(cache_path, index_col=0)
    try:
        for date in dates:
            listing_path = output / f"listing_{date}.csv"
            if listing_path.exists():
                chain = read_listing(listing_path)
            else:
                chain = bbg.bds("SX5E Index", "OPT_CHAIN", SINGLE_DATE_OVERRIDE=date.replace("-", ""))
                chain.to_csv(listing_path, index=False)
            record = {"date": date, "listing_rows": len(chain), "possibly_truncated": len(chain) >= 8000}
            if chain.empty:
                record["status"] = "no_historical_listings"
            else:
                tickers = chain.iloc[:, 0].dropna().unique().tolist()
                missing = [t for t in tickers if t not in expiry_cache.index]
                for start in range(0, len(missing), 250):
                    result = bbg.bdp(missing[start:start+250], "OPT_EXPIRE_DT")
                    expiry_cache = pd.concat([expiry_cache, result])
                    expiry_cache.to_csv(cache_path)
                expiry = nearest_expiry(expiry_cache.reindex(tickers).opt_expire_dt, date)
                actual = pd.to_datetime(expiry_cache.reindex(tickers).opt_expire_dt, errors="coerce")
                selected = actual[actual == expiry].index.tolist()
                record.update(expiry=str(expiry.date()), days=(expiry-pd.Timestamp(date)).days,
                              selected_contracts=len(selected), expiry_metadata_missing=int(actual.isna().sum()))
                metadata_path = output / f"metadata_{date}.csv"
                if metadata_path.exists():
                    metadata = pd.read_csv(metadata_path, index_col=0)
                else:
                    fields = ["OPT_EXPIRE_DT", "OPT_PUT_CALL", "OPT_STRIKE_PX", "CRNCY",
                              "OPT_UNDL_TICKER", "OPT_EXER_TYP", "SECURITY_DES"]
                    metadata = pd.concat([bbg.bdp(selected[i:i+100], fields) for i in range(0, len(selected), 100)])
                    metadata.to_csv(metadata_path)
                verified = (pd.to_datetime(metadata.opt_expire_dt).eq(expiry).all()
                            and metadata.crncy.eq("EUR").all()
                            and metadata.opt_undl_ticker.eq("SX5E").all()
                            and metadata.opt_exer_typ.eq("European").all())
                record["metadata_verified"] = bool(verified)
                quote_path = output / f"quotes_{date}.csv"
                if quote_path.exists():
                    quotes = pd.read_csv(quote_path, header=[0, 1], index_col=0).dropna(how="all")
                else:
                    quotes = pd.concat([bbg.bdh(selected[i:i+100], ["PX_BID", "PX_ASK", "PX_LAST"],
                        start_date=date, end_date=date) for i in range(0, len(selected), 100)], axis=1)
                    quotes.to_csv(quote_path)
                record.update(quote_rows=len(quotes), status="acquired" if verified else "metadata_rejected")
            records.append(record)
            pd.DataFrame(records).to_csv(output / "historical_availability.csv", index=False)
            print(json.dumps(record), flush=True)
        # Probe existing candidate curve identities. Returned currency/name is evidence;
        # neither the ticker pattern nor a returned yield is treated as a discount curve.
        curves = ["YCSW0133 Index", "YCSW0045 Index", "EESWE1 Curncy"]
        bbg.bdp(curves, ["NAME", "CRNCY", "SECURITY_TYP", "DAY_CNT_DES"]).to_csv(output / "curve_metadata.csv")
        for date in (dates[0], dates[len(dates)//2], dates[-1]):
            for ticker in curves[:2]:
                curve = bbg.bds(ticker, "CURVE_TENOR_RATES", CURVE_DATE=date.replace("-", ""))
                curve.to_csv(output / f"curve_{ticker.split()[0]}_{date}.csv", index=False)
                print("curve", ticker, date, len(curve), list(curve.columns), flush=True)
        bbg.bdh(["SX5E Index", "SX5GT Index", "EESWE1 Curncy"], "PX_LAST",
                start_date=dates[0], end_date=dates[-1]).to_csv(output / "spot_total_ois.csv")
    finally:
        bbg.disconnect()


def run_local(local=Locals.PILOT):
    """Fetch a finite, explicit pilot date list with the repository external environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--action", choices=[case.value for case in Locals], default=local.value)
    args = parser.parse_args()
    if args.action == Locals.PILOT.value:
        acquire(args.output, sorted(args.dates))
    elif args.action == Locals.CURVES.value:
        acquire_curves(args.output, sorted(args.dates))


if __name__ == "__main__":
    run_local(local=Locals.PILOT)
