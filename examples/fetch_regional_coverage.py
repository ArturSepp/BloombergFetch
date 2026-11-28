"""Private Bloomberg coverage probe, not a calibrated regional ERP study."""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import pandas as pd


class Locals(Enum):
    """Isolated provider operation; run in the external Bloomberg environment."""

    FETCH = "fetch"


def run_local(local=Locals.FETCH):
    """Retrieve observed smiles and forwards without introducing synthetic carry."""
    import bbg_fetch as bbg

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2026-11-18")
    args = p.parse_args()
    if local != Locals.FETCH:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    end = pd.Timestamp(args.end)
    try:
        for ticker in ("SX5E Index", "UKX Index", "NKY Index"):
            record = {"ticker": ticker, "start": args.start, "end": str(end.date())}
            try:
                mapping = bbg.IMPVOL_FIELDS_MNY_12M
                iv = (
                    bbg.fetch_fields_timeseries_per_ticker(
                        ticker=ticker,
                        fields=list(mapping),
                        start_date=pd.Timestamp(args.start),
                        end_date=end,
                        CshAdjNormal=False,
                        CshAdjAbnormal=False,
                        CapChg=False,
                    ).rename(columns=mapping)
                    * 0.01
                )
                forward = bbg.bdh(
                    ticker,
                    "IVOL_IMPLIED_FORWARD",
                    start_date=args.start,
                    end_date=end,
                    IVOL_MATURITY="MATURITY_360D",
                )
                forward.columns = ["forward_360d"]
                underlying = bbg.fetch_fields_timeseries_per_ticker(
                    ticker=ticker,
                    fields=["PX_LAST", "TOT_RETURN_INDEX_GROSS_DVDS", "EQY_DVD_YLD_12M"],
                    start_date=pd.Timestamp(args.start),
                    end_date=end,
                    CshAdjNormal=False,
                    CshAdjAbnormal=False,
                    CapChg=False,
                )
                frame = pd.concat([iv, forward, underlying], axis=1, sort=True).sort_index()
                path = args.output / (ticker.split()[0].lower() + "_360d.csv")
                frame.to_csv(path)
                complete = frame.dropna(subset=list(mapping.values()) + ["forward_360d", "PX_LAST"])
                record.update(
                    status="retrieved",
                    rows=len(frame),
                    complete_smiles=len(complete),
                    first_complete=str(complete.index.min()),
                    last_complete=str(complete.index.max()),
                    nonmissing=frame.notna().sum().to_dict(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            except Exception as exc:
                record.update(status="failed", error=repr(exc))
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.output / "probe_manifest.json").write_text(
                json.dumps(
                    {
                        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                        "markets": records,
                        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "status": "coverage only; no regional convention, settlement, discount or valuation contract accepted",
                        "moneyness": "provider fixed spot-moneyness fields; not forward moneyness",
                        "forward": "provider IVOL_IMPLIED_FORWARD with MATURITY_360D, never synthetic carry",
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        bbg.disconnect()


if __name__ == "__main__":
    run_local(local=Locals.FETCH)
